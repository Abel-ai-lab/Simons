"""Pack training windows to skip leading NaN padding while isolating windows with segment masks."""

import numpy as np
import torch

from simons.training.data import MAX_VARIATES


def _guillotine(heights, spans, width, patches):
    """Pack taller, then longer rectangles first, splitting free space right and below."""
    n = len(heights)
    order = np.lexsort((-spans, -heights))
    cap = 3 * n + 3  # Each window adds at most a row and two splits.
    big = np.iinfo(np.int64).max
    fr = np.zeros(cap, np.int64)
    fy = np.zeros(cap, np.int64)
    fx = np.zeros(cap, np.int64)
    fh = np.full(cap, -1, np.int64)  # Mark unused entries so they never fit.
    fw = np.full(cap, -1, np.int64)
    key = np.full(cap, big, np.int64)
    count, rows = 0, 0
    row_of = np.empty(n, np.int64)
    slot_of = np.empty(n, np.int64)
    time_of = np.empty(n, np.int64)
    stride_y, stride_r = patches + 1, (patches + 1) * (width + 1)
    dead = 0
    for i in order.tolist():
        h, w = int(heights[i]), int(spans[i])
        if dead > 64 and 2 * dead > count:  # Unique keys preserve choices during compaction.
            keep = np.flatnonzero(fh[:count] >= 0)
            m = len(keep)
            for a in (fr, fy, fx, fh, fw, key):
                a[:m] = a[keep]
            fh[m:count] = fw[m:count] = -1
            key[m:count] = big
            count, dead = m, 0
        live_h, live_w = fh[:count], fw[:count]
        j = int(np.where((live_h >= h) & (live_w >= w), key[:count], big).argmin()) if count else 0
        if not count or fh[j] < h or fw[j] < w:
            j = count
            fr[j], fy[j], fx[j], fh[j], fw[j], key[j] = rows, 0, 0, width, patches, rows * stride_r
            count += 1
            rows += 1
        r, y, x, H, W = int(fr[j]), int(fy[j]), int(fx[j]), int(fh[j]), int(fw[j])
        fh[j] = fw[j] = -1
        key[j] = big
        dead += 1
        row_of[i], slot_of[i], time_of[i] = r, y, x
        for (ny, nx, nh, nw) in ((y, x + w, h, W - w), (y + h, x, H - h, W)):
            if nh > 0 and nw > 0:
                fr[count], fy[count], fx[count], fh[count], fw[count] = r, ny, nx, nh, nw
                key[count] = r * stride_r + ny * stride_y + nx
                count += 1
    return row_of, slot_of, time_of, rows


def plan(slot_ids: np.ndarray, slot_first: np.ndarray, patches: int, width: int = MAX_VARIATES,
         row_tokens: int = 0):
    """Return (source,segment,rows) for flattened [R,W] IDs and first observed patches.
    `source`/`segment` flatten [rows,width,patches]; -1 marks filler; windows stay aligned."""
    unpacked = patches
    patches = max(patches, row_tokens)
    valid = np.flatnonzero(slot_ids >= 0)
    slots = valid[np.argsort(slot_ids[valid], kind="stable")]
    _, starts, counts = np.unique(slot_ids[slots], return_index=True, return_counts=True)
    window_first = np.minimum.reduceat(slot_first[slots], starts)
    span = unpacked - window_first
    keep = span > 0
    starts, counts, window_first, span = starts[keep], counts[keep], window_first[keep], span[keep]
    windows = len(starts)
    window_row, window_slot, time_offset, rows = _guillotine(counts, span, width, patches)
    window_base = window_row * width + window_slot
    run_window = np.repeat(np.arange(windows), counts)
    run_variate = np.arange(len(run_window)) - np.repeat(np.cumsum(counts) - counts, counts)
    run_slot = slots[np.repeat(starts, counts) + run_variate]
    packed_slot = window_base[run_window] + run_variate
    dest_start = packed_slot * patches + time_offset[run_window]
    source_start = run_slot * unpacked + window_first[run_window]
    length = span[run_window]
    total = int(length.sum())
    within = np.arange(total) - np.repeat(np.cumsum(length) - length, length)
    dest = np.repeat(dest_start, length) + within
    source = np.full(rows * width * patches, -1, dtype=np.int64)
    segment = np.full(rows * width * patches, -1, dtype=np.int64)
    source[dest] = np.repeat(source_start, length) + within
    segment[dest] = np.repeat(run_window, length)
    return source, segment, rows


def gather(tensor: torch.Tensor, source: torch.Tensor, fill=0.0) -> torch.Tensor:
    """Rows of a [N, ...] tensor at `source` (-1 -> `fill`)."""
    picked = tensor[source.clamp_min(0)]
    hole = (source < 0).view(-1, *([1] * (picked.ndim - 1)))
    return picked.masked_fill(hole, fill)
