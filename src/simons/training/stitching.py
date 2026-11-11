"""Extend training windows backward through contiguous `<base>@<offset>` segments."""

import re

import numpy as np

_ID = re.compile(r"^(.*)@(\d+)$")


def positions(item_ids, shapes):
    """Return predecessor positions per series, or -1, for IDs and shapes [n,2] (variates,steps).
    Links require matching base IDs, variate counts and contiguous offsets."""
    matches = [_ID.match(x) for x in item_ids]
    codes, offsets, keep = {}, np.zeros(len(item_ids), np.int64), []
    base = np.full(len(item_ids), -1, np.int64)
    for k, match in enumerate(matches):
        if match is None:
            continue
        base[k] = codes.setdefault(match.group(1), len(codes))
        offsets[k] = int(match.group(2))
        keep.append(k)
    prev = np.full(len(item_ids), -1, np.int64)
    keep = np.asarray(keep, np.int64)
    if len(keep) < 2:
        return prev.tolist()
    order = keep[np.lexsort((keep, offsets[keep], base[keep]))]
    a, b = order[:-1], order[1:]
    shapes = np.asarray(shapes, np.int64)
    link = ((base[a] == base[b]) & (offsets[a] + shapes[a, 1] == offsets[b])
            & (shapes[a, 0] == shapes[b, 0]))
    prev[b[link]] = a[link]
    return prev.tolist()


def extend(source_arrays, prev, position, channels, end, want):
    """Window [channels, :end] of segment `position`, extended backward through its
    predecessors until it holds `want` steps (or the run starts). Returns a float32 copy."""
    group = source_arrays[position]
    parts = [np.asarray(group[channels, :end], dtype="float32")]
    have = end
    p = prev[position]
    while have < want and p >= 0:
        before = source_arrays[p]
        take = min(want - have, before.shape[1])
        parts.append(np.asarray(before[channels, before.shape[1] - take:], dtype="float32"))
        have += take
        p = prev[p] if take == before.shape[1] else -1
    if len(parts) == 1:
        return np.array(parts[0], copy=True)
    return np.concatenate(parts[::-1], axis=1)
