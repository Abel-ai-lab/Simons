"""Data-parallel training with token-weighted losses and one flat gradient all-reduce."""

import os
from dataclasses import dataclass

import torch

STREAM_OFFSET = 1_000_003


@dataclass
class Rank:
    rank: int = 0
    world: int = 1
    local: int = 0

    @classmethod
    def from_env(cls, device_type: str) -> "Rank":
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world <= 1:
            return cls()
        import torch.distributed as dist

        rank, local = int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0"))
        if device_type == "cuda":
            torch.cuda.set_device(local)
        dist.init_process_group("nccl" if device_type == "cuda" else "gloo")
        return cls(rank, world, local)

    @property
    def active(self) -> bool:
        return self.world > 1

    @property
    def main(self) -> bool:
        return self.rank == 0

    def split(self, total: int) -> tuple[int, int]:
        """(offset, size) of this rank's share of `total` items per step."""
        base, extra = divmod(total, self.world)
        size = base + (1 if self.rank < extra else 0)
        if size <= 0:
            raise ValueError(f"{total} items per step cannot be split over {self.world} ranks")
        return self.rank * base + min(self.rank, extra), size

    def stream_seed(self, seed: int) -> int:
        """Return the initialization value for this rank's private random streams."""
        return seed + STREAM_OFFSET * self.rank

    def loader_workers(self, workers: int) -> int:
        cpus = os.cpu_count() or 1
        if not self.active:
            return min(workers, cpus)
        return max(2, min(workers, cpus // (2 * self.world)))

    def scale_loss(self, loss: torch.Tensor, count: torch.Tensor):
        """(objective to back-propagate, global mean loss for logging)."""
        if not self.active:
            return loss, loss.detach()
        import torch.distributed as dist

        count = count.detach().to(torch.float64)
        parts = torch.stack([loss.detach().to(torch.float64) * count, count])
        dist.all_reduce(parts)
        total = parts[1].clamp_min(1.0)
        return loss * (self.world * count / total).to(loss.dtype), (parts[0] / total).float()

    def mean_value(self, value: torch.Tensor) -> torch.Tensor:
        """The mean of a scalar over ranks (for logging; every rank must call it)."""
        if not self.active:
            return value
        import torch.distributed as dist

        total = value.detach().to(torch.float64).reshape(1).clone()
        dist.all_reduce(total)
        return (total[0] / self.world).float()

    def average_gradients(self, model):
        """Average gradients across ranks; missing local gradients count as zero.
        Leave a gradient unset if it is missing on every rank."""
        if not self.active:
            return
        import torch.distributed as dist
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        params = [p for p in model.parameters() if p.requires_grad]
        device = params[0].device
        present = torch.tensor([int(p.grad is not None) for p in params], device=device)
        dist.all_reduce(present)
        keep = []
        for p, n in zip(params, present.tolist()):
            if n > 0:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                keep.append(p)
            else:
                p.grad = None
        if not keep:
            return
        grads = [p.grad for p in keep]
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        flat.div_(self.world)
        for p, g in zip(keep, _unflatten_dense_tensors(flat, grads)):
            p.grad = g

    def broadcast_model(self, model):
        if self.active:
            import torch.distributed as dist

            for t in [*model.parameters(), *model.buffers()]:
                dist.broadcast(t.data, 0)

    def barrier(self):
        if self.active:
            import torch.distributed as dist

            dist.barrier()

    def close(self):
        if self.active:
            import torch.distributed as dist

            dist.destroy_process_group()
