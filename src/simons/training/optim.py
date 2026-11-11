"""NorMuon for hidden matrices; AdamW for embeddings, head, norms, biases and covariate modules."""

import math

import torch

MOMENTUM = 0.95
ROW_BETA2 = 0.95
ADAM_BETAS = (0.9, 0.95)
EPS = 1e-8

# Polar Express coefficients use l=1e-3 and a 1.01 safety factor.
# Amsel et al. 2025, github.com/NoahAmsel/PolarExpress
_POLAR_RAW = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
POLAR_EXPRESS = [(a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _POLAR_RAW]


@torch.no_grad()
def orthogonalize(G: torch.Tensor) -> torch.Tensor:
    """Approximate U V^T of a stack [k, m, n] of matrices."""
    X = G.bfloat16()
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    for a, b, c in POLAR_EXPRESS:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tall:
        X = X.mT
    return X


def split_parameters(model):
    """(hidden matrices with their row-block counts, all other parameters)."""
    hidden, splits = [], []
    for block in [*model.time_blocks, model.variate_block]:
        matrices = [
            (block.qkv.weight, 3),  # Orthogonalize fused q, k, v blocks separately.
            (block.out.weight, 1),
            (block.mlp[0].weight, 1),
            (block.mlp[2].weight, 1),
        ]
        for weight, blocks in matrices:
            hidden.append(weight)
            splits.append(blocks)
    hidden.append(model.embed_mlp[2].weight)
    splits.append(1)
    chosen = {id(p) for p in hidden}
    other = [p for p in model.parameters() if id(p) not in chosen]
    return hidden, splits, other


class NorMuonAdamW(torch.optim.Optimizer):
    """NorMuon (hidden matrices) + AdamW (everything else) over a Simons model."""

    def __init__(self, model, muon_lr: float = 1e-3, adam_lr: float = 1e-3,
                 weight_decay: float = 0.01, mup_base_width: int = 704):
        hidden, splits, other = split_parameters(model)
        head = {id(p) for p in model.head.parameters()}
        rest = [p for p in other if id(p) not in head]
        head_params = [p for p in other if id(p) in head]
        hidden_lr = muon_lr * math.sqrt(mup_base_width / model.config.width)
        head_lr = adam_lr * (mup_base_width / model.config.width)
        groups = [
            dict(params=hidden, splits=splits, kind="normuon", peak_lr=hidden_lr, lr=hidden_lr,
                 weight_decay=weight_decay),
            dict(params=rest, splits=[1] * len(rest), kind="adamw", peak_lr=adam_lr, lr=adam_lr,
                 weight_decay=0.0),
            dict(params=head_params, splits=[1] * len(head_params), kind="adamw", peak_lr=head_lr,
                 lr=head_lr, weight_decay=0.0),
        ]
        super().__init__(groups, {})

    @staticmethod
    def _decay_and_apply(params, updates, lr, wd):
        if wd > 0:
            mask = torch._foreach_mul(updates, params)
            torch._foreach_sign_(mask)
            torch._foreach_clamp_min_(mask, 0.0)
            torch._foreach_mul_(mask, params)
            torch._foreach_add_(updates, mask, alpha=wd)
        torch._foreach_add_(params, updates, alpha=-lr)

    def _adam(self, group, params):
        beta1, beta2 = ADAM_BETAS
        by_step = {}
        for p in params:
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            by_step.setdefault(state["step"], []).append(p)
        for step, chosen in by_step.items():
            grads = [p.grad for p in chosen]
            m = [self.state[p]["exp_avg"] for p in chosen]
            v = [self.state[p]["exp_avg_sq"] for p in chosen]
            torch._foreach_lerp_(m, grads, 1 - beta1)
            torch._foreach_mul_(v, beta2)
            torch._foreach_addcmul_(v, grads, grads, value=1 - beta2)
            denom = torch._foreach_sqrt(v)
            torch._foreach_div_(denom, math.sqrt(1 - beta2**step))
            torch._foreach_add_(denom, EPS)
            updates = torch._foreach_div(m, denom)
            torch._foreach_mul_(updates, 1 / (1 - beta1**step))
            self._decay_and_apply(chosen, updates, group["lr"], group["weight_decay"])

    def _normuon(self, group, params, splits):
        bufs, grads = [], []
        for p in params:
            state = self.state[p]
            if not state:
                state["momentum"] = torch.zeros_like(p)
                state["row_v"] = torch.zeros(p.shape[0], 1, device=p.device, dtype=torch.float32)
            bufs.append(state["momentum"])
            grads.append(p.grad)
        torch._foreach_lerp_(bufs, grads, 1 - MOMENTUM)
        nesterov = torch._foreach_lerp(grads, bufs, MOMENTUM)
        by_shape = {}
        for index, (u, blocks) in enumerate(zip(nesterov, splits)):
            pieces = u.view(blocks, u.shape[0] // blocks, u.shape[1])
            tall = pieces.shape[1] > pieces.shape[2]
            if tall:
                pieces = pieces.mT
            by_shape.setdefault(tuple(pieces.shape[1:]), []).append((index, pieces, tall))
        orthogonal = [None] * len(params)
        for items in by_shape.values():
            stack = torch.cat([pieces for _, pieces, _ in items])
            out = orthogonalize(stack).float()
            start = 0
            for index, pieces, tall in items:
                ortho = out[start : start + pieces.shape[0]]
                orthogonal[index] = ortho.mT if tall else ortho
                start += pieces.shape[0]
        updates = []
        for p, ortho in zip(params, orthogonal):  # [blocks,m,n]
            v = self.state[p]["row_v"].view(ortho.shape[0], ortho.shape[1], 1)
            v.lerp_(ortho.square().mean(-1, keepdim=True), 1 - ROW_BETA2)
            ortho = ortho / (v.sqrt() + EPS)
            m, n = ortho.shape[1:]
            scale = 0.2 * math.sqrt(m * n) / (ortho.norm(dim=(-2, -1), keepdim=True) + 1e-12)
            updates.append((ortho * scale).reshape(p.shape).to(p.dtype))
        self._decay_and_apply(params, updates, group["lr"], group["weight_decay"])

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            pairs = [(p, s) for p, s in zip(group["params"], group["splits"]) if p.grad is not None]
            if not pairs:
                continue
            params = [p for p, _ in pairs]
            if group["kind"] == "normuon":
                self._normuon(group, params, [s for _, s in pairs])
            else:
                self._adam(group, params)
