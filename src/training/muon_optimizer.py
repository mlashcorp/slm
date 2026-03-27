"""
Muon optimizer: Momentum orthogonalized by Newton-Schulz.

Based on Karpathy's nanochat implementation:
https://github.com/Karpathy/nanochat/blob/master/nanochat/optim.py

Adapted from: https://github.com/KellerJordan/modded-nanogpt

Background
==========

Newton-Schulz iteration computes the zeroth power / orthogonalization of G.
We use a quintic iteration whose coefficients maximize the slope at zero.

For minimizing steps, it's empirically effective to keep increasing the
slope at zero beyond the point where the iteration converges all the way
to one. This iteration produces something like US'V^T where S' is diagonal
with S_{ii}' ~ Uniform(0.5, 1.5), which doesn't hurt performance.

Key Components
==============

1. Nesterov Momentum: Smooth gradients with momentum
2. Polar Express: Orthogonalize update using 5 Newton-Schulz iterations
3. Variance Reduction: Per-neuron adaptive learning rate
4. Cautious Weight Decay: Only apply when gradient and param have same sign

Usage
=====

    from src.training.muon_optimizer import MuonAdamW

    param_groups = [
        # AdamW: embeddings, scalars, small params
        dict(kind='adamw', params=lm_head, lr=0.008, betas=(0.8, 0.96)),
        dict(kind='adamw', params=embeddings, lr=0.2, betas=(0.8, 0.995)),

        # Muon: matrix params (must all have same shape for stacking)
        dict(kind='muon', params=matrices, lr=0.02, momentum=0.95, beta2=0.9),
    ]

    optimizer = MuonAdamW(param_groups)
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Optional

COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32


# =============================================================================
# AdamW Fused Kernel
# =============================================================================

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step_t: Tensor,
    lr_t: Tensor,
    beta1_t: Tensor,
    beta2_t: Tensor,
    eps_t: Tensor,
    wd_t: Tensor,
) -> None:
    """Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update.

    All in one compiled graph to eliminate Python overhead between ops.
    Uses 0-D CPU tensors to avoid recompilation when hyperparameter values change.
    """
    # Weight decay (decoupled, applied before update)
    p.mul_(1 - lr_t * wd_t)

    # Update running averages
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)

    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t

    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


# =============================================================================
# Muon: Polar Express Coefficients
# =============================================================================

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.4232355116970523),
]


@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,
    stacked_params: Tensor,
    momentum_buffer: Tensor,
    second_momentum_buffer: Tensor,
    momentum_t: Tensor,
    lr_t: Tensor,
    wd_t: Tensor,
    beta2_t: Tensor,
    ns_steps: int,
    red_dim: int,
) -> None:
    """Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update.

    All in one compiled graph to eliminate Python overhead.

    Args:
        stacked_grads: (K, M, N) stacked gradients
        stacked_params: (K, M, N) stacked parameters
        momentum_buffer: (K, M, N) first moment buffer
        second_momentum_buffer: (K, M, 1) or (K, 1, N) factored second moment
        momentum_t: momentum coefficient
        lr_t: learning rate
        wd_t: weight decay
        beta2_t: beta2 for second moment
        ns_steps: number of Newton-Schulz iterations (5)
        red_dim: reduction dimension (-1 or -2)
    """
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar Express orthogonalization
    # Cast to bf16 for speed (fp16 has limited exponent range)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)

    if g.size(-2) > g.size(-1):
        # Tall matrix
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        # Wide matrix
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    g = X

    # Variance reduction (NorMuon-style)
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()

    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)

    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)

    # Only apply weight decay where gradient and param have same sign
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# =============================================================================
# MuonAdamW Optimizer (Single GPU)
# =============================================================================

class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others.

    Muon uses Newton-Schulz iteration to orthogonalize gradients, which
    can lead to faster convergence for transformer models.

    Important:
    - Muon should NOT be used for embedding layer, final linear layer, or scalars.
      Use AdamW for those.
    - All params in a Muon group must have the same shape (for stacking).

    Args:
        param_groups: List of dicts with:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """

    def __init__(self, param_groups: List[Dict]):
        super().__init__(param_groups, defaults={})

        # 0-D CPU tensors to avoid torch.compile recompilation
        # AdamW
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        # Muon
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: Dict) -> None:
        """AdamW update for each param in the group."""
        for p in group['params']:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            # Fill 0-D tensors
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            # Fused update
            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t,
                self._adamw_beta1_t, self._adamw_beta2_t,
                self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: Dict) -> None:
        """Muon update for all params in the group (stacked for efficiency)."""
        params: List[Tensor] = group['params']
        if not params:
            return

        # Filter to params with gradients
        params = [p for p in params if p.grad is not None]
        if not params:
            return

        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        num_params = len(params)

        # Get or create group-level buffers
        state = self.state[p]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)

        momentum_buffer = state["momentum_buffer"]

        if "second_momentum_buffer" not in state:
            # Factored: either per-row or per-column
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)

        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Fill 0-D tensors
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group.get("beta2", 0.9))
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        # Fused update
        muon_step_fused(
            stacked_grads, stacked_params,
            momentum_buffer, second_momentum_buffer,
            self._muon_momentum_t, self._muon_lr_t,
            self._muon_wd_t, self._muon_beta2_t,
            group.get("ns_steps", 5), red_dim,
        )

        # Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        """Perform one optimization step."""
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")


# =============================================================================
# Helper to Setup Parameter Groups
# =============================================================================

def setup_param_groups(
    model: nn.Module,
    matrix_lr: float = 0.02,
    embedding_lr: float = 0.2,
    unembedding_lr: float = 0.008,
    scalar_lr: float = 0.5,
    weight_decay: float = 0.28,
    momentum: float = 0.95,
    beta2: float = 0.9,
) -> List[Dict]:
    """Setup parameter groups for MuonAdamW optimizer.

    Separates parameters into:
    - AdamW: embeddings, lm_head, scalars, biases
    - Muon: matrix weights (2D params in transformer blocks)

    Args:
        model: NanoChat GPT model
        matrix_lr: Learning rate for Muon (matrix params)
        embedding_lr: Learning rate for embeddings
        unembedding_lr: Learning rate for lm_head
        scalar_lr: Learning rate for scalars
        weight_decay: Weight decay for Muon
        momentum: Momentum for Muon
        beta2: Beta2 for Muon variance reduction

    Returns:
        List of parameter group dicts
    """
    # Collect parameters by type
    matrix_params = []  # For Muon
    embedding_params = []
    lm_head_params = []
    value_embed_params = []
    scalar_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Skip 1D params (biases, scalars) - they go to AdamW
        if param.dim() == 1:
            scalar_params.append(param)
            continue

        # Embeddings
        if 'wte' in name or 'value_embeds' in name:
            if 'value_embeds' in name:
                value_embed_params.append(param)
            else:
                embedding_params.append(param)
            continue

        # LM head
        if 'lm_head' in name:
            lm_head_params.append(param)
            continue

        # Per-layer scalars
        if 'lambda' in name or 'gate' in name:
            scalar_params.append(param)
            continue

        # Matrix params (2D weights in transformer)
        if param.dim() == 2:
            matrix_params.append(param)

    # Group matrix params by shape (required for Muon stacking)
    matrix_param_groups = {}
    for p in matrix_params:
        shape = p.shape
        if shape not in matrix_param_groups:
            matrix_param_groups[shape] = []
        matrix_param_groups[shape].append(p)

    # Build param groups
    param_groups = []

    # AdamW groups
    if lm_head_params:
        param_groups.append(dict(
            kind='adamw',
            params=lm_head_params,
            lr=unembedding_lr,
            betas=(0.8, 0.96),
            eps=1e-10,
            weight_decay=0.01,
        ))

    if embedding_params:
        param_groups.append(dict(
            kind='adamw',
            params=embedding_params,
            lr=embedding_lr,
            betas=(0.8, 0.995),
            eps=1e-10,
            weight_decay=0.001,
        ))

    if value_embed_params:
        param_groups.append(dict(
            kind='adamw',
            params=value_embed_params,
            lr=embedding_lr * 0.5,
            betas=(0.8, 0.995),
            eps=1e-10,
            weight_decay=0.01,
        ))

    if scalar_params:
        param_groups.append(dict(
            kind='adamw',
            params=scalar_params,
            lr=scalar_lr,
            betas=(0.8, 0.95),
            eps=1e-10,
            weight_decay=0.0,
        ))

    # Muon groups (one per shape)
    for shape, params in matrix_param_groups.items():
        param_groups.append(dict(
            kind='muon',
            params=params,
            lr=matrix_lr,
            momentum=momentum,
            ns_steps=5,
            beta2=beta2,
            weight_decay=weight_decay,
        ))

    return param_groups


if __name__ == "__main__":
    # Test Muon optimizer on a simple model
    print("Testing MuonAdamW optimizer...")

    # Create simple model
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    )

    # Setup param groups
    param_groups = setup_param_groups(
        model,
        matrix_lr=0.02,
        embedding_lr=0.2,
        unembedding_lr=0.008,
        scalar_lr=0.5,
    )

    print(f"Created {len(param_groups)} parameter groups:")
    for i, group in enumerate(param_groups):
        print(f"  Group {i}: kind={group['kind']}, lr={group.get('lr', 'N/A')}, params={len(group['params'])}")

    # Create optimizer
    optimizer = MuonAdamW(param_groups)

    # Test optimization step
    x = torch.randn(32, 64)
    y = model(x).sum()
    y.backward()
    optimizer.step()

    print("Optimizer step successful!")
