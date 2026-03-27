import torch


def precompute_freqs_cos_sin(
    head_dim: int,
    seq_len: int,
    theta: float = 10_000.0,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cos/sin tables using the rotate_half (split-half) convention.
    Returns (cos, sin) each of shape (seq_len, head_dim).
    """
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)            # (seq_len, head_dim // 2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors (rotate_half convention).
    q, k:       (batch, seq_len, n_heads, head_dim)
    cos, sin:   (seq_len, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(2)     # (1, seq_len, 1, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_rot = (q.float() * cos) + (_rotate_half(q.float()) * sin)
    k_rot = (k.float() * cos) + (_rotate_half(k.float()) * sin)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)
