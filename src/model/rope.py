import torch


def precompute_freqs_cis(
    head_dim: int,
    seq_len: int,
    theta: float = 10_000.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Precompute RoPE complex exponentials.
    Returns complex tensor of shape (seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)           # (seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)   # complex64


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.
    q, k:       (batch, seq_len, n_heads, head_dim)
    freqs_cis:  (seq_len, head_dim // 2) complex
    """
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)     # (1, seq_len, 1, head_dim//2)
    q_rot = torch.view_as_real(q_ * freqs).flatten(-2)
    k_rot = torch.view_as_real(k_ * freqs).flatten(-2)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)
