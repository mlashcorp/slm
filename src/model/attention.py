import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import ModelConfig
from src.model.rope import apply_rotary_emb


def doc_ids_to_cu_seqlens(doc_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """
    Convert (B, S) doc_ids into cu_seqlens for flash_attn_varlen_func.
    Documents must be contiguous within each batch row.
    Returns (cu_seqlens, max_seqlen) where cu_seqlens is int32 of shape
    (total_docs + 1,) and max_seqlen is a plain Python int.
    """
    B, S = doc_ids.shape
    cu = [0]
    for b in range(B):
        row = doc_ids[b]
        boundaries = torch.cat([row.new_tensor([True]), row[1:] != row[:-1]])
        starts = torch.where(boundaries)[0]
        lengths = torch.diff(torch.cat([starts, row.new_tensor([S])]))
        for l in lengths.tolist():
            cu.append(cu[-1] + int(l))
    max_seqlen = max(cu[i + 1] - cu[i] for i in range(len(cu) - 1))
    return torch.tensor(cu, dtype=torch.int32, device=doc_ids.device), max_seqlen


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.use_flash = config.use_flash_attn
        assert self.n_heads % self.n_kv_heads == 0
        self.n_groups = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,                        # (batch, seq, hidden)
        cos: torch.Tensor,                      # (seq, head_dim)
        sin: torch.Tensor,                      # (seq, head_dim)
        attn_mask: torch.Tensor | None = None,  # (batch, 1, seq, seq) float additive
        cu_seqlens: torch.Tensor | None = None, # (total_docs+1,) int32 — varlen path
        max_seqlen: int | None = None,          # max doc length (Python int, avoids .item() graph break)
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, cos, sin)

        # flash_attn_varlen_func: packed sequences, O(S) memory, native GQA support
        if self.use_flash and cu_seqlens is not None:
            from flash_attn import flash_attn_varlen_func
            q_flat = q.reshape(B * S, self.n_heads, self.head_dim)
            k_flat = k.reshape(B * S, self.n_kv_heads, self.head_dim)
            v_flat = v.reshape(B * S, self.n_kv_heads, self.head_dim)
            out_flat = flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
            )
            out = out_flat.view(B, S, self.n_heads * self.head_dim)

        # flash_attn_func: no doc masking, causal only
        elif self.use_flash and attn_mask is None:
            from flash_attn import flash_attn_func
            k_exp = k.repeat_interleave(self.n_groups, dim=2)
            v_exp = v.repeat_interleave(self.n_groups, dim=2)
            out = flash_attn_func(q, k_exp, v_exp, causal=True).reshape(B, S, self.n_heads * self.head_dim)

        # SDPA fallback (used when flash_attn not available)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)
            if attn_mask is not None:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)

        return self.o_proj(out)
