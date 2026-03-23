import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.config import ModelConfig
from src.model.rope import apply_rotary_emb


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
        freqs_cis: torch.Tensor,                # (seq, head_dim // 2) complex
        attn_mask: torch.Tensor | None = None,  # (batch, 1, seq, seq) float additive
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        if self.use_flash and attn_mask is None:
            from flash_attn import flash_attn_func
            k = k.repeat_interleave(self.n_groups, dim=2)
            v = v.repeat_interleave(self.n_groups, dim=2)
            out = flash_attn_func(q, k, v, causal=True)
        else:
            # Standard SDPA — transpose to (B, H, S, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)
            if attn_mask is not None:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2)   # (B, S, H, D)

        out = out.reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)
