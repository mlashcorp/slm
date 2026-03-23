import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from src.model.config import ModelConfig
from src.model.norm import RMSNorm
from src.model.attention import GroupedQueryAttention
from src.model.mlp import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_grad_ckpt = config.use_gradient_checkpointing
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def _forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), freqs_cis, attn_mask=attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_grad_ckpt and self.training:
            return checkpoint(self._forward, x, freqs_cis, attn_mask, use_reentrant=False)
        return self._forward(x, freqs_cis, attn_mask)
