import torch
import torch.nn as nn
from src.model.config import ModelConfig
from src.model.norm import RMSNorm
from src.model.block import TransformerBlock
from src.model.rope import precompute_freqs_cos_sin
from src.model.attention import doc_ids_to_cu_seqlens


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_freqs_cos_sin(
            config.head_dim, config.max_position_embeddings, theta=config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    @staticmethod
    def build_doc_mask(doc_ids: torch.Tensor) -> torch.Tensor:
        """SDPA fallback: (B, 1, S, S) float additive mask."""
        B, S = doc_ids.shape
        same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
        causal = torch.ones(S, S, device=doc_ids.device, dtype=torch.bool).tril()
        mask = same_doc & causal.unsqueeze(0)
        float_mask = torch.zeros(B, 1, S, S, device=doc_ids.device)
        float_mask.masked_fill_(~mask.unsqueeze(1), float("-inf"))
        return float_mask

    def forward(
        self,
        input_ids: torch.Tensor,              # (batch, seq_len)
        doc_ids: torch.Tensor | None = None,  # (batch, seq_len) for intra-doc masking
    ) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos = self.rope_cos[:S]
        sin = self.rope_sin[:S]

        cu_seqlens = None
        max_seqlen = None
        attn_mask = None
        if doc_ids is not None:
            if self.config.use_flash_attn:
                cu_seqlens, max_seqlen = doc_ids_to_cu_seqlens(doc_ids)
            else:
                attn_mask = self.build_doc_mask(doc_ids)

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask=attn_mask, cu_seqlens=cu_seqlens,
                      max_seqlen=max_seqlen)

        x = self.norm(x)
        return self.lm_head(x)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (p for p in self.parameters() if not trainable_only or p.requires_grad)
        return sum(p.numel() for p in params)
