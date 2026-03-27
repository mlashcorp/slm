"""
NanoChat model configuration — aligned with karpathy/nanogpt GPTConfig.

Reference: https://github.com/karpathy/nanogpt
"""
from dataclasses import dataclass


@dataclass
class NanoChatConfig:
    """GPT configuration, matching karpathy/nanogpt field names.

    Preset configs use our smaller sizes designed for fast training
    on a single GPU, not the full GPT-2 sizes.
    """
    block_size: int = 1024       # sequence / context length
    vocab_size: int = 50304      # r50k_base (50257) padded to multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0         # 0 for pretraining, try 0.1 for fine-tuning
    bias: bool = False           # no bias in Linears or LayerNorms (slightly better)

    @classmethod
    def d12(cls) -> "NanoChatConfig":
        """~85M params — quick validation config."""
        return cls(n_layer=12, n_head=12, n_embd=768)

    @classmethod
    def d24(cls) -> "NanoChatConfig":
        """~350M params — GPT-2 medium scale."""
        return cls(n_layer=24, n_head=16, n_embd=1024)

    @classmethod
    def d26(cls) -> "NanoChatConfig":
        """~774M params — GPT-2 large scale."""
        return cls(n_layer=36, n_head=20, n_embd=1280)

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

    def num_parameters(self) -> dict:
        """Rough parameter count estimate."""
        h, v, d = self.n_embd, self.vocab_size, self.n_layer
        wte = v * h
        wpe = self.block_size * h
        attn = d * (3 * h * h + h * h)   # QKV + O
        mlp = d * (h * 4 * h + 4 * h * h)  # fc + proj
        total = wte + wpe + attn + mlp
        return {"wte": wte, "wpe": wpe, "attn": attn, "mlp": mlp, "total": total}
