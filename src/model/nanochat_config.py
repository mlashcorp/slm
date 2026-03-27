"""
NanoChat model configuration -- aligned with karpathy/nanoGPT GPTConfig.

Reference: https://github.com/karpathy/nanoGPT
"""
from dataclasses import dataclass, field


@dataclass
class NanoChatConfig:
    """GPT configuration using karpathy/nanoGPT field names.

    Core fields (matching nanoGPT):
        block_size, vocab_size, n_layer, n_head, n_embd, dropout, bias

    Extended fields for nanochat features:
        n_kv_head, rope_theta, window_pattern, and use_* feature flags.
    """

    # --- Core (nanoGPT-compatible) ---
    block_size: int = 1024
    vocab_size: int = 50304       # r50k_base (50257) padded to multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    # --- Extended architecture ---
    n_kv_head: int = 0            # 0 means n_kv_head = n_head (MHA)
    intermediate_size: int = 0    # 0 means 4 * n_embd
    rope_theta: float = 100_000.0

    # Sliding window attention: "SSSL" = Short-Short-Short-Long
    window_pattern: str = "SSSL"

    # --- Feature flags ---
    use_value_embeds: bool = True
    use_attn_gate: bool = True
    use_resid_lambdas: bool = True
    use_post_lambdas: bool = True
    use_x0_lambdas: bool = True
    use_sa_lambdas: bool = True
    use_smear: bool = True
    use_backout: bool = True
    use_skip_connection: bool = True
    use_bigram_embeds: bool = True

    # --- Feature parameters ---
    smear_gate_channels: int = 12
    skip_source_layer: int = 3
    skip_target_layer: int = 6
    bigram_vocab_size: int = 32768

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

        if self.n_kv_head == 0:
            self.n_kv_head = self.n_head

        if self.intermediate_size == 0:
            self.intermediate_size = 4 * self.n_embd

        assert self.n_embd % self.n_kv_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_kv_head ({self.n_kv_head})"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def d12(cls) -> "NanoChatConfig":
        """~12M params -- quick validation config."""
        return cls(n_layer=12, n_head=6, n_embd=384)

    @classmethod
    def d24(cls) -> "NanoChatConfig":
        """~50M params -- GPT-2 level."""
        return cls(n_layer=24, n_head=4, n_embd=512)

    @classmethod
    def d26(cls) -> "NanoChatConfig":
        """~60M params -- beat GPT-2."""
        return cls(n_layer=26, n_head=5, n_embd=640)

    def num_parameters(self) -> dict:
        """Rough parameter count estimate.

        Returns dict with component-level counts for optimizer group sizing.
        """
        h = self.n_embd
        v = self.vocab_size
        d = self.n_layer
        ff = self.intermediate_size
        kv_dim = self.n_kv_head * self.head_dim

        wte = v * h
        lm_head = v * h  # untied

        # Per-layer: Q + K + V + O projections
        attn_per_layer = (
            h * (self.n_head * self.head_dim)      # Q
            + h * (self.n_kv_head * self.head_dim)  # K
            + h * (self.n_kv_head * self.head_dim)  # V
            + (self.n_head * self.head_dim) * h      # O
        )
        attn = d * attn_per_layer

        # Per-layer: MLP up + down
        mlp = d * (h * ff + ff * h)

        # Transformer matrices (used for optimizer group sizing)
        transformer_matrices = attn + mlp

        total = wte + lm_head + transformer_matrices

        return {
            "wte": wte,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "attn": attn,
            "mlp": mlp,
            "total": total,
        }
