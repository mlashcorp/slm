from dataclasses import dataclass
from src.data.tokenizer import VOCAB_SIZE


@dataclass
class ModelConfig:
    # Dimensions
    vocab_size: int = VOCAB_SIZE       # 49156 (49152 base + 4 FIM tokens)
    hidden_size: int = 960
    intermediate_size: int = 2560
    num_hidden_layers: int = 32
    # Attention
    num_attention_heads: int = 15
    num_key_value_heads: int = 5       # GQA: groups = n_heads / n_kv_heads
    head_dim: int = 64                 # hidden_size / num_attention_heads
    # Context
    max_position_embeddings: int = 2048
    rope_theta: float = 100_000.0
    # Misc
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    use_flash_attn: bool = False
    use_gradient_checkpointing: bool = False

    @classmethod
    def phase1_135m(cls) -> "ModelConfig":
        """SmolLM2-135M architecture. ~135M params. ~1 day on RTX 5090."""
        return cls(
            vocab_size=VOCAB_SIZE,
            hidden_size=576,
            intermediate_size=1536,
            num_hidden_layers=30,
            num_attention_heads=9,
            num_key_value_heads=3,
            head_dim=64,
            max_position_embeddings=2048,
            rope_theta=100_000.0,
            use_flash_attn=True,
        )

    @classmethod
    def phase2_360m(cls) -> "ModelConfig":
        """SmolLM2-360M architecture. ~360M params. ~4 days on RTX 5090."""
        return cls(
            vocab_size=VOCAB_SIZE,
            hidden_size=960,
            intermediate_size=2560,
            num_hidden_layers=32,
            num_attention_heads=15,
            num_key_value_heads=5,
            head_dim=64,
            max_position_embeddings=2048,
            rope_theta=100_000.0,
        )

    @classmethod
    def phase3_2b(cls) -> "ModelConfig":
        """Custom 2B config. ~28-44 days on RTX 5090. Requires 8-bit AdamW + grad ckpt."""
        return cls(
            vocab_size=VOCAB_SIZE,
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=32,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=128,
            max_position_embeddings=4096,
            rope_theta=500_000.0,
            use_flash_attn=True,
            use_gradient_checkpointing=True,
        )
