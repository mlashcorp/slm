"""
NanoChat model configuration.

Based on Karpathy's nanochat implementation:
https://github.com/karpathy/nanochat

Key differences from SmolLM2:
- ReLU^2 activation (instead of GELU)
- QK normalization (RMSNorm on Q/K before attention)
- Value embeddings (ResFormer-style)
- Sliding window attention (SSSL pattern)
- Per-layer scalars (resid_lambdas, x0_lambdas)
- Smear gate (bigram-like mixing)
- Backout mechanism (remove low-level features)
- Softcap logits
- Untied embeddings
- No GQA (for simplicity)
"""
from dataclasses import dataclass


@dataclass
class NanoChatConfig:
    """NanoChat GPT configuration.
    
    The model size is controlled primarily by `depth` (number of layers).
    All other dimensions are computed automatically:
    - hidden_size = depth * aspect_ratio
    - num_heads = hidden_size // head_dim
    - intermediate_size = 4 * hidden_size
    
    This follows the nanochat approach where depth is the single "complexity dial".
    """
    # Core dimensions
    vocab_size: int = 32768
    sequence_len: int = 2048
    
    # Model size (depth is the primary control)
    depth: int = 24  # Number of transformer layers
    aspect_ratio: int = 21  # hidden_size = depth * aspect_ratio
    head_dim: int = 128  # Dimension per attention head
    
    # Attention
    window_pattern: str = "SSSL"  # Sliding window pattern: S=short, L=long
    
    # RoPE
    rope_theta: float = 100_000.0
    
    # Normalization
    rms_norm_eps: float = 1e-6
    
    # Value embeddings (ResFormer-style)
    # Alternating layers get value embeddings, last layer always included
    use_value_embeds: bool = True
    ve_gate_channels: int = 12  # Channels for value embedding gate
    
    # Per-layer scalars
    use_resid_lambdas: bool = True  # Residual scaling per layer
    use_x0_lambdas: bool = True  # Initial embedding blending per layer
    
    # Smear (bigram-like mixing)
    use_smear: bool = True
    smear_gate_channels: int = 24  # Channels for smear gate
    
    # Backout (remove low-level features before final projection)
    use_backout: bool = True
    
    # Logit softcap
    logit_softcap: float = 15.0  # Squash logits to [-15, 15] range
    
    # Computed properties
    @property
    def hidden_size(self) -> int:
        """Compute hidden size from depth and aspect ratio."""
        base = self.depth * self.aspect_ratio
        # Round up to nearest multiple of head_dim for clean division
        return ((base + self.head_dim - 1) // self.head_dim) * self.head_dim
    
    @property
    def num_heads(self) -> int:
        """Number of attention heads."""
        return self.hidden_size // self.head_dim
    
    @property
    def intermediate_size(self) -> int:
        """MLP intermediate size (4x hidden)."""
        return 4 * self.hidden_size
    
    @property
    def num_kv_heads(self) -> int:
        """Number of key/value heads (no GQA in nanochat)."""
        return self.num_heads
    
    @classmethod
    def d12(cls) -> "NanoChatConfig":
        """Quick validation config: ~12M params, trains in minutes."""
        return cls(
            depth=12,
            aspect_ratio=32,  # 384 hidden
            head_dim=64,
            vocab_size=32768,
        )
    
    @classmethod
    def d24(cls) -> "NanoChatConfig":
        """GPT-2 capability config: ~50M params, 2-5B tokens."""
        return cls(
            depth=24,
            aspect_ratio=21,  # ~504 -> 512 hidden
            head_dim=128,
            vocab_size=32768,
        )
    
    @classmethod  
    def d26(cls) -> "NanoChatConfig":
        """Slightly larger than GPT-2: ~60M params."""
        return cls(
            depth=26,
            aspect_ratio=24,  # ~624 -> 640 hidden
            head_dim=128,
            vocab_size=32768,
        )
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.head_dim == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by head_dim ({self.head_dim})"
        assert self.head_dim % 8 == 0, \
            f"head_dim ({self.head_dim}) must be divisible by 8 for Flash Attention"
        assert all(c in "SL" for c in self.window_pattern), \
            f"window_pattern must only contain 'S' or 'L', got {self.window_pattern}"
    
    def num_parameters(self) -> dict:
        """Estimate parameter counts for scaling law analysis."""
        h = self.hidden_size
        d = self.depth
        v = self.vocab_size
        
        # Embedding (not tied with lm_head)
        wte = v * h
        
        # Value embeddings (alternating layers, last layer always)
        # About half the layers have value embeddings
        ve_count = (d + 1) // 2
        value_embeds = ve_count * v * (self.num_kv_heads * self.head_dim)
        
        # LM head (untied)
        lm_head = v * h
        
        # Transformer matrices (per layer)
        # Attention: Q, K, V, O (4 matrices)
        # Q, K, V: h * head_dim each (but K, V may be smaller with GQA)
        # O: h * h
        attn_per_layer = h * self.head_dim * 3 + h * h  # Simplified (no GQA)
        
        # MLP: up (h -> 4h), down (4h -> h)
        mlp_per_layer = h * self.intermediate_size * 2
        
        transformer_matrices = d * (attn_per_layer + mlp_per_layer)
        
        # Scalars (negligible)
        scalars = d * 2 + 2 + 2  # resid_lambdas, x0_lambdas, smear, backout
        
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }
