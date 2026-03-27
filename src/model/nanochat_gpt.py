"""
NanoChat GPT model implementation.

Based on Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT
Gap-closed against KellerJordan/modded-nanogpt.

Key differences from nanoGPT:
- RMSNorm (no learnable params) instead of LayerNorm
- RoPE instead of absolute position embeddings
- ReLU^2 activation instead of GELU
- Untied embeddings (separate wte and lm_head)
- QK normalization
- Value embeddings (ResFormer-style)
- Sliding window attention (SSSL pattern)
- Per-layer scalars (resid_lambdas, post_lambdas, x0_lambdas, sa_lambdas)
- Smear gate, attention output gate, backout, skip connection, bigram embeddings
- Softcap logits (shifted sigmoid)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.model.nanochat_config import NanoChatConfig


COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32


def norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.

    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_value_embed(layer_idx: int, n_layer: int) -> bool:
    """Returns True if layer should have Value Embedding (alternating, last always)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings.

    Args:
        x: (B, T, H, D) attention tensor
        cos, sin: (1, T, 1, D//2) rotary embeddings

    Returns:
        Rotated tensor of same shape
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def precompute_rotary_embeddings(
    seq_len: int, head_dim: int, base: float = 100_000.0, device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary position embeddings.

    Returns:
        cos, sin tensors of shape (1, seq_len, 1, head_dim//2)
    """
    if device is None:
        device = torch.device("cpu")

    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))

    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)

    cos = freqs.cos().to(COMPUTE_DTYPE)
    sin = freqs.sin().to(COMPUTE_DTYPE)

    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]

    return cos, sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with QK norm, value embeddings,
    attention output gate, sa_lambdas scaling, and sliding window support.
    """

    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

        if config.use_value_embeds and has_value_embed(layer_idx, config.n_layer):
            self.ve_gate = Linear(12, self.n_kv_head, bias=False)
        else:
            self.ve_gate = None

        if config.use_attn_gate:
            self.attn_gate = Linear(12, self.n_head, bias=False)
        else:
            self.attn_gate = None

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        window_size: Tuple[int, int],
        sa_lam: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input tensor (pre-normed)
            ve: (B, T, kv_dim) value embedding (optional, flat)
            cos, sin: Rotary embeddings
            window_size: (left, right) tuple for sliding window
            sa_lam: (2,) per-sublayer scalars [qkv_scale, o_scale] or None
        """
        B, T, C = x.size()

        qkv_scale = sa_lam[0] if sa_lam is not None else 1.0
        o_scale = sa_lam[1] if sa_lam is not None else 1.0

        q = qkv_scale * self.c_q(x)
        k = qkv_scale * self.c_k(x)
        v = qkv_scale * self.c_v(x)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): gate uses cat([x[:6], ve[:6]])
        if ve is not None and self.ve_gate is not None:
            gate_input = torch.cat([x[..., :6], ve[..., :6]], dim=-1)
            gate = 2 * torch.sigmoid(self.ve_gate(gate_input))
            ve_r = ve.view(B, T, self.n_kv_head, self.head_dim)
            v = v + gate.unsqueeze(-1) * ve_r

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK normalization + sharpening
        q = norm(q) * 1.2
        k = norm(k) * 1.2

        scale = 1.0 / math.sqrt(self.head_dim)

        if window_size[0] < T:
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            for i in range(T):
                mask[i, :max(0, i - window_size[0])] = True
            attn_mask = mask.masked_fill(mask, float('-inf'))
            attn_mask = attn_mask.masked_fill(~mask, 0.0)
        else:
            attn_mask = None

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=(attn_mask is None),
            scale=scale,
        )

        y = y.transpose(1, 2).contiguous()

        if self.attn_gate is not None:
            gate = torch.sigmoid(self.attn_gate(x[..., :12]))
            y = y * gate.unsqueeze(-1)

        y = y.view(B, T, C)
        y = o_scale * self.c_proj(y)

        return y


class MLP(nn.Module):
    """MLP with ReLU^2 activation."""

    def __init__(self, config: NanoChatConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, config.intermediate_size, bias=False)
        self.c_proj = Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, config: NanoChatConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        window_size: Tuple[int, int],
    ) -> torch.Tensor:
        x = x + self.attn(norm(x), ve, cos, sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    """NanoChat GPT model.

    Uses nanoGPT-compatible config field names (n_layer, n_head, n_embd, block_size).
    """

    def __init__(self, config: NanoChatConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config

        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([
                Block(config, i) for i in range(config.n_layer)
            ]),
        })

        # Untied LM head (differs from nanoGPT which ties wte and lm_head)
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

        # Per-layer residual scalars: shape (n_layer, 2) for attn and MLP
        if config.use_resid_lambdas:
            self.resid_lambdas = nn.Parameter(torch.full((config.n_layer, 2), 1.1 ** 0.5))
        else:
            self.register_buffer("resid_lambdas", torch.full((config.n_layer, 2), 1.1 ** 0.5), persistent=False)

        if config.use_post_lambdas:
            self.post_lambdas = nn.Parameter(torch.ones(config.n_layer, 2))
        else:
            self.register_buffer("post_lambdas", torch.ones(config.n_layer, 2), persistent=False)

        if config.use_x0_lambdas:
            self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        else:
            self.register_buffer("x0_lambdas", torch.zeros(config.n_layer), persistent=False)

        if config.use_sa_lambdas:
            self.sa_lambdas = nn.Parameter(torch.tensor([[0.5, 1.0]] * config.n_layer, dtype=torch.float32))
        else:
            self.register_buffer("sa_lambdas", torch.tensor([[0.5, 1.0]] * config.n_layer, dtype=torch.float32), persistent=False)

        # Smear gate
        if config.use_smear:
            self.smear_gate = Linear(config.smear_gate_channels, 1, bias=False)
            self.smear_lambda = nn.Parameter(torch.zeros(1))
        else:
            self.smear_gate = None
            self.register_buffer("smear_lambda", torch.zeros(1), persistent=False)

        # Backout
        if config.use_backout:
            self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        else:
            self.register_buffer("backout_lambda", 0.2 * torch.ones(1), persistent=False)

        # Skip connection + attention-free layer
        if config.use_skip_connection:
            self.skip_gate = Linear(12, 1, bias=False)
            self.skip_lambda = nn.Parameter(torch.tensor(-1.5))
        else:
            self.skip_gate = None
            self.register_buffer("skip_lambda", torch.tensor(-1.5), persistent=False)

        # Value embeddings (ResFormer-style)
        if config.use_value_embeds:
            kv_dim = config.n_kv_head * config.head_dim
            self.value_embeds = nn.ModuleDict({
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_value_embed(i, config.n_layer)
            })
        else:
            self.value_embeds = None

        # Bigram embeddings
        if config.use_bigram_embeds:
            self.bigram_embed = nn.Embedding(config.bigram_vocab_size, config.n_embd)
            self.bigram_lambdas = nn.Parameter(torch.full((config.n_layer,), 0.05))
            rng = torch.Generator()
            rng.manual_seed(42)
            rand1 = torch.randint(1, 2 ** 31, (1,), dtype=torch.long, generator=rng)
            rand2 = torch.randint(1, 2 ** 31, (1,), dtype=torch.long, generator=rng)
            self.register_buffer("bigram_rand1", rand1, persistent=True)
            self.register_buffer("bigram_rand2", rand2, persistent=True)
        else:
            self.bigram_embed = None
            self.register_buffer("bigram_lambdas", torch.zeros(config.n_layer), persistent=False)

        # Precompute rotary embeddings (10x over-provision)
        rotary_seq_len = config.block_size * 10
        cos, sin = precompute_rotary_embeddings(rotary_seq_len, config.head_dim, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """Initialize model weights.

        Strategy from nanochat:
        - Embedding: normal, std=0.8
        - LM head: normal, std=0.001
        - Attention Q,K,V: uniform with std = sqrt(3/d)
        - Attention output: zeros
        - MLP up: uniform with std = 0.4 * sqrt(3/d)
        - MLP down: zeros
        - attn_gate, skip_gate: zeros
        """
        n_embd = self.config.n_embd
        s = math.sqrt(3) * n_embd ** -0.5

        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        for block in self.transformer.h:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)

            if block.attn.attn_gate is not None:
                nn.init.zeros_(block.attn.attn_gate.weight)

            nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            nn.init.zeros_(block.mlp.c_proj.weight)

        if self.config.use_x0_lambdas:
            n_layer = self.config.n_layer
            for i in range(n_layer):
                self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        if self.skip_gate is not None:
            nn.init.zeros_(self.skip_gate.weight)

        if self.value_embeds is not None:
            for ve in self.value_embeds.values():
                nn.init.uniform_(ve.weight, -s, s)

        if self.bigram_embed is not None:
            nn.init.normal_(self.bigram_embed.weight, mean=0.0, std=0.02)

        if self.smear_gate is not None:
            nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.weight.data = self.transformer.wte.weight.data.to(COMPUTE_DTYPE)
            if self.value_embeds is not None:
                for ve in self.value_embeds.values():
                    ve.weight.data = ve.weight.data.to(COMPUTE_DTYPE)
            if self.bigram_embed is not None:
                self.bigram_embed.weight.data = self.bigram_embed.weight.data.to(COMPUTE_DTYPE)

    def _compute_window_sizes(self) -> list:
        """Compute per-layer window sizes for sliding window attention."""
        pattern = self.config.window_pattern.upper()
        T = self.config.block_size

        long_window = T
        short_window = -(-T // 4 // 128) * 128  # Ceil to FA3 tile size

        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }

        window_sizes = []
        for layer_idx in range(self.config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])

        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)

        return window_sizes

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            idx: (B, T) input token indices
            targets: (B, T) target tokens for loss computation

        Returns:
            If targets provided: loss scalar
            Otherwise: (B, T, vocab_size) logits
        """
        B, T = idx.size()

        assert T <= self.rope_cos.size(1), f"Sequence {T} exceeds rotary cache {self.rope_cos.size(1)}"
        cos = self.rope_cos[:, :T]
        sin = self.rope_sin[:, :T]

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Smear: mix previous token's embedding into current position
        if self.smear_gate is not None and T > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                self.smear_gate(x[:, 1:, :self.config.smear_gate_channels])
            )
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        x0 = x

        # Bigram embeddings
        if self.bigram_embed is not None:
            tok = idx.long()
            prev = torch.cat([tok[:, :1], tok[:, :-1]], dim=1)
            bigram_idx = (self.bigram_rand1 * tok ^ self.bigram_rand2 * prev) % (self.config.bigram_vocab_size - 1)
            x0_bigram = self.bigram_embed(bigram_idx).to(x.dtype)
        else:
            x0_bigram = None

        # Backout: cache at ~64% depth (7/11)
        backout_layer = round(self.config.n_layer * 7 / 11)
        x_backout = None

        window_sizes = self._compute_window_sizes()

        skip_connection = None
        cfg = self.config

        for i, block in enumerate(self.transformer.h):
            x0_inject = self.x0_lambdas[i] * x0
            if x0_bigram is not None:
                x0_inject = x0_inject + self.bigram_lambdas[i] * x0_bigram

            if self.value_embeds is not None and str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx).to(x.dtype)
            else:
                ve = None

            sa_lam = self.sa_lambdas[i] if cfg.use_sa_lambdas else None

            if cfg.use_skip_connection and i == cfg.skip_source_layer:
                skip_connection = x

            if cfg.use_skip_connection and i == cfg.skip_target_layer and skip_connection is not None:
                gate_val = torch.sigmoid(self.skip_lambda) * torch.sigmoid(
                    self.skip_gate(x[..., :12])
                )
                skip_out = gate_val * skip_connection
                x = self.resid_lambdas[i, 0] * x + self.post_lambdas[i, 0] * skip_out + x0_inject
                mlp_out = block.mlp(norm(x))
                x = self.resid_lambdas[i, 1] * x + self.post_lambdas[i, 1] * mlp_out
            else:
                attn_out = block.attn(norm(x), ve, cos, sin, window_sizes[i], sa_lam)
                x = self.resid_lambdas[i, 0] * x + self.post_lambdas[i, 0] * attn_out + x0_inject
                mlp_out = block.mlp(norm(x))
                x = self.resid_lambdas[i, 1] * x + self.post_lambdas[i, 1] * mlp_out

            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout

        x = norm(x)

        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]

        # Softcap logits: shifted sigmoid
        logits = logits.float()
        logits = 23.0 * torch.sigmoid((logits + 5.0) / 7.5)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return loss

        return logits

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def estimate_flops_per_token(self) -> int:
        """Estimate FLOPs per token (forward + backward)."""
        nparams = self.num_parameters()

        wte = self.transformer.wte.weight.numel()
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values()) if self.value_embeds else 0
        bigram_numel = self.bigram_embed.weight.numel() if self.bigram_embed else 0
        scalars = (self.resid_lambdas.numel() + self.post_lambdas.numel() +
                   self.x0_lambdas.numel() + self.sa_lambdas.numel() +
                   self.bigram_lambdas.numel())
        if self.smear_gate is not None:
            scalars += self.smear_gate.weight.numel() + self.smear_lambda.numel()
        scalars += self.backout_lambda.numel()

        nparams_exclude = wte + value_embeds_numel + bigram_numel + scalars

        H = self.config.n_head
        D = self.config.head_dim
        T = self.config.block_size

        attn_flops = 0
        window_sizes = self._compute_window_sizes()
        for window_size in window_sizes:
            window = window_size[0]
            effective_seq = T if window < 0 else min(window, T)
            attn_flops += 12 * H * D * effective_seq

        flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return flops_per_token

    @torch.no_grad()
    def generate(
        self,
        tokens: list,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42,
    ) -> list:
        """Naive autoregressive generation."""
        device = next(self.parameters()).device

        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat([ids, next_id], dim=1)
            yield next_id.item()
