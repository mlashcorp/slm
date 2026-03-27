"""
Shape and correctness tests for all model components.
TDD: tests written before implementation.
"""
import torch
import pytest


def cpu_config():
    """Phase1 config with flash_attn disabled — flash_attn is CUDA-only."""
    from src.model.config import ModelConfig
    cfg = ModelConfig.phase1_135m()
    cfg.use_flash_attn = False
    return cfg


# ── RMSNorm ──────────────────────────────────────────────────────────────────

def test_rmsnorm_shape():
    from src.model.norm import RMSNorm
    norm = RMSNorm(64)
    x = torch.randn(2, 16, 64)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_scale():
    from src.model.norm import RMSNorm
    norm = RMSNorm(4)
    x = torch.ones(1, 1, 4)
    out = norm(x)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


# ── RoPE ─────────────────────────────────────────────────────────────────────

def test_rope_output_shape():
    from src.model.rope import precompute_freqs_cos_sin, apply_rotary_emb
    seq_len, head_dim = 16, 64
    cos, sin = precompute_freqs_cos_sin(head_dim, seq_len, theta=10000.0)
    q = torch.randn(2, seq_len, 4, head_dim)
    k = torch.randn(2, seq_len, 4, head_dim)
    q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_preserves_norm():
    from src.model.rope import precompute_freqs_cos_sin, apply_rotary_emb
    seq_len, head_dim = 8, 32
    cos, sin = precompute_freqs_cos_sin(head_dim, seq_len, theta=10000.0)
    q = torch.randn(1, seq_len, 2, head_dim)
    k = torch.randn(1, seq_len, 2, head_dim)
    q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)


# ── GQA Attention ─────────────────────────────────────────────────────────────

def test_gqa_output_shape():
    from src.model.attention import GroupedQueryAttention
    from src.model.config import ModelConfig
    from src.model.rope import precompute_freqs_cos_sin
    cfg = cpu_config()
    attn = GroupedQueryAttention(cfg)
    batch, seq = 2, 16
    x = torch.randn(batch, seq, cfg.hidden_size)
    cos, sin = precompute_freqs_cos_sin(cfg.head_dim, seq, theta=cfg.rope_theta)
    out = attn(x, cos, sin)
    assert out.shape == (batch, seq, cfg.hidden_size)


def test_gqa_causal_mask():
    from src.model.attention import GroupedQueryAttention
    from src.model.config import ModelConfig
    from src.model.rope import precompute_freqs_cos_sin
    cfg = cpu_config()
    attn = GroupedQueryAttention(cfg)
    attn.eval()
    batch, seq = 1, 8
    x = torch.randn(batch, seq, cfg.hidden_size)
    cos, sin = precompute_freqs_cos_sin(cfg.head_dim, seq, theta=cfg.rope_theta)
    x2 = x.clone()
    x2[0, 4] += 10.0
    with torch.no_grad():
        out1 = attn(x, cos, sin)
        out2 = attn(x2, cos, sin)
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-4)


# ── SwiGLU ───────────────────────────────────────────────────────────────────

def test_swiglu_shape():
    from src.model.mlp import SwiGLU
    mlp = SwiGLU(hidden_size=576, intermediate_size=1536)
    x = torch.randn(2, 16, 576)
    out = mlp(x)
    assert out.shape == x.shape


# ── TransformerBlock ──────────────────────────────────────────────────────────

def test_block_shape():
    from src.model.block import TransformerBlock
    from src.model.config import ModelConfig
    from src.model.rope import precompute_freqs_cos_sin
    cfg = cpu_config()
    block = TransformerBlock(cfg)
    batch, seq = 2, 16
    x = torch.randn(batch, seq, cfg.hidden_size)
    cos, sin = precompute_freqs_cos_sin(cfg.head_dim, seq, theta=cfg.rope_theta)
    out = block(x, cos, sin)
    assert out.shape == x.shape


# ── Full Transformer ──────────────────────────────────────────────────────────

def test_transformer_forward_shape():
    from src.model.transformer import Transformer
    cfg = cpu_config()
    model = Transformer(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_transformer_param_count():
    from src.model.transformer import Transformer
    cfg = cpu_config()
    model = Transformer(cfg)
    total = sum(p.numel() for p in model.parameters())
    # Expect ~135M. Allow ±10%.
    assert 120_000_000 < total < 150_000_000, f"Got {total:,} params"


def test_tied_embeddings():
    from src.model.transformer import Transformer
    cfg = cpu_config()
    model = Transformer(cfg)
    assert model.embed_tokens.weight is model.lm_head.weight


def test_transformer_doc_mask():
    from src.model.transformer import Transformer
    cfg = cpu_config()
    model = Transformer(cfg)
    model.eval()
    B, S = 1, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    # All tokens same doc — should match no-mask output
    doc_ids_same = torch.zeros(B, S, dtype=torch.long)
    with torch.no_grad():
        logits_no_mask = model(input_ids)
        logits_same_doc = model(input_ids, doc_ids=doc_ids_same)
    # Same doc = standard causal, outputs should match
    assert torch.allclose(logits_no_mask, logits_same_doc, atol=1e-4)
