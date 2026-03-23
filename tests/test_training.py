"""
Tests for training infrastructure: LR schedules, checkpointing, Trainer loop.
"""
import math
import tempfile
from pathlib import Path
import torch
import pytest


# ── LR Schedules ─────────────────────────────────────────────────────────────

def test_cosine_lr_warmup():
    from src.training.schedule import cosine_lr
    # At step 0, LR should be 0
    assert cosine_lr(0, warmup_steps=100, total_steps=1000, max_lr=1e-3) == 0.0
    # At warmup end, LR should be max_lr
    assert math.isclose(cosine_lr(100, warmup_steps=100, total_steps=1000, max_lr=1e-3), 1e-3, rel_tol=1e-6)


def test_cosine_lr_decay():
    from src.training.schedule import cosine_lr
    # At final step, LR should be min_lr (default 0)
    lr = cosine_lr(1000, warmup_steps=100, total_steps=1000, max_lr=1e-3, min_lr=0.0)
    assert math.isclose(lr, 0.0, abs_tol=1e-9)


def test_wsd_lr_phases():
    from src.training.schedule import wsd_lr
    warmup, stable, decay = 100, 800, 100
    max_lr, min_lr = 1e-3, 0.0
    # Warmup phase
    assert wsd_lr(0, warmup, stable, decay, max_lr) == 0.0
    assert math.isclose(wsd_lr(100, warmup, stable, decay, max_lr), max_lr, rel_tol=1e-6)
    # Stable phase
    assert wsd_lr(500, warmup, stable, decay, max_lr) == max_lr
    # Decay end
    lr_end = wsd_lr(warmup + stable + decay, warmup, stable, decay, max_lr, min_lr)
    assert math.isclose(lr_end, min_lr, abs_tol=1e-9)


def test_wsd_lr_monotone_decay():
    from src.training.schedule import wsd_lr
    warmup, stable, decay = 100, 800, 100
    max_lr = 1e-3
    # LR must be non-increasing during decay phase
    decay_start = warmup + stable
    lrs = [wsd_lr(decay_start + i, warmup, stable, decay, max_lr) for i in range(decay + 1)]
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _tiny_model_and_opt():
    from src.model.config import ModelConfig
    from src.model.transformer import Transformer
    cfg = ModelConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=64,
    )
    model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, opt, cfg


def test_checkpoint_roundtrip():
    from src.training.checkpoint import save_checkpoint, load_checkpoint
    model, opt, cfg = _tiny_model_and_opt()

    # Do a fake step to make weights non-default
    x = torch.randint(0, 256, (1, 8))
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(model, opt, step=1, loss=9.9, config=cfg, output_dir=tmpdir)
        ckpt_path = str(Path(tmpdir) / "step-00000001")
        assert Path(ckpt_path).exists()

        # Load into fresh model
        model2, opt2, _ = _tiny_model_and_opt()
        step = load_checkpoint(ckpt_path, model2, opt2)

        assert step == 1
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_save_hf_format():
    from src.training.checkpoint import save_hf_format
    model, _, cfg = _tiny_model_and_opt()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_hf_format(model, cfg, tmpdir)
        assert (Path(tmpdir) / "config.json").exists()
        assert (Path(tmpdir) / "pytorch_model.bin").exists()


# ── Trainer (CPU smoke test) ──────────────────────────────────────────────────

def test_trainer_runs_steps():
    """Verify Trainer completes 4 steps on CPU with a tiny model + fake data."""
    from src.training.trainer import Trainer
    from src.model.config import ModelConfig
    from src.model.transformer import Transformer
    from torch.utils.data import DataLoader, IterableDataset

    cfg = ModelConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=64,
    )
    model = Transformer(cfg)

    class FakeDataset(IterableDataset):
        def __iter__(self):
            for _ in range(20):
                yield {
                    "input_ids": torch.randint(0, 256, (16,)),
                    "doc_ids": torch.zeros(16, dtype=torch.long),
                }

    loader = DataLoader(FakeDataset(), batch_size=4)

    train_cfg = {
        "lr": 1e-3, "beta1": 0.9, "beta2": 0.95,
        "weight_decay": 0.1, "grad_clip": 1.0,
        "warmup_steps": 1, "total_steps": 4,
        "schedule": "wsd", "stable_steps": 2, "decay_steps": 1,
        "grad_accum_steps": 1, "use_8bit_adam": False,
        "compile": False, "log_every": 2, "save_every": 10000,
        "output_dir": tempfile.mkdtemp(),
    }

    trainer = Trainer(model, loader, train_cfg, device="cpu")
    trainer.train()
    assert trainer.global_step == 4


def test_trainer_loss_decreases():
    """Loss after 20 steps should be lower than after 1 step (basic sanity)."""
    from src.training.trainer import Trainer
    from src.model.config import ModelConfig
    from src.model.transformer import Transformer
    from torch.utils.data import DataLoader, IterableDataset

    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=64,
    )

    # Fixed batch so we're measuring memorisation
    fixed_ids = torch.randint(0, 256, (4, 16))
    fixed_doc_ids = torch.zeros(4, 16, dtype=torch.long)

    class FixedDataset(IterableDataset):
        def __iter__(self):
            for _ in range(100):
                yield {"input_ids": fixed_ids[0], "doc_ids": fixed_doc_ids[0]}

    loader = DataLoader(FixedDataset(), batch_size=None)
    train_cfg = {
        "lr": 1e-2, "beta1": 0.9, "beta2": 0.95,
        "weight_decay": 0.0, "grad_clip": 1.0,
        "warmup_steps": 1, "total_steps": 20,
        "schedule": "wsd", "stable_steps": 15, "decay_steps": 4,
        "grad_accum_steps": 1, "use_8bit_adam": False,
        "compile": False, "log_every": 100, "save_every": 10000,
        "output_dir": tempfile.mkdtemp(),
    }

    losses = []
    model = Transformer(cfg)

    class PatchedTrainer(Trainer):
        def train(self):
            from torch.utils.data import DataLoader
            import torch.nn.functional as F
            self.model.train()
            grad_accum = self.cfg.get("grad_accum_steps", 1)
            micro_step = 0
            for batch in self.dataloader:
                if self.global_step >= self.cfg["total_steps"]:
                    break
                input_ids = batch["input_ids"].to(self.device)
                doc_ids = batch.get("doc_ids")
                if doc_ids is not None:
                    doc_ids = doc_ids.to(self.device)
                inputs = input_ids[:-1].unsqueeze(0)
                targets = input_ids[1:].unsqueeze(0)
                doc_shifted = doc_ids[:-1].unsqueeze(0) if doc_ids is not None else None
                logits = self.model(inputs, doc_ids=doc_shifted)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                (loss / grad_accum).backward()
                micro_step += 1
                if micro_step % grad_accum == 0:
                    lr = self._get_lr()
                    self._set_lr(lr)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    losses.append(loss.item())

    trainer = PatchedTrainer(model, loader, train_cfg, device="cpu")
    trainer.train()
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
