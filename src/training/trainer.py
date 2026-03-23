"""
Training loop for all phases: Phase 1 (135M), Phase 2 (360M), Phase 3 (2B).
Handles gradient accumulation, WSD LR schedule, BF16, grad norm clipping, logging.
"""
import time
import torch
import torch.nn.functional as F
from src.training.schedule import cosine_lr, wsd_lr
from src.training.checkpoint import save_checkpoint


class Trainer:
    def __init__(
        self,
        model,
        dataloader,
        config: dict,
        device: str = "cuda",
        val_dataloader=None,
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.cfg = config
        self.device = device

        # Split param groups: embeddings get no weight decay
        decay_params = [p for n, p in model.named_parameters()
                        if "embed" not in n and p.requires_grad]
        no_decay_params = [p for n, p in model.named_parameters()
                           if "embed" in n and p.requires_grad]
        param_groups = [
            {"params": decay_params,    "weight_decay": config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        if config.get("use_8bit_adam", False):
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=config["lr"],
                betas=(config["beta1"], config["beta2"]),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=config["lr"],
                betas=(config["beta1"], config["beta2"]),
            )

        if config.get("compile", True) and device != "cpu":
            self.model = torch.compile(self.model)

        self.global_step = 0

    def _get_lr(self) -> float:
        cfg = self.cfg
        if cfg.get("schedule") == "wsd":
            return wsd_lr(
                self.global_step,
                warmup_steps=cfg["warmup_steps"],
                stable_steps=cfg["stable_steps"],
                decay_steps=cfg["decay_steps"],
                max_lr=cfg["lr"],
            )
        return cosine_lr(
            self.global_step,
            warmup_steps=cfg["warmup_steps"],
            total_steps=cfg["total_steps"],
            max_lr=cfg["lr"],
        )

    def _set_lr(self, lr: float):
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @torch.no_grad()
    def _eval_loss(self, max_batches: int = 50) -> float:
        if self.val_dataloader is None:
            return float("nan")
        self.model.eval()
        total, count = 0.0, 0
        for batch in self.val_dataloader:
            if count >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            doc_ids = batch.get("doc_ids")
            if doc_ids is not None:
                doc_ids = doc_ids.to(self.device)
            inputs = input_ids[:, :-1].contiguous()
            targets = input_ids[:, 1:].contiguous()
            doc_ids_shifted = doc_ids[:, :-1].contiguous() if doc_ids is not None else None
            if self.device == "cpu":
                logits = self.model(inputs, doc_ids=doc_ids_shifted)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(inputs, doc_ids=doc_ids_shifted)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total += loss.item()
            count += 1
        self.model.train()
        return total / count if count > 0 else float("nan")

    def train(self):
        cfg = self.cfg
        self.model.train()

        grad_accum = cfg.get("grad_accum_steps", 1)
        log_every = cfg.get("log_every", 100)
        save_every = cfg.get("save_every", 1000)
        total_steps = cfg["total_steps"]
        use_cuda = self.device != "cpu"

        tokens_seen = 0
        t0 = time.time()
        micro_step = 0

        for batch in self.dataloader:
            if self.global_step >= total_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            doc_ids = batch.get("doc_ids")
            if doc_ids is not None:
                doc_ids = doc_ids.to(self.device)

            # Shift: inputs = tokens[:-1], targets = tokens[1:]
            inputs = input_ids[:, :-1].contiguous()
            targets = input_ids[:, 1:].contiguous()
            doc_ids_shifted = doc_ids[:, :-1].contiguous() if doc_ids is not None else None

            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(inputs, doc_ids=doc_ids_shifted)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                    )
            else:
                logits = self.model(inputs, doc_ids=doc_ids_shifted)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )

            (loss / grad_accum).backward()
            tokens_seen += inputs.numel()
            micro_step += 1

            if micro_step % grad_accum == 0:
                lr = self._get_lr()
                self._set_lr(lr)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.get("grad_clip", 1.0)
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % log_every == 0:
                    elapsed = time.time() - t0
                    tok_per_sec = tokens_seen / elapsed
                    print(
                        f"step {self.global_step:6d} | loss {loss.item():.4f} | "
                        f"lr {lr:.2e} | gnorm {grad_norm:.3f} | {tok_per_sec / 1000:.1f}K tok/s"
                    )

                if self.global_step % save_every == 0:
                    val_loss = self._eval_loss()
                    print(f"  → val_loss {val_loss:.4f} at step {self.global_step}")
                    raw_model = getattr(self.model, "_orig_mod", self.model)
                    save_checkpoint(
                        model=raw_model,
                        optimizer=self.optimizer,
                        step=self.global_step,
                        loss=loss.item(),
                        config=getattr(raw_model, "config", cfg),
                        output_dir=cfg["output_dir"],
                    )

        print(f"Training complete. Steps: {self.global_step}, tokens: {tokens_seen:,}")
