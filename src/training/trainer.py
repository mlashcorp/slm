"""
Training loop for all phases: Phase 1 (135M), Phase 2 (360M), Phase 3 (2B).
Handles gradient accumulation, WSD LR schedule, BF16, grad norm clipping, logging.
"""
import time
import wandb
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
                fused=True,
            )

        if config.get("use_fp8", False) and device != "cpu":
            from torchao.float8 import convert_to_float8_training, Float8LinearConfig
            # pad_inner_dim=True: pads K to nearest multiple of 16 in backward pass
            # Required because B*S = 8*2047 = 16376 which is not divisible by 16
            fp8_config = Float8LinearConfig(pad_inner_dim=True)
            def _fp8_filter(mod, fqn):
                # Skip lm_head (tied embeddings) and any layer with dims not divisible by 16
                if fqn == "lm_head":
                    return False
                if not hasattr(mod, "weight"):
                    return False
                if mod.weight.shape[0] % 16 != 0 or mod.weight.shape[1] % 16 != 0:
                    return False
                return True
            convert_to_float8_training(self.model, config=fp8_config, module_filter_fn=_fp8_filter)

        if config.get("compile", True) and device != "cpu":
            self.model = torch.compile(self.model, mode="max-autotune-no-cudagraphs")

        self.global_step = 0

        # W&B: use mode="disabled" as a no-op stub when not enabled
        wandb_mode = "online" if config.get("wandb_enabled", False) else "disabled"
        wandb.init(
            project=config.get("wandb_project", "slm"),
            name=config.get("wandb_run_name", "run"),
            config=config,
            mode=wandb_mode,
            resume="allow",
        )

        # Write run ID so eval_watcher can log to the same W&B run
        if config.get("wandb_enabled", False):
            from pathlib import Path
            run_id_path = Path(config.get("output_dir", ".")) / "wandb_run_id.txt"
            run_id_path.parent.mkdir(parents=True, exist_ok=True)
            run_id_path.write_text(wandb.run.id)

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
        loss_accum = 0.0

        for batch in self.dataloader:
            if self.global_step >= total_steps:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            doc_ids = batch.get("doc_ids") if cfg.get("use_doc_mask", True) else None
            if doc_ids is not None:
                doc_ids = doc_ids.to(self.device, non_blocking=True)

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
            loss_accum += loss.item() / grad_accum
            tokens_seen += inputs.numel()
            micro_step += 1

            if micro_step % grad_accum == 0:
                lr = self._get_lr()
                self._set_lr(lr)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.get("grad_clip", 1.0)
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.global_step % log_every == 0:
                    elapsed = time.time() - t0
                    tok_per_sec = tokens_seen / elapsed
                    print(
                        f"step {self.global_step:6d} | loss {loss_accum:.4f} | "
                        f"lr {lr:.2e} | gnorm {grad_norm:.3f} | {tok_per_sec / 1000:.1f}K tok/s"
                    )
                    wandb.log({
                        "train/loss":          loss_accum,
                        "train/lr":            lr,
                        "train/grad_norm":     grad_norm.item(),
                        "train/tok_per_sec":   tok_per_sec,
                        "train/tokens_total":  tokens_seen,
                    }, step=self.global_step)

                loss_accum = 0.0

                if self.global_step % save_every == 0:
                    val_loss = self._eval_loss()
                    print(f"  → val_loss {val_loss:.4f} at step {self.global_step}")
                    wandb.log({"val/loss": val_loss}, step=self.global_step)
                    raw_model = getattr(self.model, "_orig_mod", self.model)
                    save_checkpoint(
                        model=raw_model,
                        optimizer=self.optimizer,
                        step=self.global_step,
                        loss=loss_accum,
                        config=getattr(raw_model, "config", cfg),
                        output_dir=cfg["output_dir"],
                    )

        print(f"Training complete. Steps: {self.global_step}, tokens: {tokens_seen:,}")
        wandb.finish()
