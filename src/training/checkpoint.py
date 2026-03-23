import torch
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    step: int,
    loss: float,
    config,
    output_dir: str,
):
    """Save model weights, optimizer state, and training metadata."""
    path = Path(output_dir) / f"step-{step:08d}"
    path.mkdir(parents=True, exist_ok=True)

    torch.save({
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, path / "checkpoint.pt")

    save_hf_format(model, config, str(path))
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path: str, model, optimizer=None) -> int:
    """Load checkpoint. Returns step number."""
    ckpt = torch.load(Path(path) / "checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"]


def save_hf_format(model, config, output_dir: str):
    """
    Save in HuggingFace LlamaForCausalLM format so lm-eval-harness can load it.
    Weight names are remapped to match the HF Llama module hierarchy.
    """
    from transformers import LlamaConfig

    hf_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        tie_word_embeddings=config.tie_embeddings,
        hidden_act="silu",
        torch_dtype="bfloat16",
    )
    hf_config.save_pretrained(output_dir)

    state_dict = model.state_dict()
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("embed_tokens."):
            new_k = "model." + new_k
        elif new_k == "norm.weight":
            new_k = "model.norm.weight"
        elif new_k.startswith("layers."):
            new_k = "model." + new_k
        # lm_head stays as-is
        remapped[new_k] = v

    torch.save(remapped, Path(output_dir) / "pytorch_model.bin")
