from transformers import AutoTokenizer

FIM_TOKENS = {
    "prefix": "<|fim_prefix|>",
    "suffix": "<|fim_suffix|>",
    "middle": "<|fim_middle|>",
    "pad": "<|fim_pad|>",
}

# SmolLM2 base vocab is 49152; adding 4 FIM tokens → 49156
VOCAB_SIZE = 49156


def load_tokenizer(model_id: str = "HuggingFaceTB/SmolLM2-360M"):
    """
    Load SmolLM2 tokenizer and add FIM special tokens if not already present.

    SmolLM2's deployed tokenizer does not include FIM tokens; we add them here
    so they are available for Fill-in-Middle training. Model configs must use
    VOCAB_SIZE (49156) not the base 49152.
    """
    tok = AutoTokenizer.from_pretrained(model_id)
    missing = [t for t in FIM_TOKENS.values() if t not in tok.get_vocab()]
    if missing:
        tok.add_special_tokens({"additional_special_tokens": missing})
    return tok
