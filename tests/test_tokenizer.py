from src.data.tokenizer import load_tokenizer, FIM_TOKENS


def test_fim_tokens_in_vocab():
    tok = load_tokenizer()
    for token in FIM_TOKENS.values():
        assert token in tok.get_vocab(), f"{token} not in vocab"


def test_roundtrip():
    tok = load_tokenizer()
    text = "def hello():\n    return 42\n"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_vocab_size():
    from src.data.tokenizer import VOCAB_SIZE
    tok = load_tokenizer()
    # FIM tokens are added on top of the 49152 base vocab → 49156
    assert len(tok.get_vocab()) == VOCAB_SIZE == 49156
