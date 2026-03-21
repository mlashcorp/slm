from src.data.pack import pack_sequences


def test_packs_to_exact_length():
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10]]
    sep = 0
    packed = pack_sequences(seqs, max_len=6, sep_id=sep)
    for tokens, doc_ids in packed:
        assert len(tokens) == 6
        assert len(doc_ids) == 6


def test_no_data_loss():
    # [1,2] + sep + [3,4] = 5 tokens → exactly one complete chunk
    seqs = [[1, 2], [3, 4]]
    sep = 0
    packed = pack_sequences(seqs, max_len=5, sep_id=sep)
    assert len(packed) == 1
    tokens, _ = packed[0]
    non_sep = [t for t in tokens if t != sep]
    assert sorted(non_sep) == [1, 2, 3, 4]


def test_long_sequence_truncated():
    seqs = [list(range(100))]
    packed = pack_sequences(seqs, max_len=10, sep_id=0)
    for tokens, doc_ids in packed:
        assert len(tokens) == 10
        assert len(doc_ids) == 10


def test_doc_ids_same_within_doc():
    # Single sequence packed alone — all tokens same doc_id
    seqs = [list(range(6))]
    packed = pack_sequences(seqs, max_len=6, sep_id=0)
    assert len(packed) == 1
    tokens, doc_ids = packed[0]
    assert len(set(doc_ids)) == 1


def test_doc_ids_differ_across_docs():
    # [1,2] + sep + [3,4] = 5 tokens packed into one chunk of 5
    seqs = [[1, 2], [3, 4]]
    packed = pack_sequences(seqs, max_len=5, sep_id=0)
    assert len(packed) == 1
    tokens, doc_ids = packed[0]
    sep_pos = tokens.index(0)
    # Token before sep and token after sep must have different doc_ids
    assert doc_ids[sep_pos - 1] != doc_ids[sep_pos + 1]
