import random
from src.data.fim import apply_fim, FimMode


def test_clm_passthrough():
    tokens = [1, 2, 3, 4, 5]
    # fim_rate=0.0 forces CLM passthrough (random() >= 0.0 is always True)
    result = apply_fim(tokens, fim_rate=0.0, prefix_id=10, suffix_id=11, middle_id=12)
    assert result == tokens


def test_psm_structure():
    # PSM: <PRE> prefix <SUF> suffix <MID> middle
    # Use token IDs that do NOT overlap with the sequence (0-19)
    tokens = list(range(20))
    prefix_id, suffix_id, middle_id = 100, 101, 102
    result = apply_fim(tokens, mode=FimMode.PSM, prefix_id=prefix_id,
                       suffix_id=suffix_id, middle_id=middle_id)
    assert result[0] == prefix_id
    assert suffix_id in result
    assert middle_id in result
    # All original tokens still present
    original_in_result = [t for t in result if t not in (prefix_id, suffix_id, middle_id)]
    assert sorted(original_in_result) == sorted(tokens)


def test_spm_structure():
    tokens = list(range(20))
    prefix_id, suffix_id, middle_id = 100, 101, 102
    result = apply_fim(tokens, mode=FimMode.SPM, prefix_id=prefix_id,
                       suffix_id=suffix_id, middle_id=middle_id)
    assert result[0] == suffix_id
    assert prefix_id in result
    assert middle_id in result


def test_fim_rate():
    random.seed(42)
    tokens = list(range(50))
    prefix_id, suffix_id, middle_id = 10, 11, 12
    fim_count = 0
    N = 1000
    for _ in range(N):
        result = apply_fim(tokens, fim_rate=0.5, prefix_id=prefix_id,
                           suffix_id=suffix_id, middle_id=middle_id)
        if result[0] in (prefix_id, suffix_id):
            fim_count += 1
    # Should be approximately 50%, allow ±5%
    assert 0.45 < fim_count / N < 0.55
