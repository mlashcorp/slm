import random
from enum import Enum
from typing import Optional


class FimMode(Enum):
    PSM = "psm"  # Prefix-Suffix-Middle
    SPM = "spm"  # Suffix-Prefix-Middle


def apply_fim(
    tokens: list[int],
    fim_rate: float = 0.5,
    prefix_id: int = None,
    suffix_id: int = None,
    middle_id: int = None,
    mode: Optional[FimMode] = None,  # None = auto-sample
) -> list[int]:
    """
    Apply Fill-in-Middle transformation to a token sequence.

    With probability fim_rate, pick a random split point and rearrange as:
      PSM: [prefix_id, prefix_tokens, suffix_id, suffix_tokens, middle_id, middle_tokens]
      SPM: [suffix_id, suffix_tokens, prefix_id, prefix_tokens, middle_id, middle_tokens]

    With probability 1 - fim_rate, return tokens unchanged (CLM).
    """
    if mode is None:
        if random.random() >= fim_rate:
            return tokens
        mode = random.choice([FimMode.PSM, FimMode.SPM])

    if mode is None:
        return tokens

    n = len(tokens)
    if n < 4:
        return tokens  # too short to split meaningfully

    # Pick two split points: prefix|middle|suffix
    i = random.randint(1, n - 2)
    j = random.randint(i + 1, n - 1)

    prefix = tokens[:i]
    middle = tokens[i:j]
    suffix = tokens[j:]

    if mode == FimMode.PSM:
        return [prefix_id] + prefix + [suffix_id] + suffix + [middle_id] + middle
    else:  # SPM
        return [suffix_id] + suffix + [prefix_id] + prefix + [middle_id] + middle
