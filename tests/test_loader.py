"""
Smoke test for MixedPackedDataset and make_dataloader.
Uses the hydrated python_edu dataset; skipped if it doesn't exist.
"""
import pytest
import torch

PYTHON_EDU_PATH = "data/python_edu_hydrated"
FINEWEB_EDU_PATH = "data/fineweb_edu"
MAX_LEN = 128
BATCH_SIZE = 4


def datasets_available():
    from pathlib import Path
    return (
        Path(PYTHON_EDU_PATH).exists()
        and Path(FINEWEB_EDU_PATH).exists()
    )


@pytest.mark.skipif(not datasets_available(), reason="hydrated datasets not present")
def test_mixed_packed_dataset_yields_correct_shapes():
    from src.data.loader import MixedPackedDataset

    ds = MixedPackedDataset(
        python_edu_path=PYTHON_EDU_PATH,
        fineweb_edu_path=FINEWEB_EDU_PATH,
        max_len=MAX_LEN,
        fim_rate=0.5,
        seed=42,
    )
    batch = next(iter(ds))
    assert "input_ids" in batch
    assert "doc_ids" in batch
    assert batch["input_ids"].shape == (MAX_LEN,)
    assert batch["doc_ids"].shape == (MAX_LEN,)
    assert batch["input_ids"].dtype == torch.long
    assert batch["doc_ids"].dtype == torch.long


@pytest.mark.skipif(not datasets_available(), reason="hydrated datasets not present")
def test_mixed_packed_dataset_vocab_range():
    from src.data.loader import MixedPackedDataset
    from src.data.tokenizer import VOCAB_SIZE

    ds = MixedPackedDataset(
        python_edu_path=PYTHON_EDU_PATH,
        fineweb_edu_path=FINEWEB_EDU_PATH,
        max_len=MAX_LEN,
        fim_rate=0.5,
        seed=42,
    )
    for i, batch in enumerate(ds):
        assert batch["input_ids"].max().item() < VOCAB_SIZE
        assert batch["input_ids"].min().item() >= 0
        if i >= 9:
            break


@pytest.mark.skipif(not datasets_available(), reason="hydrated datasets not present")
def test_make_dataloader_batch_shapes():
    from src.data.loader import make_dataloader

    train_loader, val_loader = make_dataloader(
        python_edu_path=PYTHON_EDU_PATH,
        fineweb_edu_path=FINEWEB_EDU_PATH,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
        fim_rate=0.5,
        num_workers=0,  # avoid multiprocessing in tests
        seed=42,
        val_split=0.01,
    )

    batch = next(iter(train_loader))
    assert batch["input_ids"].shape == (BATCH_SIZE, MAX_LEN)
    assert batch["doc_ids"].shape == (BATCH_SIZE, MAX_LEN)

    val_batch = next(iter(val_loader))
    assert val_batch["input_ids"].shape == (BATCH_SIZE, MAX_LEN)
    assert val_batch["doc_ids"].shape == (BATCH_SIZE, MAX_LEN)
