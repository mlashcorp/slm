"""
Mixed dataloader: streams python-edu and fineweb-edu at a 50/50 ratio,
applies FIM, packs to context length, and yields batches.
"""
import random
from datasets import load_from_disk
from torch.utils.data import IterableDataset, DataLoader
import torch
from src.data.tokenizer import load_tokenizer, FIM_TOKENS
from src.data.fim import apply_fim


class MixedPackedDataset(IterableDataset):
    def __init__(
        self,
        python_edu_path: str,
        fineweb_edu_path: str,
        max_len: int,
        fim_rate: float = 0.5,
        seed: int = 42,
    ):
        self.python_edu_path = python_edu_path
        self.fineweb_edu_path = fineweb_edu_path
        self.max_len = max_len
        self.fim_rate = fim_rate
        self.seed = seed
        self.tokenizer = load_tokenizer()

    def __iter__(self):
        # Offset seed per worker so each worker produces a distinct data stream
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info is not None else 0)
        rng = random.Random(seed)
        py_ds = load_from_disk(self.python_edu_path)
        fw_ds = load_from_disk(self.fineweb_edu_path)

        py_iter = iter(py_ds.shuffle(seed=self.seed))
        fw_iter = iter(fw_ds.shuffle(seed=self.seed))

        sep_id = self.tokenizer.eos_token_id
        vocab = self.tokenizer.get_vocab()
        prefix_id = vocab[FIM_TOKENS["prefix"]]
        suffix_id = vocab[FIM_TOKENS["suffix"]]
        middle_id = vocab[FIM_TOKENS["middle"]]

        py_buf: list[int] = []
        fw_buf: list[int] = []
        py_doc_buf: list[int] = []
        fw_doc_buf: list[int] = []
        py_doc_idx = 0
        fw_doc_idx = 0

        def next_tokens(it, tok_buf, doc_buf, doc_idx, field):
            if len(tok_buf) >= self.max_len:
                chunk = tok_buf[:self.max_len]
                doc_chunk = doc_buf[:self.max_len]
                return chunk, doc_chunk, tok_buf[self.max_len:], doc_buf[self.max_len:], doc_idx
            try:
                ex = next(it)
                toks = self.tokenizer.encode(ex[field], add_special_tokens=False)
                toks = apply_fim(toks, fim_rate=self.fim_rate,
                                 prefix_id=prefix_id, suffix_id=suffix_id, middle_id=middle_id)
                toks = toks + [sep_id]
                tok_buf.extend(toks)
                doc_buf.extend([doc_idx] * len(toks))
                doc_idx += 1
            except StopIteration:
                pass
            return None, None, tok_buf, doc_buf, doc_idx

        while True:
            source = rng.choice(["python", "fineweb"])

            if source == "python":
                chunk = None
                while chunk is None:
                    prev = len(py_buf)
                    chunk, doc_chunk, py_buf, py_doc_buf, py_doc_idx = next_tokens(
                        py_iter, py_buf, py_doc_buf, py_doc_idx, "text")
                    if chunk is None and len(py_buf) <= prev:
                        break  # iterator exhausted, no progress
                if chunk is None:
                    break
            else:
                chunk = None
                while chunk is None:
                    prev = len(fw_buf)
                    chunk, doc_chunk, fw_buf, fw_doc_buf, fw_doc_idx = next_tokens(
                        fw_iter, fw_buf, fw_doc_buf, fw_doc_idx, "text")
                    if chunk is None and len(fw_buf) <= prev:
                        break  # iterator exhausted, no progress
                if chunk is None:
                    break

            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "doc_ids": torch.tensor(doc_chunk, dtype=torch.long),
            }


def make_dataloader(
    python_edu_path: str,
    fineweb_edu_path: str,
    max_len: int,
    batch_size: int,
    fim_rate: float = 0.5,
    num_workers: int = 4,
    seed: int = 42,
    val_split: float = 0.01,  # hold out 1% of python-edu as validation
) -> tuple[DataLoader, DataLoader]:
    """Returns (train_loader, val_loader). Val is ~1% of python-edu, no FIM."""
    py_ds = load_from_disk(python_edu_path)
    split = py_ds.train_test_split(test_size=val_split, seed=seed)
    train_py_path = python_edu_path + "_train_split"
    val_py_path = python_edu_path + "_val_split"
    split["train"].save_to_disk(train_py_path)
    split["test"].save_to_disk(val_py_path)

    train_dataset = MixedPackedDataset(
        python_edu_path=train_py_path,
        fineweb_edu_path=fineweb_edu_path,
        max_len=max_len,
        fim_rate=fim_rate,
        seed=seed,
    )
    val_dataset = MixedPackedDataset(
        python_edu_path=val_py_path,
        fineweb_edu_path=fineweb_edu_path,
        max_len=max_len,
        fim_rate=0.0,   # no FIM on validation — cleaner perplexity signal
        seed=seed,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
