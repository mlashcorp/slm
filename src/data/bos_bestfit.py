"""
BOS-aligned Best-Fit Data Loader for NanoChat training.

Based on nanochat's dataloader.py:
https://github.com/karpathy/nanochat/blob/master/nanochat/dataloader.py

Key Features:
- Every sequence starts with BOS token
- Best-fit cropping: find largest doc that fits to minimize waste
- 100% utilization (no padding)
- ~35% tokens cropped (but every token is trained on)

Algorithm:
    For each row:
        while pos < row_capacity:
            # Find largest doc that fits entirely
            best = max(docs, key=len) if max_len <= remaining else None
            if best:
                row[pos:pos+len(best)] = best
                pos += len(best)
            else:
                # Crop shortest to fill remaining space
                shortest = min(docs, key=len)
                row[pos:] = shortest[:remaining]
                pos = row_capacity
"""
import torch
import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterator, Tuple, Dict, Optional, List


def list_parquet_files(data_dir: Path) -> List[Path]:
    """List all parquet files in the data directory.

    Args:
        data_dir: Directory containing parquet files

    Returns:
        List of parquet file paths
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = sorted([
        f for f in data_dir.iterdir()
        if f.suffix == '.parquet' and not f.name.endswith('.tmp')
    ])

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    return parquet_files


def iterate_documents_batched(
    parquet_paths: List[Path],
    split: str = "train",
    start: int = 0,
    step: int = 1,
) -> Iterator[Tuple[List[str], Tuple[int, int]]]:
    """Iterate through dataset in batches of documents.

    Args:
        parquet_paths: List of parquet file paths
        split: 'train' or 'val' (last file is validation)
        start: Starting row group index (for DDP)
        step: Step size for row groups (for DDP)

    Yields:
        (documents, (file_idx, row_group_idx)) tuples
    """
    # Split: train = all but last file, val = last file
    if split == "train":
        paths = parquet_paths[:-1]
    else:
        paths = parquet_paths[-1:]

    for file_idx, filepath in enumerate(paths):
        pf = pq.ParquetFile(filepath)

        for rg_idx in range(start, pf.num_row_groups, step):
            row_group = pf.read_row_group(rg_idx)
            documents = row_group.column('text').to_pylist()
            yield documents, (file_idx, rg_idx)


class Tokenizer:
    """Simple BPE tokenizer wrapper for Phase 0.

    Uses tiktoken for efficient encoding (same as GPT-4/nanochat).
    """

    def __init__(self, tokenizer_path: Optional[Path] = None):
        """Initialize tokenizer.

        Args:
            tokenizer_path: Path to tokenizer file. If None, uses cl100k_base.
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken not installed. Install with: pip install tiktoken"
            )

        if tokenizer_path and tokenizer_path.exists():
            # Load custom tokenizer
            self._tokenizer = tiktoken.load(str(tokenizer_path))
        else:
            # Use GPT-4's tokenizer as default
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

        # Special tokens
        self._bos_token = "<|endoftext|>"  # BOS is same as EOS for GPT-4
        self._eos_token = "<|endoftext|>"
        self._bos_id = self._tokenizer.encode_single_token(self._bos_token)
        self._eos_id = self._tokenizer.encode_single_token(self._eos_token)

    def get_bos_token_id(self) -> int:
        """Get BOS token ID."""
        return self._bos_id

    def get_eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self._eos_id

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._tokenizer.nvocab

    def encode(
        self,
        documents: List[str],
        prepend: Optional[int] = None,
        append: Optional[int] = None,
        num_threads: int = 4,
    ) -> List[List[int]]:
        """Encode documents to token IDs.

        Args:
            documents: List of text documents
            prepend: Token ID to prepend (e.g., BOS)
            append: Token ID to append (e.g., EOS)
            num_threads: Number of threads for encoding

        Returns:
            List of token ID lists
        """
        if prepend is None:
            prepend = self._bos_id

        def encode_doc(doc: str) -> List[int]:
            tokens = self._tokenizer.encode_ordinary(doc)
            if prepend is not None:
                tokens = [prepend] + tokens
            if append is not None:
                tokens = tokens + [append]
            return tokens

        # Encode in parallel using thread pool
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            token_lists = list(executor.map(encode_doc, documents))

        return token_lists

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(tokens)


def best_fit_data_loader(
    data_dir: Path,
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    split: str = "train",
    buffer_size: int = 1000,
    tokenizer_threads: int = 4,
    device: str = "cuda",
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict]]:
    """BOS-aligned best-fit data loader.

    Args:
        data_dir: Directory containing parquet files
        tokenizer: Tokenizer instance
        batch_size: Number of sequences per batch
        seq_len: Sequence length (context window)
        split: 'train' or 'val'
        buffer_size: Number of documents to buffer for best-fit selection
        tokenizer_threads: Number of threads for tokenization
        device: Device to load data to

    Yields:
        (inputs, targets, state_dict) tuples where:
            inputs: (batch_size, seq_len) input tokens
            targets: (batch_size, seq_len) target tokens (shifted by 1)
            state_dict: Dict with current position info for resumption
    """
    parquet_paths = list_parquet_files(data_dir)

    # Row capacity includes BOS (we shift targets)
    row_capacity = seq_len + 1

    # Document buffer for best-fit selection
    doc_buffer: List[List[int]] = []

    # Current position tracking
    current_file_idx = 0
    current_rg_idx = 0

    # Document iterator
    doc_iter = iterate_documents_batched(parquet_paths, split)

    def refill_buffer():
        """Refill document buffer from parquet files."""
        nonlocal current_file_idx, current_rg_idx

        try:
            documents, (current_file_idx, current_rg_idx) = next(doc_iter)
        except StopIteration:
            # Wrap around
            doc_iter = iterate_documents_batched(parquet_paths, split)
            documents, (current_file_idx, current_rg_idx) = next(doc_iter)

        # Encode documents with BOS token prepended
        token_lists = tokenizer.encode(
            documents,
            prepend=tokenizer.get_bos_token_id(),
            num_threads=tokenizer_threads,
        )

        for tokens in token_lists:
            if len(tokens) > 1:  # Skip empty documents
                doc_buffer.append(tokens)

    # Pre-allocate buffers
    use_cuda = device == "cuda" and torch.cuda.is_available()
    actual_device = torch.device(device) if use_cuda else torch.device("cpu")

    row_buffer = torch.zeros((batch_size, row_capacity), dtype=torch.long)

    # CPU buffer for pinning (for faster GPU transfer)
    if use_cuda:
        cpu_buffer = torch.zeros(2 * batch_size * seq_len, dtype=torch.long, pin_memory=True)
        cpu_inputs = cpu_buffer[:batch_size * seq_len].view(batch_size, seq_len)
        cpu_targets = cpu_buffer[batch_size * seq_len:].view(batch_size, seq_len)

    gpu_buffer = torch.zeros(2 * batch_size * seq_len, dtype=torch.long, device=actual_device)
    inputs = gpu_buffer[:batch_size * seq_len].view(batch_size, seq_len)
    targets = gpu_buffer[batch_size * seq_len:].view(batch_size, seq_len)

    while True:
        # Build batch using best-fit algorithm
        for row_idx in range(batch_size):
            pos = 0

            while pos < row_capacity:
                # Refill buffer if needed
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                if not doc_buffer:
                    break

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0

                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    # Use largest fitting doc
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits - crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Split into inputs and targets (shifted by 1)
        if use_cuda:
            cpu_inputs.copy_(row_buffer[:, :-1])
            cpu_targets.copy_(row_buffer[:, 1:])
            gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        else:
            inputs.copy_(row_buffer[:, :-1])
            targets.copy_(row_buffer[:, 1:])

        # Position state for resumption
        state_dict = {
            "file_idx": current_file_idx,
            "rg_idx": current_rg_idx,
            "buffer_size": len(doc_buffer),
        }

        yield inputs, targets, state_dict


def create_data_loader(
    batch_size: int,
    seq_len: int,
    split: str = "train",
    data_dir: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    device: str = "cuda",
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict]]:
    """Create a data loader for Phase 0 training.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        split: 'train' or 'val'
        data_dir: Data directory (default: data/dclm)
        tokenizer_path: Tokenizer file path
        device: Device to load data to

    Returns:
        Data loader iterator
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "dclm"

    tokenizer = Tokenizer(tokenizer_path)

    return best_fit_data_loader(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        split=split,
        device=device,
    )


if __name__ == "__main__":
    # Test data loader
    print("Testing BOS-aligned best-fit data loader...")

    # Use current directory's data if available
    data_dir = Path(__file__).parent.parent / "data" / "dclm"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Run 'python scripts/download_dclm.py -n 8' to download sample data")
    else:
        try:
            loader = create_data_loader(
                batch_size=4,
                seq_len=128,
                split="train",
                data_dir=data_dir,
            )

            inputs, targets, state = next(loader)
            print(f"Loaded batch: inputs {inputs.shape}, targets {targets.shape}")
            print(f"State: {state}")
        except Exception as e:
            print(f"Error: {e}")
