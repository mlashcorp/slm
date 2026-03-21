from typing import Iterator


def pack_sequences(
    sequences: list[list[int]],
    max_len: int,
    sep_id: int,
) -> list[tuple[list[int], list[int]]]:
    """
    Greedily pack token sequences into fixed-length chunks.
    Each sequence is separated by sep_id. Long sequences are truncated.
    Returns only complete chunks of exactly max_len tokens.

    Returns list of (tokens, doc_ids) tuples. doc_ids[i] is the document
    index within the packed chunk for token i — used to build intra-document
    attention masks (tokens in different documents must not attend to each other).
    """
    chunks = []
    current: list[int] = []
    current_doc_ids: list[int] = []
    doc_idx = 0

    for seq in sequences:
        # Truncate if single sequence is longer than max_len
        if len(seq) > max_len:
            seq = seq[:max_len]

        # Add separator before sequence (except if current is empty)
        if current:
            to_add = [sep_id] + seq
            doc_ids_to_add = [doc_idx] + [doc_idx + 1] * len(seq)
            doc_idx += 1
        else:
            to_add = seq
            doc_ids_to_add = [doc_idx] * len(seq)

        if len(current) + len(to_add) <= max_len:
            current.extend(to_add)
            current_doc_ids.extend(doc_ids_to_add)
        else:
            # Flush current if non-empty, then start new chunk
            if current:
                if len(current) == max_len:
                    chunks.append((current, current_doc_ids))
                doc_idx += 1
                current = seq
                current_doc_ids = [doc_idx] * len(seq)
            else:
                current = seq
                current_doc_ids = [doc_idx] * len(seq)

        if len(current) == max_len:
            chunks.append((current, current_doc_ids))
            current = []
            current_doc_ids = []
            doc_idx += 1

    # Drop the last incomplete chunk (no padding)
    return chunks


def pack_dataset_streaming(
    dataset,
    tokenizer,
    max_len: int,
    fim_rate: float = 0.5,
    text_field: str = "text",
) -> Iterator[tuple[list[int], list[int]]]:
    """
    Stream a HuggingFace dataset, tokenize, apply FIM, and yield (tokens, doc_ids) pairs.
    doc_ids[i] is the document index for token i within the packed chunk.
    """
    from src.data.fim import apply_fim
    from src.data.tokenizer import FIM_TOKENS

    vocab = tokenizer.get_vocab()
    prefix_id = vocab[FIM_TOKENS["prefix"]]
    suffix_id = vocab[FIM_TOKENS["suffix"]]
    middle_id = vocab[FIM_TOKENS["middle"]]
    sep_id = tokenizer.eos_token_id

    token_buffer: list[int] = []
    doc_id_buffer: list[int] = []
    doc_idx = 0

    for example in dataset:
        text = example[text_field]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens = apply_fim(tokens, fim_rate=fim_rate,
                           prefix_id=prefix_id, suffix_id=suffix_id, middle_id=middle_id)
        tokens = tokens + [sep_id]
        doc_ids = [doc_idx] * len(tokens)
        doc_idx += 1

        token_buffer.extend(tokens)
        doc_id_buffer.extend(doc_ids)

        while len(token_buffer) >= max_len:
            yield token_buffer[:max_len], doc_id_buffer[:max_len]
            token_buffer = token_buffer[max_len:]
            doc_id_buffer = doc_id_buffer[max_len:]
