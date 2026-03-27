# Dolma Dataset

**HuggingFace:** https://huggingface.co/datasets/allenai/dolma  
**License:** ODC-BY  
**Paper:** https://arxiv.org/abs/2402.00159

## Overview

Dolma is a dataset of **3 trillion tokens** from a diverse mix of web content, academic publications, code, books, and encyclopedic materials. It was used to train OLMo models.

## Versions

| Version | Release Date | Size (gzip) | Description |
|---------|--------------|-------------|-------------|
| `v1_7` (default) | 2024-04-15 | 4.5 TB | Used to train OLMo-7B-v1.7. New sources, more quality filtering, fuzzy deduplication. |
| `v1_6` | 2024-01-31 | 5.4 TB | Deduplication of documents with too few tokens or too many repeated n-grams. |
| `v1_6-sample` | 2024-01-31 | 16.4 GB | Smaller sample (~10B tokens) for data exploration. |
| `v1_5` | 2023-10-31 | 6.4 TB | Used to train OLMo-1B. ~3 trillion tokens. |
| `v1_5-sample` | 2023-10-31 | 2.9 TB | Sample of ~1.9T tokens used to train OLMo-7B. |

## v1.7 Summary Statistics

| Source | Provenance | Documents (M) | OLMo Tokens (B) | Sample Prop | Cutoff Date |
|--------|------------|---------------|-----------------|-------------|-------------|
| Dolma's CC | Common Crawl | 875.2 | 1,195.5 | 50% | Mar 2023 |
| Refined Web | Falcon RefinedWeb | 664.0 | 456.4 | 100% | Feb 2023 |
| StarCoder | StarCoder | 206.6 | 263.8 | 100% | May 2023 |
| C4 | C4 | 249.9 | 138.4 | 50% | Apr 2019 |
| Reddit | PushShift API | 377.4 | 79.9 | 100% | Mar 2023 |
| Semantic Scholar | peS2o | 38.8 | 57.2 | 100% | Mar 2023 |
| arXiv | RedPajama v1 | 1.5 | 28.0 | 100% | Mar 2023 |
| StackExchange | RedPajama v1 | 29.3 | 19.6 | 100% | Mar 2023 |
| Flan | Flan Collection | 52.1 | 16.5 | 100% | Feb 2023 |
| CC News | Common Crawl | 22.0 | 14.3 | 100% | Mar 2023 |
| OpenWebMath | OpenWebMath | 2.9 | 12.6 | 100% | May 2023 |
| Algebraic Stack | Proof Pile II | 2.8 | 12.6 | 100% | Oct 2023 |
| Project Gutenberg | Project Gutenberg | 0.056 | 5.3 | 100% | Mar 2023 |
| MegaWika | MetaWika | 3.2 | 4.6 | 100% | Jul 2023 |
| Wikipedia & Wikibooks | Wikimedia | 6.2 | 3.7 | 200% | Mar 2023 |
| **Total** | | **2,532.0** | **2,308.5** | | |

**Actual training tokens for OLMo 7B-v1.7:** 1.715 trillion (after sampling proportion applied)

## Data Processing Pipeline

1. **Quality filtering** — Language detection, text quality scores, deduplication
2. **Fuzzy deduplication** — Near-duplicate removal
3. **Tokenization** — OLMo tokenizer (not Llama)
4. **Provenance tracking** — Each document tracks its source

## Download

```bash
DATA_DIR="<path_to_data>"
PARALLEL_DOWNLOADS=8
DOLMA_VERSION="v1_7"

git clone https://huggingface.co/datasets/allenai/dolma
mkdir -p "${DATA_DIR}"
cat "dolma/urls/${DOLMA_VERSION}.txt" | xargs -n 1 -P "${PARALLEL_DOWNLOADS}" wget -q -P "$DATA_DIR"
```

## Loading with HuggingFace Datasets

```python
import os
from datasets import load_dataset

os.environ["DATA_DIR"] = "<path_to_your_data_directory>"
dataset = load_dataset("allenai/dolma", split="train")
```

## Citation

```bibtex
@article{dolma,
  title = {{Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research}},
  author={Luca Soldaini and Rodney Kinney and Akshita Bhagia and Dustin Schwenk and David Atkinson and Russell Authur and Ben Bogin and Khyathi Chandu and Jennifer Dumas and Yanai Elazar and Valentin Hofmann and Ananya Harsh Jha and Sachin Kumar and Li Lucy and Xinxi Lyu and Nathan Lambert and Ian Magnusson and Jacob Morrison and Niklas Muennighoff and Aakanksha Naik and Crystal Nam and Matthew E. Peters and Abhilasha Ravichander and Kyle Richardson and Zejiang Shen and Emma Strubell and Nishant Subramani and Oyvind Tafjord and Pete Walsh and Luke Zettlemoyer and Noah A. Smith and Hannaneh Hajishirzi and Iz Beltagy and Dirk Groeneveld and Jesse Dodge and Kyle Lo},
  year = {2024},
  journal={arXiv preprint},
}
```
