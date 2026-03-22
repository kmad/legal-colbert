# legal-colbert

A fine-tuned ColBERT model for legal contract retrieval, with an end-to-end pipeline for chunking, indexing, querying, and benchmarking.

## Benchmark Results

**MLEB Contractual Clause Retrieval** (45 queries, 90 passages):

| Model | Type | NDCG@10 | R@1 | R@10 |
|---|---|---|---|---|
| **legal-colbert-v1 (ours)** | ColBERT, fine-tuned | **0.778** | **0.711** | **0.889** |
| GTE-ModernColBERT-v1 | ColBERT, base | 0.672 | 0.556 | — |
| BGE-large-en-v1.5 | Bi-encoder, 1024d | 0.737 | 0.644 | — |
| all-MiniLM-L6-v2 | Bi-encoder, 384d | 0.629 | 0.556 | — |

Fine-tuning on CUAD + ACORD with BM25 hard negatives added **+10.6 NDCG points** over the base model.

## Quick Start

```bash
# Install dependencies
pip install pylate sentence-transformers datasets pymupdf

# Index a contract PDF and query it interactively
python retrieve.py interactive contract.pdf
```

## Pipeline

### 1. Training (`prepare_data.py`, `train.py`)

Fine-tunes [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) on legal retrieval data:

- **CUAD** (Contract Understanding Attainment Dataset): 5,449 QA pairs from 510 commercial contracts
- **ACORD** (Atticus Clause Retrieval Dataset): 793 query-clause pairs across 9 clause categories
- **BM25 hard negative mining**: for each query, the highest-scoring non-relevant passage from the corpus

```bash
python prepare_data.py        # Download data, mine hard negatives
python train.py --batch-size 16 --num-epochs 3 --bf16
```

Training takes ~3.5 minutes on an A100 for 3 epochs over 5,929 triplets.

### 2. Chunking (`chunk.py`)

Semantic chunking using sentence embedding similarity (all-MiniLM-L6-v2):

- Splits text into sentences (handling legal abbreviations)
- Computes cosine similarity between consecutive sentences
- Breaks where similarity drops below the 25th percentile
- Merges small chunks, splits large ones to respect token limits
- Works on both structured PDFs and messy raw text

```python
from chunk import chunk_pdf, chunk_document

# From PDF
chunks = chunk_pdf("contract.pdf", max_tokens=384)

# From raw text
chunks = chunk_document(text, max_tokens=384)
```

### 3. Retrieval (`retrieve.py`)

Indexes chunks with ColBERT token embeddings, queries via MaxSim late interaction scoring:

```bash
# Index a PDF
python retrieve.py index contract.pdf

# Query
python retrieve.py query "What are the events of default?"

# Interactive mode
python retrieve.py interactive contract.pdf
```

Processing time (CPU, MacBook):
- 54-page PDF: ~3.8s to index, ~80ms per query
- 9K-word text: ~0.9s to index, ~80ms per query

### 4. Cross-Contract Comparison (`compare_terms.py`)

Pulls contracts from CUAD and compares common legal terms (indemnification, governing law, termination, etc.) across them:

```bash
python compare_terms.py
```

### 5. Benchmarking (`benchmark.py`)

Evaluates on the [MLEB Contractual Clause Retrieval](https://huggingface.co/datasets/isaacus/contractual-clause-retrieval) task:

```bash
python benchmark.py
```

## Model

Weights: [kmad00/legal-colbert-v1](https://huggingface.co/kmad00/legal-colbert-v1) on HuggingFace

```python
from pylate import models
model = models.ColBERT(model_name_or_path="kmad00/legal-colbert-v1")
```

## Architecture

ColBERT (Contextualized Late Interaction over BERT) produces per-token embeddings for both queries and documents. Scoring uses MaxSim: for each query token, find the max-similarity document token, then sum across all query tokens. This captures fine-grained term-level matching that single-vector bi-encoders miss.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Training data prep (CUAD + ACORD + BM25 negatives) |
| `train.py` | PyLate ColBERT fine-tuning |
| `chunk.py` | Semantic document chunker |
| `retrieve.py` | Index + query pipeline |
| `compare_terms.py` | Cross-contract term comparison |
| `benchmark.py` | MLEB benchmark evaluation |
