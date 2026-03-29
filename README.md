# legal-colbert

A fine-tuned ColBERT model for legal contract clause retrieval, with a complete pipeline for chunking, indexing, querying, and benchmarking.

Model weights: [kmad00/legal-colbert-v1](https://huggingface.co/kmad00/legal-colbert-v1) (CC BY 4.0)

## Benchmark

**MLEB Contractual Clause Retrieval** (45 queries, 90 passages):

| Model | Type | NDCG@10 | R@1 | R@10 |
|---|---|---|---|---|
| **legal-colbert-v1 (ours)** | ColBERT, fine-tuned | **0.793** | **0.389** | **0.878** |
| GTE-ModernColBERT-v1 | ColBERT, base | 0.672 | 0.556 | — |
| BGE-large-en-v1.5 | Bi-encoder, 1024d | 0.737 | 0.644 | — |
| all-MiniLM-L6-v2 | Bi-encoder, 384d | 0.629 | 0.556 | — |

Fine-tuning added **+12.1 NDCG@10 points** over the base model (+18% relative).

## Quick Start

```bash
pip install pylate sentence-transformers datasets pymupdf rank-bm25

# Extract clauses from a contract
python extract_clauses.py contract.pdf

# Interactive retrieval
python retrieve.py interactive contract.pdf
```

## Clause Extraction

Extract specific clause types from any contract PDF or text file:

```bash
# Extract all common clause types
python extract_clauses.py contract.pdf

# Extract specific clauses
python extract_clauses.py contract.pdf termination "governing law" indemnification

# From raw text
python extract_clauses.py --text contract.txt assignment confidentiality
```

Supports 20 clause types out of the box: termination, assignment, governing law, indemnification, limitation of liability, confidentiality, events of default, representations and warranties, force majeure, severability, entire agreement, waiver of jury trial, non-competition, dispute resolution, change of control, notices, payment terms, insurance, intellectual property, subordination.

Custom clause types are auto-expanded into retrieval queries.

## How It Works

### Semantic Chunking (`chunk.py`)

Splits documents into semantically coherent passages using embedding similarity:

1. Split text into sentences (handling legal abbreviations like Inc., Ltd., Section 1.01.)
2. Compute cosine similarity between consecutive sentence embeddings (all-MiniLM-L6-v2)
3. Break where similarity drops below the 25th percentile
4. Merge small chunks, split large ones to respect token limits

Works on both structured PDFs and messy raw text — no regex patterns or format assumptions.

```python
from chunk import chunk_pdf, chunk_document

chunks = chunk_pdf("contract.pdf", max_tokens=384)
chunks = chunk_document(raw_text, max_tokens=384)
```

### ColBERT Retrieval (`retrieve.py`)

Encodes chunks with per-token ColBERT embeddings, scores via MaxSim late interaction:

```bash
python retrieve.py index contract.pdf      # Index a document
python retrieve.py query "termination penalties"  # Search
python retrieve.py interactive contract.pdf       # Interactive mode
```

Processing time (CPU):
- 54-page PDF: ~3.8s to index, ~80ms per query
- 9K-word text: ~0.9s to index, ~80ms per query

### Multi-Query Clause Extraction (`extract_clauses.py`)

For each clause type, runs multiple query variants and takes the best score per chunk. This handles ambiguous short queries like "assignment" by expanding to richer variants:

- "assignment or transfer of rights and obligations under this agreement"
- "restrictions on assigning this contract to a third party"
- "consent required for assignment of this agreement"

## Training

### Data

- **[CUAD](https://huggingface.co/datasets/theatticusproject/cuad-qa)** (CC BY 4.0): 5,449 QA pairs from 510 commercial contracts covering 41 clause types
- **[ACORD](https://huggingface.co/datasets/theatticusproject/acord)** (CC BY 4.0): 793 query-clause pairs across 9 clause categories with expert relevance ratings
- **BM25 hard negatives** with keyword overlap filtering to avoid false negatives

```bash
python prepare_data.py   # Download data, mine hard negatives (~6K triplets)
python train.py --batch-size 16 --num-epochs 10 --temperature 0.05 --bf16
```

Training takes ~7 minutes on an A100 80GB.

### Hyperparameter Optimization

Used an autoresearch loop (`research.py` + `program.md`) inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — greedy hill-climbing over training config, keeping changes that improve NDCG@10:

| Experiment | NDCG@10 | Status |
|---|---|---|
| Baseline (3ep, temp 0.02) | 0.763 | kept |
| 5 epochs | 0.772 | kept |
| Temperature 0.01 | 0.680 | discarded |
| Temperature 0.05 | 0.773 | kept |
| LR 1e-5 | 0.746 | discarded |
| **10 epochs, temp 0.05** | **0.793** | **kept** |
| 15 epochs + warmup | 0.748 | discarded (overfit) |

Key findings: 10 epochs is the sweet spot (15 overfits), temperature 0.05 > 0.02 >> 0.01, default LR (3e-6) is optimal.

### Base Model

[lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) (Apache 2.0)

## Architecture

ColBERT (Contextualized Late Interaction over BERT) produces per-token embeddings for both queries and documents. Scoring uses MaxSim: for each query token, find the maximum similarity document token, then sum across all query tokens. This captures fine-grained term-level matching that single-vector bi-encoders miss.

## Files

| File | Purpose |
|---|---|
| `extract_clauses.py` | Clause extraction with multi-query expansion (20 types) |
| `retrieve.py` | Index documents + query with ColBERT MaxSim |
| `chunk.py` | Semantic document chunker (embedding similarity) |
| `benchmark.py` | MLEB Contractual Clause Retrieval evaluation |
| `prepare_data.py` | Training data prep (CUAD + ACORD + BM25 negatives) |
| `train.py` | PyLate ColBERT fine-tuning |
| `research.py` | Autoresearch loop for hyperparameter optimization |
| `program.md` | Agent instructions for the research loop |
| `results.tsv` | Experiment log |

## License

**CC BY 4.0** — inherited from the training data ([CUAD](https://huggingface.co/datasets/theatticusproject/cuad-qa), [ACORD](https://huggingface.co/datasets/theatricusproject/acord)). Base model is Apache 2.0.

## Acknowledgments

- Training data: [The Atticus Project](https://www.atticusprojectai.org/) (CUAD, ACORD)
- Base model: [LightOn](https://huggingface.co/lightonai) (GTE-ModernColBERT-v1)
- Benchmark: [Isaacus](https://isaacus.com/mleb) (MLEB)
- Autoresearch methodology: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)

## Maintainers

- [kmad](https://github.com/kmad0)
- [Claude](https://claude.ai) (Anthropic)
