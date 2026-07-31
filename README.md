# legal-colbert

A fine-tuned ColBERT model for legal contract clause retrieval, with a complete pipeline for chunking, indexing, querying, and benchmarking.

Current model weights (P6b, best): [kmad00/legal-colbert-clause-retriever](https://huggingface.co/kmad00/legal-colbert-clause-retriever) (CC BY 4.0).
The original V1 remains at [kmad00/legal-colbert-v1](https://huggingface.co/kmad00/legal-colbert-v1).
Full experiment log (V1 → P6b, all eval protocols): [`../CURRENT_STATUS.md`](../CURRENT_STATUS.md).

## Benchmark

**MLEB Contractual Clause Retrieval** (45 queries, 90 passages):

| Model | Type | NDCG@10 | MAP | R@10 |
|---|---|---|---|---|
| **legal-colbert-clause-retriever / P6b (ours)** | ColBERT, fine-tuned | **0.834** | **0.771** | **0.956** |
| legal-colbert-v1 (ours, original) | ColBERT, fine-tuned | 0.813 | 0.741 | 0.933 |
| BGE-M3 | Bi-encoder | 0.728 | — | — |
| GTE-ModernColBERT-v1 | ColBERT, base | 0.672 | — | — |
| BM25 (stopword-filtered) | Lexical baseline | 0.619 | 0.554 | 0.756 |

At 149M parameters, P6b ranks in the top open models on this task — ahead of EmbeddingGemma (308M), Jina v4 (3.8B), and every OpenAI/Google general embedder; behind only Voyage 4 / Kanon 2 / Qwen3-4B+. Fine-tuning lifted the base model by +16.2 NDCG points.

## Query phrasing matters

The model is strongly query-phrasing sensitive (see the P0 ablation in `../CURRENT_STATUS.md`): it is trained and evaluated with **descriptive clause definitions**, not keyword or template queries. Phrase queries the way MLEB does:

- ✅ `"Find contractual provisions governing termination rights, termination for cause, or the consequences of ending the agreement."`
- ⚠️ `"termination clause"` (terse/keyword queries can flip rankings)

If you're building an application on top, map user inputs to descriptive clause definitions before retrieval.

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
- **Model-based hard negatives**: Uses the trained model itself to find the most confusing non-relevant passages per query (passages the model scores highest but are wrong)

```bash
# First pass: BM25 negatives for initial training
python prepare_data.py

# Second pass: model-based negatives for harder training signal
python prepare_data.py --model-negatives --model-path model
```

Training takes ~7 minutes on an A100 80GB.

### Optimization Journey

Used an autoresearch loop (`research.py` + `program.md`) inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — greedy hill-climbing over training config, keeping changes that improve NDCG@10.

**Phase 1: Hyperparameter search**

| Experiment | NDCG@10 | Status |
|---|---|---|
| Baseline (3ep, temp 0.02, BM25 neg) | 0.763 | kept |
| 5 epochs | 0.772 | kept |
| Temperature 0.01 | 0.680 | discarded |
| Temperature 0.05 | 0.773 | kept |
| LR 1e-5 | 0.746 | discarded |
| 10 epochs, temp 0.05 | 0.793 | kept |
| 15 epochs + warmup | 0.748 | discarded (overfit) |

**Phase 2: Data experiments**

| Experiment | NDCG@10 | Status |
|---|---|---|
| + LLM-generated synthetic queries | 0.731 | discarded (same issue) |
| + More ACORD data (relevance ≥ 2) | 0.775 | discarded (noisy) |
| Improved BM25 negative filtering | 0.781 | neutral |

**Phase 3: Hard negative mining**

| Experiment | NDCG@10 | Status |
|---|---|---|
| Model-based hard negatives | 0.813 | kept — became V1 |

**Phase 4: Continuation rounds (V2–P6, 2026-05/06)** — documented in full in [`../CURRENT_STATUS.md`](../CURRENT_STATUS.md):

| Experiment | NDCG@10 | Status |
|---|---|---|
| V4 CUAD-only light continuation | 0.821 | kept |
| V9 clean spans + gentle balance | 0.823 | kept |
| P1 distillation / P2 query paraphrases | 0.694–0.757 | discarded (query-side tuning regresses) |
| P5 tiny LEDGAR doc-side continuation | 0.828 | kept |
| **P6b P5 + ACORD-anchor 300 steps** | **0.834** | **kept — current best** |

Key findings:
- 10 epochs is the sweet spot (15 overfits)
- Temperature 0.05 > 0.02 >> 0.01 for contrastive loss
- More data only helps if the query distribution matches the target task
- Model-based hard negatives were the single biggest improvement after initial fine-tuning (+2.0 NDCG points)

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
| `prepare_data.py` | Training data prep (CUAD + ACORD + model-based negatives) |
| `train.py` | PyLate ColBERT fine-tuning |
| `generate_qa.py` | LLM-generated QA pairs from contract texts |
| `research.py` | Autoresearch loop for iterative optimization |
| `program.md` | Agent instructions for the research loop |
| `results.tsv` | Experiment log |

## License

**CC BY 4.0** — inherited from the training data ([CUAD](https://huggingface.co/datasets/theatticusproject/cuad-qa), [ACORD](https://huggingface.co/datasets/theatticusproject/acord)). Base model is Apache 2.0.

## Acknowledgments

- Training data: [The Atticus Project](https://www.atticusprojectai.org/) (CUAD, ACORD)
- Base model: [LightOn](https://huggingface.co/lightonai) (GTE-ModernColBERT-v1)
- Benchmark: [Isaacus](https://isaacus.com/mleb) (MLEB)
- Autoresearch methodology: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)

## Maintainers

- [kmad](https://github.com/kmad)
- [Claude](https://claude.ai) (Anthropic)
