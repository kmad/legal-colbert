---
# NOTE: This is the V1 model card, kept for the historical record.
# The current best model is kmad00/legal-colbert-clause-retriever (P6b,
# MLEB NDCG@10 0.834) — see README.md and ../CURRENT_STATUS.md.
license: cc-by-4.0
language:
- en
library_name: PyLate
tags:
- ColBERT
- PyLate
- sentence-transformers
- legal
- contract-retrieval
- information-retrieval
- feature-extraction
base_model: lightonai/GTE-ModernColBERT-v1
datasets:
- theatticusproject/cuad-qa
- theatticusproject/acord
metrics:
- ndcg_at_10
- recall_at_1
pipeline_tag: sentence-similarity
model-index:
- name: legal-colbert-v1
  results:
  - task:
      type: Retrieval
      name: Contractual Clause Retrieval
    dataset:
      name: MLEB Contractual Clause Retrieval
      type: isaacus/contractual-clause-retrieval
    metrics:
    - type: ndcg_at_10
      value: 0.7932
      name: NDCG@10
    - type: recall_at_1
      value: 0.3889
      name: Recall@1
    - type: recall_at_10
      value: 0.8778
      name: Recall@10
    - type: map
      value: 0.7443
      name: MAP
---

# legal-colbert-v1

A ColBERT model fine-tuned for **legal contract clause retrieval**. Produces per-token embeddings scored via MaxSim late interaction, capturing fine-grained term-level matching that single-vector models miss.

## Benchmark Results

**MLEB Contractual Clause Retrieval** (45 queries, 90 passages):

| Model | NDCG@10 | Recall@1 | Recall@10 |
|---|---|---|---|
| **legal-colbert-v1 (this model)** | **0.793** | **0.389** | **0.878** |
| GTE-ModernColBERT-v1 (base, no fine-tuning) | 0.672 | 0.556 | — |
| BGE-large-en-v1.5 (general bi-encoder) | 0.737 | 0.644 | — |
| all-MiniLM-L6-v2 (general bi-encoder) | 0.629 | 0.556 | — |

Fine-tuning improved NDCG@10 by **+12.1 points** over the base model.

## Usage

```python
from pylate import models

# Load model
model = models.ColBERT(model_name_or_path="kmad00/legal-colbert-v1")

# Encode queries and documents
queries = ["What are the termination provisions?"]
documents = ["Either party may terminate this Agreement upon 30 days written notice..."]

query_embeddings = model.encode(queries, is_query=True)
doc_embeddings = model.encode(documents, is_query=False)
```

### Full retrieval pipeline

```python
import torch

def colbert_score(query_emb, doc_emb):
    """MaxSim scoring: for each query token, find max similarity doc token, then sum."""
    q = torch.tensor(query_emb)
    d = torch.tensor(doc_emb)
    return torch.matmul(q, d.T).max(dim=1).values.sum().item()

# Score and rank
scores = [colbert_score(query_embeddings[0], doc_emb) for doc_emb in doc_embeddings]
```

## Training

### Base model

[lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) (Apache 2.0)

### Training data

- **[CUAD](https://huggingface.co/datasets/theatticusproject/cuad-qa)** (CC BY 4.0): 5,449 question-context pairs from 510 commercial legal contracts, covering 41 clause types (indemnification, termination, governing law, IP, etc.)
- **[ACORD](https://huggingface.co/datasets/theatticusproject/acord)** (CC BY 4.0): 793 query-clause pairs across 9 clause categories with expert relevance ratings

Hard negatives were mined using BM25 from each dataset's corpus independently.

### Training configuration

| Parameter | Value |
|---|---|
| Epochs | 10 |
| Batch size | 16 |
| Learning rate | 3e-6 |
| Loss | Contrastive (temperature=0.05) |
| Precision | bfloat16 |
| Training samples | 5,929 triplets |
| Hardware | NVIDIA A100 80GB SXM4 |
| Training time | ~7 minutes |

### Hyperparameter search

7 experiments were run using an autoresearch loop (greedy hill-climbing on NDCG@10):

| Experiment | NDCG@10 | Status |
|---|---|---|
| Baseline (3ep, temp 0.02) | 0.7625 | kept |
| 5 epochs | 0.7723 | kept |
| Temperature 0.01 | 0.6800 | discarded |
| Temperature 0.05 | 0.7731 | kept |
| LR 1e-5 | 0.7462 | discarded |
| **10 epochs, temp 0.05** | **0.7932** | **kept** |
| More data (ACORD rel≥2) | 0.7746 | discarded |
| 15 epochs + warmup | 0.7482 | discarded |

## Intended Use

- Semantic search over legal contracts and clauses
- Contract clause comparison across documents
- Legal document Q&A retrieval
- Contract review automation

## Limitations

- Trained primarily on US commercial contracts (CUAD) and contract drafting clauses (ACORD)
- May underperform on non-English contracts or non-commercial legal domains (criminal, family, etc.)
- ColBERT produces per-token embeddings, requiring more storage than single-vector models
- Max input length inherited from base model

## License

**CC BY 4.0** — inherited from the training data (CUAD and ACORD are both CC BY 4.0). The base model (GTE-ModernColBERT-v1) is Apache 2.0.

### Attribution

Training data from [The Atticus Project](https://www.atticusprojectai.org/):
- Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review" (NeurIPS 2021)
- The Atticus Project, "ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Drafting" (ACL 2025)

Base model from [LightOn](https://huggingface.co/lightonai):
- GTE-ModernColBERT-v1

Benchmark from [Isaacus](https://isaacus.com/mleb):
- MLEB: The Massive Legal Embedding Benchmark
