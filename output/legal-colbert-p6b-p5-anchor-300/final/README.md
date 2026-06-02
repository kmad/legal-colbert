---
license: cc-by-4.0
pipeline_tag: sentence-similarity
library_name: PyLate
base_model: lightonai/GTE-ModernColBERT-v1
datasets:
- theatticusproject/cuad-qa
- theatticusproject/acord
- coastalcph/lex_glue
language:
- en
tags:
- ColBERT
- PyLate
- late-interaction
- sentence-transformers
- feature-extraction
- legal
- contracts
- clause-retrieval
- retrieval
---

# legal-colbert-clause-retriever

A small, open **late-interaction (ColBERT)** retriever fine-tuned for **finding clauses in legal contracts** — termination, assignment, limitation of liability, IP ownership, non-compete, governing law, and ~35 other common provision types. It maps queries and contract passages to sequences of 128-d token vectors and scores them with the MaxSim operator.

It is a continuation fine-tune of [`lightonai/GTE-ModernColBERT-v1`](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) (149M params, ModernBERT-base backbone).

## Results

Evaluated on the **[MLEB](https://isaacus.com/mleb) Contractual Clause Retrieval** task (NDCG@10), the published benchmark for legal contract clause retrieval. Our evaluation reproduces the official leaderboard protocol exactly (BGE-M3 scores 0.7281 through our harness, matching the leaderboard to 4 decimals).

| Metric | Score |
|---|---|
| **NDCG@10** | **0.8338** |
| MAP | 0.7713 |
| Recall@10 | 0.9556 |

**At 149M parameters this is the best accuracy-per-parameter open model on the task** — 3rd of 17 open-source models, ahead of Google's EmbeddingGemma (308M, 0.829) and the same-size legal peer Free Law ModernBERT (0.764), and behind only Qwen3-Embedding-4B/8B (which are 27–53× larger).

![Model size vs NDCG@10 on MLEB Contractual Clause Retrieval (open-source models)](clause_size_vs_ndcg.png)

## Usage

```bash
pip install pylate
```

```python
from pylate import models, rank

model = models.ColBERT("kmad00/legal-colbert-clause-retriever")

# Describe the clause you want to find
queries = model.encode(
    ["This is a contractual provision that limits the maximum liability a party can incur."],
    is_query=True,
)

# Candidate contract passages
documents = model.encode(
    [
        "In no event shall either party's aggregate liability exceed the fees paid in the prior twelve months...",
        "This Agreement shall be governed by the laws of the State of Delaware...",
    ],
    is_query=False,
)

scores = rank.rerank(
    documents_ids=[["0", "1"]],
    queries_embeddings=queries,
    documents_embeddings=[documents],
)
print(scores)
```

Queries can be plain clause names (`"governing law"`), natural-language definitions, or questions — the model is robust to phrasing. Document length 300 tokens, query length 48, output dim 128, similarity = MaxSim.

## Supported clause types

Trained and evaluated on 41 CUAD clause categories plus ACORD drafting queries and LEDGAR provision labels, including: Cap on Liability / Uncapped Liability, IP Ownership Assignment, Joint IP Ownership, License Grant, Non-Compete, Anti-Assignment, Change of Control, Governing Law, Termination for Convenience, Renewal Term, Audit Rights, Insurance, Most Favored Nation, Exclusivity, Liquidated Damages, Source Code Escrow, ROFR/ROFO/ROFN, and more. As a retriever (not a fixed classifier) it also generalizes to clause types outside this set.

## License

**CC BY 4.0.** This model is a derivative of CC BY 4.0 training data (CUAD, ACORD, LEDGAR) and an Apache 2.0 base model. You may use it commercially and non-commercially; attribution is required (see below). No share-alike obligation applies.

## Base model

- [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) — Apache 2.0 (← [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) ← [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base))

## Training data

Produced by a chain of light continuation fine-tunes. Across the full lineage it was trained on the following datasets (and no others):

- **[CUAD](https://huggingface.co/datasets/theatticusproject/cuad-qa)** — CC BY 4.0. Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review," NeurIPS 2021. The Atticus Project.
- **[ACORD](https://huggingface.co/datasets/theatticusproject/acord)** — CC BY 4.0. The Atticus Project, "ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Drafting," 2025.
- **[LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue)** (LexGLUE `ledgar` config) — CC BY 4.0. Tuggener et al., "LEDGAR," LREC 2020; derived from public-domain US SEC EDGAR filings. Chalkidis et al., "LexGLUE," ACL 2022.

Hard negatives were mined with BM25 from each dataset's own corpus. **No MLEB / `isaacus/contractual-clause-retrieval` data and no web-scraped data were used in training** — MLEB is used only as an evaluation benchmark.

## Limitations

- English-language commercial contracts (US-style); other jurisdictions/languages are out of distribution.
- Late-interaction (multi-vector) storage is heavier per document than single-vector embedders.
- The MLEB clause task is small (90 docs); treat ±1–2 points as noise.
- Trained on a narrow set of clause types; confidence is lower on provision types far from the training taxonomy.

## Acknowledgments

- Training data: [The Atticus Project](https://www.atticusprojectai.org/) (CUAD, ACORD); Tuggener et al. & [coastalcph/LexGLUE](https://github.com/coastalcph/lex-glue) (LEDGAR).
- Base model: [LightOn](https://huggingface.co/lightonai) (GTE-ModernColBERT-v1), built with [PyLate](https://github.com/lightonai/pylate).
- Benchmark: [Isaacus](https://isaacus.com/mleb) (MLEB) — evaluation only, not training.

## Full model architecture

```
ColBERT(
  (0): Transformer({'max_seq_length': 299, 'architecture': 'ModernBertModel'})
  (1): Dense({'in_features': 768, 'out_features': 128, 'bias': False, 'activation_function': 'Identity'})
)
```
