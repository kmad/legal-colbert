# Legal ColBERT Autoresearch Program

## Goal

Maximize **NDCG@10** on the MLEB Contractual Clause Retrieval benchmark.

Current baseline: **0.7780**

## Setup

1. You are on branch `autoresearch`
2. Read all source files to understand the current state
3. Log the baseline to `results.tsv` if it doesn't exist yet
4. Begin the experiment loop

## Experiment Loop

Repeat forever:

### 1. Ideate

Choose ONE change to try. Prioritize by expected impact:

**High impact (try first):**
- Add more training data (mine harder negatives using the current model instead of BM25)
- Add the MLEB contractual clause retrieval corpus itself as training signal (careful: don't overfit to the test set — use leave-one-out or cross-validation)
- Increase training epochs (try 5, 10)
- Adjust contrastive loss temperature (try 0.01, 0.05, 0.1)
- Add more ACORD data (all qrels splits, lower relevance threshold)

**Medium impact:**
- Learning rate sweep (1e-6, 3e-6, 1e-5)
- Batch size changes (8, 32)
- Warmup ratio adjustments
- Add data from additional legal datasets on HuggingFace

**Low impact (try later):**
- Different base model
- Gradient accumulation changes
- Weight decay adjustments

### 2. Implement

- Edit ONLY `prepare_data.py` and/or `train.py`
- Do NOT edit `research.py`, `benchmark.py`, or `chunk.py`
- Keep changes small and atomic — one idea per experiment
- `git commit` the change before running

### 3. Run

```bash
python research.py run --experiment "description of what you changed"
```

This will:
- Run `prepare_data.py` if needed
- Train the model
- Benchmark on MLEB
- Log results to `results.tsv`

### 4. Evaluate

- If NDCG@10 **improved**: the change is logged as "kept". Move on to the next idea.
- If NDCG@10 **did not improve**: `git reset --hard` to the previous best commit. The change is logged as "discarded".
- If the run **crashed**: fix the bug and retry, or skip and move on. Log as "crash".

### 5. Log

After each experiment, check `results.tsv` for the running log.

## Constraints

- Do NOT modify the evaluation code (`research.py`, `benchmark.py`)
- Do NOT modify the benchmark dataset
- Do NOT hardcode answers or overfit to the test queries
- Each experiment should train in under 10 minutes
- Keep `prepare_data.py` and `train.py` as the only mutable files
- Always commit before running an experiment

## What You Know

- Base model: `lightonai/GTE-ModernColBERT-v1`
- Training data: CUAD QA (5,449 pairs) + ACORD (793 pairs) with BM25 hard negatives
- Current training: 3 epochs, batch 16, lr 3e-6, temperature 0.02, bf16
- Benchmark: 45 queries, 90 passages, NDCG@10 metric
- The queries are semantic descriptions of clause types (e.g., "This is a contractual provision that permits a contracting party to deduct liabilities...")
- The passages are actual clause text from contracts
- BM25 negatives may not be hard enough — model-based mining could help
