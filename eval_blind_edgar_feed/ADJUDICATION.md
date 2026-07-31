# Blind EDGAR Eval — Adjudication Workflow

## Why

The blind set's positives are derived from contract **section headings** only.
That was enough for a quick overfitting diagnostic, but the 2026-07-30 BM25
baseline exposed the limit: BM25 (0.5177) statistically ties the best model
P6b (0.5184) and beats V1/V9/P5 here, because heading-derived labels reward
exactly the lexical query↔heading overlap BM25 exploits. A blind set that
can't separate lexical from semantic retrieval can't gate model promotion.

Standing recommendation (CURRENT_STATUS.md): expand to ~100–200 queries with
adjudicated labels before spending on more training.

## Workflow

1. **Expand the pool** (optional but recommended): re-run
   `build_blind_edgar_pool.py` / `build_blind_edgar_eval.py` to pull more
   fresh SEC EDGAR Exhibit 10 contracts and more clause queries.
2. **Build the sheet**:
   `python build_adjudication_sheet.py --model output/legal-colbert-p6b-p5-anchor-300/final`
   This pools, per query: current heading positives + BM25 top-10 + model
   top-10 into `adjudication_sheet.jsonl`, one row per (query, passage).
   Pooling from multiple retrievers matters — labeling only one system's
   candidates biases recall toward that system.
3. **Adjudicate**: fill `adjudicated_label` with 1 (passage genuinely contains
   the clause type) or 0. Ignore the `heading_label` while judging; it's shown
   for later disagreement analysis. Record edge-case reasoning in
   `adjudicator_note`. An LLM first pass is fine, but a human must review
   disagreements with `heading_label` — those are exactly the cases the
   current eval gets wrong.
4. **Apply**:
   `python build_adjudication_sheet.py --apply eval_blind_edgar_feed/adjudication_sheet.jsonl`
   writes `blind_edgar_qrels_adjudicated.json`. Evaluate against it (and keep
   BM25 as a permanent baseline column — a model that can't clearly beat BM25
   on adjudicated labels is not done).

## Caveats

- Unlabeled rows are treated as negatives on `--apply` — finish the sheet.
- Pooled labeling still under-labels passages no pooled system retrieved;
  with top-10 from ≥2 diverse systems this bias is small at this corpus size.
- Keep the adjudicated qrels file out of any training pipeline. It is an
  eval-only artifact.
