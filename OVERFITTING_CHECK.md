# Overfitting Check: Fresh SEC Blind Eval

Date: 2026-06-02

## Question

Are the recent MLEB improvements real clause-retrieval improvements, or are we overfitting to the small benchmark through repeated tuning?

## Blind Data Built

I built a fresh SEC EDGAR blind diagnostic from current 8-K Exhibit 10 filings. The labels are weak but independent: a section is relevant only when the contract itself has a matching section heading. No current retriever generated labels.

Artifacts:

- `build_blind_edgar_pool.py`
- `build_blind_edgar_eval.py`
- `eval_blind_edgar_feed/raw_contracts.jsonl`
- `eval_blind_edgar_feed/pool_manifest.json`
- `eval_blind_edgar_feed/eval_manifest.json`
- `eval_blind_edgar_feed/heading_evidence.json`
- `output/blind_overfit_assessment.json`
- `output/blind_per_query_bootstrap.json`

Blind set summary:

- Source: SEC EDGAR current 8-K feed, Exhibit 10 documents
- Acquisition scanned 16 current-feed pages and found 28 qualifying Exhibit 10 contracts under the length/dedup filters.
- Contracts: 28
- Corpus sections: 342
- Clause queries: 12
- Heading-labeled positives: 56
- Active clause types: assignment, change of control, confidentiality, dispute resolution, governing law, indemnification, intellectual property, notices, payment, representations/warranties, survival, termination

## Results

Metric: NDCG@10

| model | MLEB | fresh SEC blind | MLEB delta vs v1 | blind delta vs v1 |
|---|---:|---:|---:|---:|
| v1 | 0.8125 | 0.4834 | +0.0000 | +0.0000 |
| v9 | 0.8234 | 0.4448 | +0.0109 | -0.0386 |
| p5 | 0.8277 | 0.4664 | +0.0153 | -0.0170 |
| p6b | 0.8338 | 0.5184 | +0.0214 | +0.0350 |

Bootstrap over the 12 blind clause queries:

- p6b vs p5: mean delta +0.0521, 95% CI [+0.0013, +0.1224], P(delta > 0) = 0.9982
- p6b vs v1: mean delta +0.0351, 95% CI [-0.0695, +0.1485], P(delta > 0) = 0.7360
- p5 vs v1: mean delta -0.0170, 95% CI [-0.0955, +0.0481], P(delta > 0) = 0.3362
- v9 vs v1: mean delta -0.0386, 95% CI [-0.1156, +0.0219], P(delta > 0) = 0.1364

Other diagnostics:

- ContractNLI: p6b slightly regresses vs p5, 0.3253 vs 0.3339 macro NDCG@10. This set is NDA-heavy and not the primary target.
- ACORD test: p6b is effectively flat/slightly down vs p5, 0.2466 vs 0.2473 NDCG@10. ACORD remains a drafting-query diagnostic, not clause extraction.
- LEDGAR validation: p6b improves vs p5, 0.5910 vs 0.5696 NDCG@10.

## Verdict

The earlier v9/p5 improvements look benchmark-fragile: MLEB improved, but the fresh SEC blind set got worse versus v1.

p6b is different. Its MLEB improvement over p5 transfers to the fresh SEC blind diagnostic, and the p6b-vs-p5 bootstrap interval is positive. That means the current best checkpoint is not just MLEB overfit relative to p5.

The broader claim that p6b is clearly better than v1 on unseen legal clause retrieval is still not fully proven. It is positive on this blind set, but the p6b-vs-v1 confidence interval is wide because the blind set only has 12 clause queries and heading-derived weak labels.

## Next

1. Keep p6b as the current best checkpoint.
2. Stop selecting models solely on MLEB.
3. Expand the fresh blind set to 50-100 SEC contracts.
4. Human- or strong-LLM-adjudicate at least 150-250 exact spans, including absent clauses and hard negatives.
5. Re-run the same comparison before launching more paid training.
