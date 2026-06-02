"""Build PyLate knowledge-distillation data for the clause retriever (P1).

Emits the three datasets PyLate's KDProcessing expects:
  - queries:   {query_id, text}
  - documents: {document_id, text}
  - train:     {query_id, document_ids: [...]}   (teacher scores added later by
               score_teacher.py, which appends a `scores` column)

Each train row is one (clause-category query, positive span) pair plus a fixed
number of hard negatives, so distillation teaches the model to rank the positive
above confusable clauses according to the cross-encoder teacher.

Design choices (informed by P0):
  - Queries are MLEB-style natural-language definitions (the objective's style),
    not the rigid CUAD template.
  - Positives are clean, word-boundary-aligned spans (no +/-512 label noise).
  - The same SEED-42 held-out CUAD contracts are excluded, so train stays
    disjoint from the eval splits.
  - Per-category record cap (balance) to avoid frequent clauses dominating.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_dataset
from rank_bm25 import BM25Okapi

from build_clause_eval import CATEGORY_DEFINITIONS, category_from_question
from prepare_v2_data import SEED, aligned_span, stable_id, tokenize, sample_bm25_negatives


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data_p1_distill")
    ap.add_argument("--dev-fraction", type=float, default=0.10)
    ap.add_argument("--test-fraction", type=float, default=0.10)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--negatives", type=int, default=15, help="Hard negatives per positive (candidates = 1 + this).")
    ap.add_argument("--balance-ceiling", type=int, default=400, help="Max train (query,pos) pairs per category.")
    args = ap.parse_args()

    ds = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    rng = random.Random(SEED)

    examples = []
    contract_to_examples = defaultdict(list)
    for ex in ds:
        answers = ex.get("answers") or {}
        texts = answers.get("text") or []
        starts = answers.get("answer_start") or []
        if not texts or not starts:
            continue
        context = ex.get("context", "")
        if not context:
            continue
        answer_text = texts[0]
        answer_start = int(starts[0])
        start = max(0, answer_start - 512)
        end = min(len(context), answer_start + len(answer_text) + 512)
        if len((context[start:end]).split()) < 8:
            continue
        contract_id = (
            ex.get("title")
            or ex.get("contract_name")
            or str(ex.get("id", "")).split("_")[0]
            or stable_id("cuad-contract", context[:2000])
        )
        cat = category_from_question(ex["question"])
        clean = aligned_span(context, answer_start, answer_text, args.window)
        examples.append({"category": cat, "span": clean, "contract_id": contract_id})
        contract_to_examples[contract_id].append(examples[-1])

    contracts = list(contract_to_examples)
    rng.shuffle(contracts)
    n_dev = max(1, int(len(contracts) * args.dev_fraction))
    n_test = max(1, int(len(contracts) * args.test_fraction))
    heldout = set(contracts[: n_dev + n_test])

    # Train pool: in-train clean spans with a known clause category.
    pool = [
        e for e in examples
        if e["contract_id"] not in heldout and e["category"] in CATEGORY_DEFINITIONS
        and e["span"] and len(e["span"].split()) >= 8
    ]

    # Per-category balance cap on (query, positive) pairs.
    by_cat = defaultdict(list)
    for e in pool:
        by_cat[e["category"]].append(e)
    capped = []
    for cat, rows in by_cat.items():
        rng.shuffle(rows)
        capped.extend(rows[: args.balance_ceiling] if args.balance_ceiling > 0 else rows)
    rng.shuffle(capped)

    # Build document store and BM25 over all train spans.
    all_spans = list({e["span"] for e in pool})
    span_to_id = {s: stable_id("p1-d", s) for s in all_spans}
    bm25 = BM25Okapi([tokenize(s) for s in all_spans])
    cat_to_query = {c: CATEGORY_DEFINITIONS[c] for c in by_cat}
    query_to_positive_spans = defaultdict(set)
    for e in pool:
        query_to_positive_spans[cat_to_query[e["category"]]].add(e["span"])

    queries_rows = []
    qtext_to_id = {}
    for cat, qtext in cat_to_query.items():
        qid = stable_id("p1-q", cat)
        qtext_to_id[qtext] = qid
        queries_rows.append({"query_id": qid, "text": qtext})

    train_rows = []
    used_doc_ids = set()
    for e in capped:
        qtext = cat_to_query[e["category"]]
        pos_span = e["span"]
        negs = sample_bm25_negatives(
            qtext, query_to_positive_spans[qtext], all_spans, bm25, args.negatives, rng
        )
        cand_spans = [pos_span] + [n for n in negs if n != pos_span][: args.negatives]
        if len(cand_spans) < 2:
            continue
        doc_ids = [span_to_id[s] for s in cand_spans]
        used_doc_ids.update(doc_ids)
        train_rows.append({"query_id": qtext_to_id[qtext], "document_ids": doc_ids})

    documents_rows = [{"document_id": span_to_id[s], "text": s} for s in all_spans if span_to_id[s] in used_doc_ids]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(queries_rows).save_to_disk(str(out / "queries"))
    Dataset.from_list(documents_rows).save_to_disk(str(out / "documents"))
    Dataset.from_list(train_rows).save_to_disk(str(out / "train"))
    manifest = {
        "seed": SEED,
        "window": args.window,
        "negatives": args.negatives,
        "balance_ceiling": args.balance_ceiling,
        "n_queries": len(queries_rows),
        "n_documents": len(documents_rows),
        "n_train_rows": len(train_rows),
        "candidates_per_row": args.negatives + 1,
        "query_style": "mleb_definitions",
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
