"""Build a document-side LEDGAR continuation dataset for clause retrieval.

This is a P3 candidate after P1/P2 showed that perturbing the query side hurts
MLEB transfer. The intent is to keep V9-style CUAD training intact, then add
contract-provision breadth from LEDGAR with rigid label queries.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from rank_bm25 import BM25Okapi

from prepare_v2_data import (
    SEED,
    balance_records_by_category,
    build_cuad_eval_files,
    expand_triplets,
    load_cuad_records,
    sample_bm25_negatives,
    stable_id,
    tokenize,
    write_json,
)


DEFAULT_LEDGAR_LABELS = [
    "Applicable Laws",
    "Approvals",
    "Arbitration",
    "Assignments",
    "Authority",
    "Authorizations",
    "Binding Effects",
    "Change In Control",
    "Compliance With Laws",
    "Confidentiality",
    "Consent To Jurisdiction",
    "Consents",
    "Entire Agreements",
    "Expenses",
    "Fees",
    "Governing Laws",
    "Indemnifications",
    "Indemnity",
    "Insurances",
    "Intellectual Property",
    "Jurisdictions",
    "Liens",
    "No Defaults",
    "No Waivers",
    "Notices",
    "Payments",
    "Publicity",
    "Remedies",
    "Representations",
    "Severability",
    "Specific Performance",
    "Submission To Jurisdiction",
    "Survival",
    "Taxes",
    "Terminations",
    "Waiver Of Jury Trials",
    "Warranties",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def ledgar_query(label: str) -> str:
    return f'Find provisions related to "{label}".'


def parse_labels(raw: str) -> list[str]:
    if not raw:
        return DEFAULT_LEDGAR_LABELS
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_ledgar_rows(split: str, labels: set[str], min_words: int) -> dict[str, list[str]]:
    ds = load_dataset("lex_glue", "ledgar", split=split)
    label_names = ds.features["label"].names
    rows: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)

    for row in ds:
        label = label_names[int(row["label"])]
        if label not in labels:
            continue
        text = normalize_text(str(row.get("text") or ""))
        if len(text.split()) < min_words or text in seen[label]:
            continue
        rows[label].append(text)
        seen[label].add(text)
    return dict(rows)


def sample_by_label(
    rows: dict[str, list[str]],
    max_per_label: int,
    rng: random.Random,
) -> dict[str, list[str]]:
    sampled = {}
    for label, texts in rows.items():
        texts = list(texts)
        rng.shuffle(texts)
        sampled[label] = texts[:max_per_label] if max_per_label > 0 else texts
    return sampled


def build_ledgar_records(
    rows: dict[str, list[str]],
    negatives_per_positive: int,
    rng: random.Random,
) -> list[dict]:
    corpus = [text for texts in rows.values() for text in texts]
    bm25 = BM25Okapi([tokenize(doc) for doc in corpus]) if corpus else None
    records = []

    for label in sorted(rows):
        positives = set(rows[label])
        query = ledgar_query(label)
        for positive in rows[label]:
            if bm25 is None:
                continue
            negatives = sample_bm25_negatives(
                query,
                positives,
                corpus,
                bm25,
                negatives_per_positive,
                rng,
                max_overlap=1.0,
            )
            if negatives:
                records.append({
                    "source": f"ledgar:{label}",
                    "query": query,
                    "positive": positive,
                    "negatives": negatives,
                    "positive_grade": 1,
                })
    return records


def build_ledgar_eval_files(rows: dict[str, list[str]]) -> tuple[dict, dict, dict]:
    queries = {}
    corpus = {}
    qrels = defaultdict(dict)
    for label, texts in rows.items():
        query = ledgar_query(label)
        qid = stable_id("ledgar-q", query)
        queries[qid] = query
        for text in texts:
            did = stable_id("ledgar-d", text)
            corpus[did] = text
            qrels[qid][did] = 1
    return queries, corpus, {qid: dict(doc_scores) for qid, doc_scores in qrels.items()}


def label_counts(rows: dict[str, list[str]]) -> dict[str, int]:
    return {label: len(texts) for label, texts in sorted(rows.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data_p3_ledgar_docside")
    parser.add_argument("--negatives-per-positive", type=int, default=6)
    parser.add_argument("--cuad-dev-fraction", type=float, default=0.10)
    parser.add_argument("--cuad-test-fraction", type=float, default=0.10)
    parser.add_argument("--cuad-train-window", type=int, default=256)
    parser.add_argument("--cuad-balance-ceiling", type=int, default=400)
    parser.add_argument("--ledgar-max-per-label", type=int, default=200)
    parser.add_argument("--ledgar-eval-max-per-label", type=int, default=40)
    parser.add_argument("--ledgar-repeat", type=int, default=1)
    parser.add_argument("--ledgar-min-words", type=int, default=8)
    parser.add_argument(
        "--ledgar-labels",
        default="",
        help="Comma-separated LEDGAR labels. Default is the curated clause-retrieval label set.",
    )
    args = parser.parse_args()

    if args.ledgar_repeat < 0:
        raise ValueError("--ledgar-repeat must be >= 0")

    rng = random.Random(SEED)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = parse_labels(args.ledgar_labels)
    label_set = set(labels)

    print("Loading CUAD V9-style records...")
    cuad_records, cuad_eval = load_cuad_records(
        args.cuad_dev_fraction,
        args.cuad_test_fraction,
        args.negatives_per_positive,
        rng,
        train_window=args.cuad_train_window,
    )
    cuad_balance_counts = {}
    if args.cuad_balance_ceiling > 0:
        cuad_records, cuad_balance_counts = balance_records_by_category(
            cuad_records,
            args.cuad_balance_ceiling,
            0,
            rng,
        )
    print(f"  CUAD records: {len(cuad_records)}")

    print("Loading LEDGAR train records...")
    ledgar_train_rows = sample_by_label(
        load_ledgar_rows("train", label_set, args.ledgar_min_words),
        args.ledgar_max_per_label,
        rng,
    )
    ledgar_records = build_ledgar_records(
        ledgar_train_rows,
        args.negatives_per_positive,
        rng,
    )
    ledgar_records = ledgar_records * args.ledgar_repeat
    print(f"  LEDGAR records: {len(ledgar_records)}")

    records = cuad_records + ledgar_records
    dataset = expand_triplets(records, rng)
    dataset.save_to_disk(str(out / "train"))

    eval_manifest = {}
    for split, files in cuad_eval.items():
        queries, corpus, qrels = files
        prefix = f"cuad_{split}"
        write_json(out / f"{prefix}_queries.json", queries)
        write_json(out / f"{prefix}_corpus.json", corpus)
        write_json(out / f"{prefix}_qrels.json", qrels)
        eval_manifest[prefix] = {
            "source": f"CUAD held-out {split} contracts",
            "queries": len(queries),
            "corpus": len(corpus),
            "qrels": sum(len(v) for v in qrels.values()),
        }

    for split in ("validation", "test"):
        print(f"Loading LEDGAR {split} eval files...")
        ledgar_eval_rows = sample_by_label(
            load_ledgar_rows(split, label_set, args.ledgar_min_words),
            args.ledgar_eval_max_per_label,
            rng,
        )
        queries, corpus, qrels = build_ledgar_eval_files(ledgar_eval_rows)
        prefix = f"ledgar_{split}"
        write_json(out / f"{prefix}_queries.json", queries)
        write_json(out / f"{prefix}_corpus.json", corpus)
        write_json(out / f"{prefix}_qrels.json", qrels)
        eval_manifest[prefix] = {
            "source": f"LEDGAR {split} split, curated clause labels",
            "queries": len(queries),
            "corpus": len(corpus),
            "qrels": sum(len(v) for v in qrels.values()),
            "label_counts": label_counts(ledgar_eval_rows),
        }

    manifest = {
        "seed": SEED,
        "triplets": len(dataset),
        "multi_negative_records": len(records),
        "negatives_per_positive": args.negatives_per_positive,
        "cuad_train_window": args.cuad_train_window,
        "cuad_balance_ceiling": args.cuad_balance_ceiling,
        "cuad_balance_counts": cuad_balance_counts,
        "ledgar_repeat": args.ledgar_repeat,
        "ledgar_max_per_label": args.ledgar_max_per_label,
        "ledgar_eval_max_per_label": args.ledgar_eval_max_per_label,
        "ledgar_min_words": args.ledgar_min_words,
        "ledgar_labels": labels,
        "ledgar_train_counts": label_counts(ledgar_train_rows),
        "sources": {
            "cuad_records": len(cuad_records),
            "ledgar_records": len(ledgar_records),
        },
        "eval_splits": eval_manifest,
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
