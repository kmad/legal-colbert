"""Build (and apply) a human-adjudication sheet for the blind EDGAR eval.

The blind set's positives are heading-derived weak labels; BM25 ties the best
model on it, so it cannot currently distinguish lexical from semantic
retrieval. This tool supports upgrading it to adjudicated labels:

  1. Pool candidates per query (current heading positives + BM25 top-k, and
     optionally a ColBERT model's top-k) into a JSONL sheet with one row per
     (query, passage) pair and `adjudicated_label: null`.
  2. A human (or reviewed LLM pass) fills adjudicated_label with 0/1.
  3. `--apply` merges filled sheets into blind_edgar_qrels_adjudicated.json,
     which eval scripts can use in place of the heading qrels.

Usage:
    python build_adjudication_sheet.py                       # BM25-pooled sheet
    python build_adjudication_sheet.py --model output/...    # + model top-k
    python build_adjudication_sheet.py --apply sheet.jsonl   # merge labels

See eval_blind_edgar_feed/ADJUDICATION.md for the full workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_bm25 import tokenize_stop
from rank_bm25 import BM25Okapi

DATA = Path(__file__).parent / "eval_blind_edgar_feed"
TOP_K = 10


def build_sheet(out_path: Path, model_paths: list[str], data_dir: Path, prefix: str) -> None:
    def load(name: str):
        with open(data_dir / f"{prefix}_{name}.json") as f:
            return json.load(f)

    queries = load("queries")
    corpus = load("corpus")
    qrels = load("qrels")
    categories = load("categories")

    cids = list(corpus)
    bm25 = BM25Okapi([tokenize_stop(corpus[c]) for c in cids])

    # model tag -> qid -> ranked cids
    model_ranks: dict[str, dict[str, list[str]]] = {}
    if model_paths:
        import torch
        from pylate import models

        def maxsim(q, d):
            q = torch.tensor(q) if not isinstance(q, torch.Tensor) else q
            d = torch.tensor(d) if not isinstance(d, torch.Tensor) else d
            return torch.matmul(q, d.T).max(dim=1).values.sum().item()

        for mi, model_path in enumerate(model_paths):
            tag = Path(model_path.rstrip("/")).parent.name or f"model{mi}"
            model = models.ColBERT(model_name_or_path=model_path)
            q_emb = model.encode(list(queries.values()), is_query=True, show_progress_bar=False)
            c_emb = model.encode([corpus[c] for c in cids], is_query=False, show_progress_bar=False)
            ranks = {}
            for qi, qid in enumerate(queries):
                scored = sorted(range(len(cids)), key=lambda di: -maxsim(q_emb[qi], c_emb[di]))
                ranks[qid] = [cids[di] for di in scored[:TOP_K]]
            model_ranks[tag] = ranks

    rows = []
    for qid, qtext in queries.items():
        scores = bm25.get_scores(tokenize_stop(qtext))
        bm25_top = [cids[i] for i in np.argsort(-scores)[:TOP_K]]
        heading_pos = {d for d, s in qrels.get(qid, {}).items() if s >= 1}

        # Candidate pool: current positives + retriever top-k (union)
        pool: dict[str, list[str]] = {}
        for d in heading_pos:
            pool.setdefault(d, []).append("heading_label")
        for rank, d in enumerate(bm25_top, 1):
            pool.setdefault(d, []).append(f"bm25@{rank}")
        for tag, ranks in model_ranks.items():
            for rank, d in enumerate(ranks.get(qid, []), 1):
                pool.setdefault(d, []).append(f"{tag}@{rank}")

        for did, sources in pool.items():
            rows.append({
                "query_id": qid,
                "category": categories.get(qid, qid),
                "query": qtext,
                "passage_id": did,
                "passage_text": corpus[did],
                "heading_label": 1 if did in heading_pos else 0,
                "sources": sources,
                "adjudicated_label": None,
                "adjudicator_note": "",
            })

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    n_q = len(queries)
    print(f"Wrote {len(rows)} candidate pairs across {n_q} queries to {out_path}")
    print("Fill adjudicated_label (0/1) for every row, then run with --apply.")


def apply_sheet(sheet_path: Path, out_path: Path) -> None:
    qrels: dict[str, dict[str, int]] = {}
    unlabeled = 0
    with open(sheet_path) as f:
        for line in f:
            row = json.loads(line)
            label = row.get("adjudicated_label")
            if label is None:
                unlabeled += 1
                continue
            if int(label) >= 1:
                qrels.setdefault(row["query_id"], {})[row["passage_id"]] = 1
    if unlabeled:
        print(f"WARNING: {unlabeled} rows still unlabeled — they are treated as negatives.")
    with open(out_path, "w") as f:
        json.dump(qrels, f, indent=2)
    n_pos = sum(len(v) for v in qrels.values())
    print(f"Wrote {n_pos} adjudicated positives for {len(qrels)} queries to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None, help="ColBERT model(s) to pool top-k candidates from (repeatable).")
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--prefix", default="blind_edgar")
    ap.add_argument("--out", default="")
    ap.add_argument("--apply", default=None, help="Filled sheet to merge into adjudicated qrels.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else data_dir / "adjudication_sheet.jsonl"
    if args.apply:
        apply_sheet(Path(args.apply), data_dir / f"{args.prefix}_qrels_adjudicated.json")
    else:
        build_sheet(out, args.model or [], data_dir, args.prefix)


if __name__ == "__main__":
    main()
