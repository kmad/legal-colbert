"""Evaluate a ColBERT model on the clause eval with macro + tiny-n-robust metrics.

Reports per-category NDCG@10/MAP, the macro mean over all categories, and a
macro mean that excludes tiny-n categories (n_relevant < --min-n) whose noisy
single-digit relevance counts otherwise dominate model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pylate import models


def dcg(rels, k):
    a = np.array(rels[:k], dtype=np.float32)
    return float(np.sum(a / np.log2(np.arange(2, a.size + 2)))) if a.size else 0.0


def ndcg(rels, k):
    ideal = dcg(sorted(rels, reverse=True), k)
    return dcg(rels, k) / ideal if ideal > 0 else 0.0


def maxsim(q, d):
    q = q if isinstance(q, torch.Tensor) else torch.tensor(q)
    d = d if isinstance(d, torch.Tensor) else torch.tensor(d)
    return torch.matmul(q, d.T).max(dim=1).values.sum().item()


def load(p):
    with open(p) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="clause_test")
    ap.add_argument("--min-n", type=int, default=5, help="Exclude categories with fewer relevant docs from the robust macro.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--label", default="")
    ap.add_argument("--output-json", default="")
    args = ap.parse_args()

    data = Path(args.data_dir)
    queries = load(data / f"{args.split}_queries.json")
    corpus = load(data / f"{args.split}_corpus.json")
    qrels = load(data / f"{args.split}_qrels.json")
    categories = load(data / f"{args.split}_categories.json")

    model = models.ColBERT(model_name_or_path=args.model_path)
    qids = list(queries)
    cids = list(corpus)
    q_emb = model.encode([queries[q] for q in qids], batch_size=args.batch_size, is_query=True, show_progress_bar=False)
    c_emb = model.encode([corpus[c] for c in cids], batch_size=args.batch_size, is_query=False, show_progress_bar=False)

    per_cat = {}
    for qi, qid in enumerate(qids):
        scored = sorted(((di, maxsim(q_emb[qi], c_emb[di])) for di in range(len(cids))), key=lambda x: x[1], reverse=True)
        ranked = [cids[di] for di, _ in scored]
        relevant = {d for d, s in qrels.get(qid, {}).items() if s >= 1}
        rels = [1 if d in relevant else 0 for d in ranked]
        ap = 0.0
        if relevant:
            hits = tot = 0
            for rank, d in enumerate(ranked, 1):
                if d in relevant:
                    hits += 1
                    tot += hits / rank
            ap = tot / len(relevant)
        per_cat[categories.get(qid, qid)] = {
            "ndcg@10": round(ndcg(rels, 10), 4),
            "map": round(ap, 4),
            "n_relevant": len(relevant),
        }

    cats = sorted(per_cat)
    all_ndcg = [per_cat[c]["ndcg@10"] for c in cats]
    all_map = [per_cat[c]["map"] for c in cats]
    big = [c for c in cats if per_cat[c]["n_relevant"] >= args.min_n]
    big_ndcg = [per_cat[c]["ndcg@10"] for c in big]
    big_map = [per_cat[c]["map"] for c in big]

    summary = {
        "label": args.label or args.model_path,
        "split": args.split,
        "n_categories": len(cats),
        "macro_ndcg@10": round(float(np.mean(all_ndcg)), 4),
        "macro_map": round(float(np.mean(all_map)), 4),
        f"macro_ndcg@10_n>={args.min_n}": round(float(np.mean(big_ndcg)), 4),
        f"macro_map_n>={args.min_n}": round(float(np.mean(big_map)), 4),
        "n_big_categories": len(big),
        "per_category": dict(sorted(per_cat.items(), key=lambda kv: kv[1]["ndcg@10"])),
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_category"}, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
