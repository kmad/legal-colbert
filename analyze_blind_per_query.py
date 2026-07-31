"""Per-query blind eval and bootstrap deltas for overfitting checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pylate import models

from eval_v2 import load_json, maxsim, ndcg_at_k, split_paths


DEFAULT_MODELS = [
    ("v1", "model"),
    ("v9", "output/legal-colbert-v9-spans-capped/final"),
    ("p5", "output/legal-colbert-p5-v9-ledgar-400/final"),
    ("p6b", "output/legal-colbert-p6b-p5-anchor-300/final"),
]


def per_query_ndcg(model_path: str, data_dir: str, split: str, batch_size: int) -> dict[str, float]:
    queries_path, corpus_path, qrels_path = split_paths(Path(data_dir), split)
    queries = load_json(queries_path)
    corpus = load_json(corpus_path)
    qrels = load_json(qrels_path)
    query_ids = list(queries)
    corpus_ids = list(corpus)
    model = models.ColBERT(model_name_or_path=model_path)
    query_embs = model.encode(
        [queries[qid] for qid in query_ids],
        batch_size=batch_size,
        is_query=True,
        show_progress_bar=False,
    )
    corpus_embs = model.encode(
        [corpus[did] for did in corpus_ids],
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=False,
    )
    scores = {}
    for qi, qid in enumerate(query_ids):
        scored = [(di, maxsim(query_embs[qi], corpus_embs[di])) for di in range(len(corpus_ids))]
        scored.sort(key=lambda x: x[1], reverse=True)
        rels = [qrels.get(qid, {}).get(corpus_ids[di], 0) for di, _ in scored]
        scores[qid] = round(ndcg_at_k(rels, 10), 6)
    return scores


def bootstrap_delta(a: dict[str, float], b: dict[str, float], n_boot: int, seed: int) -> dict:
    qids = sorted(set(a) & set(b))
    diffs = np.array([a[qid] - b[qid] for qid in qids], dtype=np.float32)
    rng = np.random.default_rng(seed)
    draws = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return {
        "queries": len(qids),
        "mean_delta": round(float(diffs.mean()), 6),
        "median_query_delta": round(float(np.median(diffs)), 6),
        "queries_positive": int((diffs > 0).sum()),
        "queries_negative": int((diffs < 0).sum()),
        "bootstrap_ci95": [round(float(np.quantile(draws, 0.025)), 6), round(float(np.quantile(draws, 0.975)), 6)],
        "bootstrap_p_delta_gt_0": round(float((draws > 0).mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="eval_blind_edgar_feed")
    parser.add_argument("--split", default="blind_edgar")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="output/blind_per_query_bootstrap.json")
    args = parser.parse_args()

    per_query = {
        label: per_query_ndcg(path, args.data_dir, args.split, args.batch_size)
        for label, path in DEFAULT_MODELS
    }
    categories = load_json(Path(args.data_dir) / f"{args.split}_categories.json")
    result = {
        "data_dir": args.data_dir,
        "split": args.split,
        "per_query_ndcg@10": per_query,
        "categories": categories,
        "deltas": {
            "p6b_vs_p5": bootstrap_delta(per_query["p6b"], per_query["p5"], args.n_boot, args.seed),
            "p6b_vs_v1": bootstrap_delta(per_query["p6b"], per_query["v1"], args.n_boot, args.seed),
            "p5_vs_v1": bootstrap_delta(per_query["p5"], per_query["v1"], args.n_boot, args.seed),
            "v9_vs_v1": bootstrap_delta(per_query["v9"], per_query["v1"], args.n_boot, args.seed),
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["deltas"], indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
