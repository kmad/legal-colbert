"""Evaluate a ColBERT model on BEIR-style graded or binary eval splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pylate import models


def dcg_at_k(rels: list[float], k: int) -> float:
    rels_arr = np.array(rels[:k], dtype=np.float32)
    if rels_arr.size == 0:
        return 0.0
    return float(np.sum(rels_arr / np.log2(np.arange(2, rels_arr.size + 2))))


def ndcg_at_k(rels: list[float], k: int) -> float:
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def maxsim(q_emb, d_emb) -> float:
    q = torch.tensor(q_emb) if not isinstance(q_emb, torch.Tensor) else q_emb
    d = torch.tensor(d_emb) if not isinstance(d_emb, torch.Tensor) else d_emb
    return torch.matmul(q, d.T).max(dim=1).values.sum().item()


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def split_paths(data: Path, split: str) -> tuple[Path, Path, Path]:
    return (
        data / f"{split}_queries.json",
        data / f"{split}_corpus.json",
        data / f"{split}_qrels.json",
    )


def available_splits(data: Path) -> list[str]:
    splits = []
    for path in sorted(data.glob("*_queries.json")):
        split = path.name[: -len("_queries.json")]
        _, corpus_path, qrels_path = split_paths(data, split)
        if corpus_path.exists() and qrels_path.exists():
            splits.append(split)
    return splits


def evaluate_split(
    model: models.ColBERT,
    data_dir: str = "data_v2",
    split: str = "dev",
    batch_size: int = 32,
) -> dict[str, float]:
    data = Path(data_dir)
    queries_path, corpus_path, qrels_path = split_paths(data, split)
    queries = load_json(queries_path)
    corpus = load_json(corpus_path)
    qrels = load_json(qrels_path)

    query_ids = list(queries)
    corpus_ids = list(corpus)
    max_qrel = max((score for rows in qrels.values() for score in rows.values()), default=0)
    relevant_threshold = 1 if max_qrel <= 1 else 3
    strong_threshold = 1 if max_qrel <= 1 else 4

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

    ndcg10 = []
    ndcg5 = []
    ap = []
    recall_at = {1: [], 5: [], 10: []}
    precision4_at5 = []

    for qi, qid in enumerate(query_ids):
        scored = [(di, maxsim(query_embs[qi], corpus_embs[di])) for di in range(len(corpus_ids))]
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [corpus_ids[di] for di, _ in scored]
        rels = [qrels.get(qid, {}).get(did, 0) for did in ranked]
        relevant = {
            did
            for did, score in qrels.get(qid, {}).items()
            if score >= relevant_threshold
        }
        strong = {
            did
            for did, score in qrels.get(qid, {}).items()
            if score >= strong_threshold
        }

        ndcg10.append(ndcg_at_k(rels, 10))
        ndcg5.append(ndcg_at_k(rels, 5))

        if relevant:
            hits = 0
            total_precision = 0.0
            for rank, did in enumerate(ranked, 1):
                if did in relevant:
                    hits += 1
                    total_precision += hits / rank
            ap.append(total_precision / len(relevant))
            for k in recall_at:
                recall_at[k].append(len(relevant & set(ranked[:k])) / len(relevant))

        if strong:
            precision4_at5.append(len(strong & set(ranked[:5])) / 5.0)

    metrics = {
        "ndcg@5": float(np.mean(ndcg5)),
        "ndcg@10": float(np.mean(ndcg10)),
        "map": float(np.mean(ap)) if ap else 0.0,
        "recall@1": float(np.mean(recall_at[1])) if recall_at[1] else 0.0,
        "recall@5": float(np.mean(recall_at[5])) if recall_at[5] else 0.0,
        "recall@10": float(np.mean(recall_at[10])) if recall_at[10] else 0.0,
        "precision4@5": float(np.mean(precision4_at5)) if precision4_at5 else 0.0,
        "queries": len(query_ids),
        "corpus": len(corpus_ids),
        "qrels": sum(len(rows) for rows in qrels.values()),
        "relevant_threshold": relevant_threshold,
    }
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}


def evaluate(
    model_path: str,
    data_dir: str = "data_v2",
    batch_size: int = 32,
    split: str = "dev",
) -> dict[str, dict[str, float]] | dict[str, float]:
    data = Path(data_dir)
    model = models.ColBERT(model_name_or_path=model_path)
    if split == "all":
        return {
            name: evaluate_split(model, data_dir, name, batch_size)
            for name in available_splits(data)
        }
    return evaluate_split(model, data_dir, split, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model")
    parser.add_argument("--data-dir", default="data_v2")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    metrics = evaluate(args.model_path, args.data_dir, args.batch_size, args.split)
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
