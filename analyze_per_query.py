"""Per-query winner/loser analysis between two ColBERT models on eval splits.

Runs CPU inference, computes per-query ndcg@10/map, and reports which clause
families each model wins/loses on. Designed to find loser families to mine hard
negatives for, before paying for another training run.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from pylate import models


def dcg_at_k(rels, k):
    a = np.array(rels[:k], dtype=np.float32)
    if a.size == 0:
        return 0.0
    return float(np.sum(a / np.log2(np.arange(2, a.size + 2))))


def ndcg_at_k(rels, k):
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def maxsim(q, d):
    q = q if isinstance(q, torch.Tensor) else torch.tensor(q)
    d = d if isinstance(d, torch.Tensor) else torch.tensor(d)
    return torch.matmul(q, d.T).max(dim=1).values.sum().item()


def category_of(query: str) -> str:
    m = re.search(r'related to "([^"]+)"', query)
    return m.group(1) if m else query[:40]


def load(p):
    with open(p) as f:
        return json.load(f)


def encode_corpus(model, corpus_ids, corpus, batch_size):
    return model.encode(
        [corpus[c] for c in corpus_ids],
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=False,
    )


def per_query(model, queries, corpus, qrels, corpus_embs, corpus_ids, batch_size):
    query_ids = list(queries)
    max_qrel = max((s for r in qrels.values() for s in r.values()), default=0)
    rel_thresh = 1 if max_qrel <= 1 else 3
    q_embs = model.encode(
        [queries[q] for q in query_ids],
        batch_size=batch_size,
        is_query=True,
        show_progress_bar=False,
    )
    out = {}
    for qi, qid in enumerate(query_ids):
        scored = [(di, maxsim(q_embs[qi], corpus_embs[di])) for di in range(len(corpus_ids))]
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [corpus_ids[di] for di, _ in scored]
        rels = [qrels.get(qid, {}).get(did, 0) for did in ranked]
        relevant = {d for d, s in qrels.get(qid, {}).items() if s >= rel_thresh}
        ndcg10 = ndcg_at_k(rels, 10)
        ap = 0.0
        if relevant:
            hits = 0
            tot = 0.0
            for rank, did in enumerate(ranked, 1):
                if did in relevant:
                    hits += 1
                    tot += hits / rank
            ap = tot / len(relevant)
        out[qid] = {
            "category": category_of(queries[qid]),
            "ndcg@10": round(ndcg10, 4),
            "map": round(ap, 4),
            "n_relevant": len(relevant),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_v4_clause_cuad_only")
    ap.add_argument("--model-a", default="model")
    ap.add_argument("--model-b", default="output/legal-colbert-v4-clause-cuud-only/final")
    ap.add_argument("--label-a", default="V1")
    ap.add_argument("--label-b", default="V4")
    ap.add_argument("--splits", nargs="+", default=["cuad_dev", "cuad_test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output-json", default="output/per_query_v1_vs_v4.json")
    args = ap.parse_args()

    data = Path(args.data_dir)
    print(f"loading {args.label_a}: {args.model_a}")
    model_a = models.ColBERT(model_name_or_path=args.model_a)
    print(f"loading {args.label_b}: {args.model_b}")
    model_b = models.ColBERT(model_name_or_path=args.model_b)

    report = {}
    for split in args.splits:
        queries = load(data / f"{split}_queries.json")
        corpus = load(data / f"{split}_corpus.json")
        qrels = load(data / f"{split}_qrels.json")
        corpus_ids = list(corpus)
        print(f"\n[{split}] {len(queries)} queries x {len(corpus_ids)} docs")

        t = time.time()
        ce_a = encode_corpus(model_a, corpus_ids, corpus, args.batch_size)
        print(f"  {args.label_a} corpus encoded in {round(time.time()-t)}s")
        pa = per_query(model_a, queries, corpus, qrels, ce_a, corpus_ids, args.batch_size)
        del ce_a

        t = time.time()
        ce_b = encode_corpus(model_b, corpus_ids, corpus, args.batch_size)
        print(f"  {args.label_b} corpus encoded in {round(time.time()-t)}s")
        pb = per_query(model_b, queries, corpus, qrels, ce_b, corpus_ids, args.batch_size)
        del ce_b

        rows = []
        for qid in queries:
            a, b = pa[qid], pb[qid]
            rows.append({
                "qid": qid,
                "category": a["category"],
                "n_relevant": a["n_relevant"],
                f"ndcg_{args.label_a}": a["ndcg@10"],
                f"ndcg_{args.label_b}": b["ndcg@10"],
                "ndcg_delta": round(b["ndcg@10"] - a["ndcg@10"], 4),
                f"map_{args.label_a}": a["map"],
                f"map_{args.label_b}": b["map"],
                "map_delta": round(b["map"] - a["map"], 4),
            })
        rows.sort(key=lambda r: r["ndcg_delta"])
        report[split] = {
            "mean_ndcg_a": round(np.mean([r[f"ndcg_{args.label_a}"] for r in rows]), 4),
            "mean_ndcg_b": round(np.mean([r[f"ndcg_{args.label_b}"] for r in rows]), 4),
            "rows": rows,
        }

        print(f"  mean ndcg@10: {args.label_a}={report[split]['mean_ndcg_a']} "
              f"{args.label_b}={report[split]['mean_ndcg_b']}")
        print(f"  --- biggest {args.label_b} LOSSES (regressions) ---")
        for r in rows[:8]:
            if r["ndcg_delta"] < 0:
                print(f"    {r['ndcg_delta']:+.3f}  {r['category']:<35} "
                      f"(n_rel={r['n_relevant']}, {r[f'ndcg_{args.label_a}']}->{r[f'ndcg_{args.label_b}']})")
        print(f"  --- biggest {args.label_b} WINS ---")
        for r in rows[::-1][:8]:
            if r["ndcg_delta"] > 0:
                print(f"    {r['ndcg_delta']:+.3f}  {r['category']:<35} "
                      f"(n_rel={r['n_relevant']}, {r[f'ndcg_{args.label_a}']}->{r[f'ndcg_{args.label_b}']})")

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
