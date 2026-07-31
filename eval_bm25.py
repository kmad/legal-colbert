"""BM25 retrieval baseline over the same eval protocol as the ColBERT models.

Runs BM25Okapi on the four gates used for model selection/diagnostics:
  - MLEB Contractual Clause Retrieval (HF, same revision as benchmark.py)
  - eval_clause_mleb clause_test (macro + n>=5 robust, eval_clause.py protocol)
  - data_v2 acord_test (graded qrels, eval_v2.py protocol, relevant >= 3)
  - eval_blind_edgar_feed blind_edgar (macro over categories)

Two tokenizations: "plain" (lowercase \\w+) and "stop" (legal stopwords removed,
same list prepare_v2_data.py uses for hard-negative mining).

Usage:
    python eval_bm25.py --output-json output/bm25_baseline.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from prepare_v2_data import STOPWORDS

HERE = Path(__file__).parent


def tokenize_plain(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def tokenize_stop(text: str) -> list[str]:
    return [w for w in tokenize_plain(text) if w not in STOPWORDS and len(w) > 2]


def dcg(rels, k):
    a = np.array(rels[:k], dtype=np.float64)
    return float(np.sum(a / np.log2(np.arange(2, a.size + 2)))) if a.size else 0.0


def ndcg(rels, k):
    ideal = dcg(sorted(rels, reverse=True), k)
    return dcg(rels, k) / ideal if ideal > 0 else 0.0


def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def bm25_rank(tokenizer, corpus_ids, corpus_texts, query_text, bm25):
    scores = bm25.get_scores(tokenizer(query_text))
    order = np.argsort(-scores)
    return [corpus_ids[i] for i in order]


def rank_all(tokenizer, queries: dict, corpus: dict):
    corpus_ids = list(corpus)
    bm25 = BM25Okapi([tokenizer(corpus[c]) for c in corpus_ids])
    return {
        qid: bm25_rank(tokenizer, corpus_ids, None, qtext, bm25)
        for qid, qtext in queries.items()
    }


def query_metrics(ranked, qrel_row, relevant_threshold=1, graded_ndcg=True):
    """Metrics for one query. graded_ndcg uses raw scores in NDCG (benchmark.py /
    eval_v2.py); otherwise binary rels (eval_clause.py)."""
    relevant = {d for d, s in qrel_row.items() if s >= relevant_threshold}
    if graded_ndcg:
        rels = [qrel_row.get(d, 0) for d in ranked]
    else:
        rels = [1 if d in relevant else 0 for d in ranked]
    out = {"ndcg@10": ndcg(rels, 10)}
    ap = 0.0
    rec = {}
    if relevant:
        hits = tot = 0
        for rank, d in enumerate(ranked, 1):
            if d in relevant:
                hits += 1
                tot += hits / rank
        ap = tot / len(relevant)
        for k in (1, 5, 10):
            rec[f"recall@{k}"] = len(relevant & set(ranked[:k])) / len(relevant)
    out["map"] = ap
    out.update(rec)
    return out, bool(relevant)


def eval_mleb(tokenizer):
    """MLEB contractual clause retrieval — benchmark.py protocol (graded NDCG,
    mean over queries, relevant = score > 0)."""
    from datasets import load_dataset

    DATASET_ID = "isaacus/contractual-clause-retrieval"
    DATASET_REV = "48ed7bcb1f50896a0f71461a04b2df0ca84329d9"
    corpus_ds = load_dataset(DATASET_ID, "corpus", revision=DATASET_REV, split="corpus")
    queries_ds = load_dataset(DATASET_ID, "queries", revision=DATASET_REV, split="queries")
    qrels_ds = load_dataset(DATASET_ID, "default", revision=DATASET_REV, split="test")

    corpus = {r["_id"]: r["text"] for r in corpus_ds}
    queries = {r["_id"]: r["text"] for r in queries_ds}
    qrels = {}
    for r in qrels_ds:
        qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = r["score"]

    ranked_all = rank_all(tokenizer, queries, corpus)
    ndcgs, aps, recs = [], [], {1: [], 5: [], 10: []}
    for qid, ranked in ranked_all.items():
        m, has_rel = query_metrics(ranked, qrels.get(qid, {}), 1, graded_ndcg=True)
        ndcgs.append(m["ndcg@10"])
        if has_rel:
            aps.append(m["map"])
            for k in (1, 5, 10):
                recs[k].append(m[f"recall@{k}"])
    return {
        "ndcg@10": round(float(np.mean(ndcgs)), 4),
        "map": round(float(np.mean(aps)), 4),
        "recall@1": round(float(np.mean(recs[1])), 4),
        "recall@5": round(float(np.mean(recs[5])), 4),
        "recall@10": round(float(np.mean(recs[10])), 4),
        "n_queries": len(ranked_all),
    }


def eval_macro_split(tokenizer, data_dir: Path, split: str, min_n: int = 5):
    """eval_clause.py protocol: binary rels, per-category, macro + n>=min_n."""
    queries = load_json(data_dir / f"{split}_queries.json")
    corpus = load_json(data_dir / f"{split}_corpus.json")
    qrels = load_json(data_dir / f"{split}_qrels.json")
    categories = load_json(data_dir / f"{split}_categories.json")

    ranked_all = rank_all(tokenizer, queries, corpus)
    per_cat = {}
    for qid, ranked in ranked_all.items():
        m, _ = query_metrics(ranked, qrels.get(qid, {}), 1, graded_ndcg=False)
        relevant_n = sum(1 for s in qrels.get(qid, {}).values() if s >= 1)
        per_cat[categories.get(qid, qid)] = {
            "ndcg@10": round(m["ndcg@10"], 4),
            "map": round(m["map"], 4),
            "n_relevant": relevant_n,
        }
    cats = sorted(per_cat)
    big = [c for c in cats if per_cat[c]["n_relevant"] >= min_n]
    return {
        "macro_ndcg@10": round(float(np.mean([per_cat[c]["ndcg@10"] for c in cats])), 4),
        "macro_map": round(float(np.mean([per_cat[c]["map"] for c in cats])), 4),
        f"macro_ndcg@10_n>={min_n}": round(float(np.mean([per_cat[c]["ndcg@10"] for c in big])), 4) if big else None,
        f"macro_map_n>={min_n}": round(float(np.mean([per_cat[c]["map"] for c in big])), 4) if big else None,
        "n_categories": len(cats),
        "n_big_categories": len(big),
    }


def eval_v2_split(tokenizer, data_dir: Path, split: str):
    """eval_v2.py protocol: graded NDCG, relevant threshold 3 for graded qrels."""
    queries = load_json(data_dir / f"{split}_queries.json")
    corpus = load_json(data_dir / f"{split}_corpus.json")
    qrels = load_json(data_dir / f"{split}_qrels.json")
    max_qrel = max((s for row in qrels.values() for s in row.values()), default=0)
    threshold = 1 if max_qrel <= 1 else 3

    ranked_all = rank_all(tokenizer, queries, corpus)
    ndcgs, aps, recs = [], [], {1: [], 5: [], 10: []}
    for qid, ranked in ranked_all.items():
        m, has_rel = query_metrics(ranked, qrels.get(qid, {}), threshold, graded_ndcg=True)
        ndcgs.append(m["ndcg@10"])
        if has_rel:
            aps.append(m["map"])
            for k in (1, 5, 10):
                recs[k].append(m[f"recall@{k}"])
    return {
        "ndcg@10": round(float(np.mean(ndcgs)), 4),
        "map": round(float(np.mean(aps)), 4),
        "recall@10": round(float(np.mean(recs[10])), 4),
        "relevant_threshold": threshold,
        "n_queries": len(ranked_all),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-json", default="output/bm25_baseline.json")
    args = ap.parse_args()

    variants = {"bm25_plain": tokenize_plain, "bm25_stopfiltered": tokenize_stop}
    results = {}
    for name, tok in variants.items():
        print(f"\n=== {name} ===", flush=True)
        r = {}
        print("MLEB contractual clause retrieval...", flush=True)
        r["mleb"] = eval_mleb(tok)
        print(json.dumps(r["mleb"]))
        print("eval_clause_mleb clause_test...", flush=True)
        r["clause_mleb_test"] = eval_macro_split(tok, HERE / "eval_clause_mleb", "clause_test")
        print(json.dumps(r["clause_mleb_test"]))
        print("acord_test...", flush=True)
        r["acord_test"] = eval_v2_split(tok, HERE / "data_v2", "acord_test")
        print(json.dumps(r["acord_test"]))
        print("blind_edgar...", flush=True)
        r["blind_edgar"] = eval_macro_split(tok, HERE / "eval_blind_edgar_feed", "blind_edgar")
        print(json.dumps(r["blind_edgar"]))
        results[name] = r

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
