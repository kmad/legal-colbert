"""Unified evaluation harness: score one ColBERT model on every gate, once.

Replaces the scattered per-gate scripts (benchmark.py, eval_clause.py,
eval_v2.py) with a single pass that loads the model once and writes one JSON
per model to output/evals/<label>.json. The metric code below reproduces the
protocols of those scripts exactly — eval_bm25.py is the cross-check — so new
numbers stay comparable with everything already stored in output/.

Protocols
  mleb_graded_mean  benchmark.py: graded NDCG@10 (raw qrel scores), mean over
                    queries; MAP/recall over queries that have relevant docs.
  graded_mean       eval_v2.py: same, plus ndcg@5 and precision4@5; relevance
                    threshold 3 (strong 4) when qrels are graded 0-4, else 1.
  macro_binary      eval_clause.py: binary NDCG@10/MAP per category, rounded to
                    4dp, then macro-averaged; plus a macro over categories with
                    n_relevant >= 5 (tiny-n categories are label noise).

Usage:
    HF_HUB_OFFLINE=1 .venv/bin/python evaluate.py --model output/<run>/final
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent

MLEB_DATASET_ID = "isaacus/contractual-clause-retrieval"
MLEB_DATASET_REV = "48ed7bcb1f50896a0f71461a04b2df0ca84329d9"

# Models that trained on the 41 CATEGORY_DEFINITIONS query strings. eval_clause_mleb
# is built from those exact strings, so scoring these models on the clause_mleb gate
# measures query memorization, not retrieval skill: p1-contrastive scored 0.753 there
# while landing at 0.750 on MLEB, below the 0.823 baseline it was supposed to beat.
# See ../CURRENT_STATUS.md "Eval-leakage lesson". Pass --allow-leaked to override.
TAINTED_MODELS = ["*p1-distill*", "*p1-contrastive*"]
TAINTED_GATES = {"clause_mleb"}

GATES = {
    "mleb": {
        "protocol": "mleb_graded_mean",
        "source": f"{MLEB_DATASET_ID}@{MLEB_DATASET_REV[:8]}",
        "headline": "ndcg@10",
    },
    "clause_mleb": {
        "protocol": "macro_binary",
        "dir": "eval_clause_mleb",
        "split": "clause_test",
        "headline": "macro_ndcg@10",
    },
    "acord_test": {
        "protocol": "graded_mean",
        "dir": "data_v2",
        "split": "acord_test",
        "headline": "ndcg@10",
    },
    "blind_edgar": {
        "protocol": "macro_binary",
        "dir": "eval_blind_edgar_feed",
        "split": "blind_edgar",
        "headline": "macro_ndcg@10",
        # Stored per-model blind_edgar JSONs were produced by eval_v2.py, the BM25
        # baseline by the macro path. qrels are binary and categories are 1:1 with
        # queries, so ndcg@10 agrees; report both so either is directly comparable.
        "extra_protocols": ["graded_mean"],
    },
    "contractnli": {
        "protocol": "macro_binary",
        "dir": "eval_contractnli_blind",
        "split": "contractnli_test",
        "headline": "macro_ndcg@10",
    },
    "ledgar_validation": {
        "protocol": "graded_mean",
        "dir": "data_p3_ledgar_docside",
        "split": "ledgar_validation",
        "headline": "ndcg@10",
    },
}

MIN_N = 5  # eval_clause.py --min-n default


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def dcg(rels, k: int) -> float:
    a = np.array(rels[:k], dtype=np.float64)
    return float(np.sum(a / np.log2(np.arange(2, a.size + 2)))) if a.size else 0.0


def ndcg(rels, k: int) -> float:
    ideal = dcg(sorted(rels, reverse=True), k)
    return dcg(rels, k) / ideal if ideal > 0 else 0.0


def average_precision(ranked, relevant: set) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, did in enumerate(ranked, 1):
        if did in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def rank_all(model, queries: dict, corpus: dict, batch_size: int) -> dict[str, list[str]]:
    """MaxSim ranking of the whole corpus for every query."""
    qids, cids = list(queries), list(corpus)
    q_embs = model.encode(
        [queries[q] for q in qids],
        batch_size=batch_size,
        is_query=True,
        show_progress_bar=False,
    )
    c_embs = model.encode(
        [corpus[c] for c in cids],
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=False,
    )
    qs = [torch.as_tensor(q) for q in q_embs]
    ds = [torch.as_tensor(d) for d in c_embs]
    ranked = {}
    for qi, qid in enumerate(qids):
        scored = [(di, maxsim(qs[qi], ds[di])) for di in range(len(cids))]
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked[qid] = [cids[di] for di, _ in scored]
    return ranked


def maxsim(q, d) -> float:
    """One (query, document) score, computed pairwise exactly as the scripts this
    replaces do. A blocked/batched equivalent is ~4x faster but reorders the float32
    sum over query tokens, which flips near-tied documents and perturbs MAP in the
    4th decimal — not worth breaking comparability with stored results, since
    encoding dominates the runtime anyway."""
    return torch.matmul(q, d.T).max(dim=1).values.sum().item()


def graded_mean_metrics(ranked_all: dict, qrels: dict, n_corpus: int, threshold: int | None) -> dict:
    """eval_v2.py / benchmark.py protocol: graded NDCG, mean over queries."""
    max_qrel = max((s for row in qrels.values() for s in row.values()), default=0)
    if threshold is None:
        threshold = 1 if max_qrel <= 1 else 3
    strong_threshold = 1 if max_qrel <= 1 else 4

    ndcg10, ndcg5, aps = [], [], []
    recall_at = {1: [], 5: [], 10: []}
    precision4_at5 = []
    for qid, ranked in ranked_all.items():
        row = qrels.get(qid, {})
        rels = [row.get(did, 0) for did in ranked]
        relevant = {d for d, s in row.items() if s >= threshold}
        strong = {d for d, s in row.items() if s >= strong_threshold}
        ndcg10.append(ndcg(rels, 10))
        ndcg5.append(ndcg(rels, 5))
        if relevant:
            aps.append(average_precision(ranked, relevant))
            for k in recall_at:
                recall_at[k].append(len(relevant & set(ranked[:k])) / len(relevant))
        if strong:
            precision4_at5.append(len(strong & set(ranked[:5])) / 5.0)

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731
    return {
        "ndcg@5": round(mean(ndcg5), 4),
        "ndcg@10": round(mean(ndcg10), 4),
        "map": round(mean(aps), 4),
        "recall@1": round(mean(recall_at[1]), 4),
        "recall@5": round(mean(recall_at[5]), 4),
        "recall@10": round(mean(recall_at[10]), 4),
        "precision4@5": round(mean(precision4_at5), 4),
        "queries": len(ranked_all),
        "corpus": n_corpus,
        "qrels": sum(len(row) for row in qrels.values()),
        "relevant_threshold": threshold,
    }


def macro_binary_metrics(ranked_all: dict, qrels: dict, categories: dict) -> dict:
    """eval_clause.py protocol: binary per-category metrics, macro-averaged."""
    per_cat = {}
    for qid, ranked in ranked_all.items():
        row = qrels.get(qid, {})
        relevant = {d for d, s in row.items() if s >= 1}
        rels = [1 if d in relevant else 0 for d in ranked]
        per_cat[categories.get(qid, qid)] = {
            "ndcg@10": round(ndcg(rels, 10), 4),
            "map": round(average_precision(ranked, relevant), 4),
            "n_relevant": len(relevant),
        }

    cats = sorted(per_cat)
    big = [c for c in cats if per_cat[c]["n_relevant"] >= MIN_N]
    macro = lambda names, key: (  # noqa: E731
        round(float(np.mean([per_cat[c][key] for c in names])), 4) if names else None
    )
    return {
        "macro_ndcg@10": macro(cats, "ndcg@10"),
        "macro_map": macro(cats, "map"),
        f"macro_ndcg@10_n>={MIN_N}": macro(big, "ndcg@10"),
        f"macro_map_n>={MIN_N}": macro(big, "map"),
        "n_categories": len(cats),
        "n_big_categories": len(big),
        "per_category": dict(sorted(per_cat.items(), key=lambda kv: kv[1]["ndcg@10"])),
    }


def load_mleb() -> tuple[dict, dict, dict]:
    from datasets import load_dataset

    kw = {"revision": MLEB_DATASET_REV}
    corpus_ds = load_dataset(MLEB_DATASET_ID, "corpus", split="corpus", **kw)
    queries_ds = load_dataset(MLEB_DATASET_ID, "queries", split="queries", **kw)
    qrels_ds = load_dataset(MLEB_DATASET_ID, "default", split="test", **kw)
    corpus = {r["_id"]: r["text"] for r in corpus_ds}
    queries = {r["_id"]: r["text"] for r in queries_ds}
    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        qrels.setdefault(r["query-id"], {})[r["corpus-id"]] = r["score"]
    return corpus, queries, qrels


def load_gate(name: str, cfg: dict) -> tuple[dict, dict, dict, dict]:
    """Returns (queries, corpus, qrels, categories)."""
    if cfg["protocol"] == "mleb_graded_mean":
        corpus, queries, qrels = load_mleb()
        return queries, corpus, qrels, {}
    data, split = HERE / cfg["dir"], cfg["split"]
    categories_path = data / f"{split}_categories.json"
    return (
        load_json(data / f"{split}_queries.json"),
        load_json(data / f"{split}_corpus.json"),
        load_json(data / f"{split}_qrels.json"),
        load_json(categories_path) if categories_path.exists() else {},
    )


def compute(protocol: str, ranked_all, qrels, categories, n_corpus) -> dict:
    if protocol == "macro_binary":
        return macro_binary_metrics(ranked_all, qrels, categories)
    # benchmark.py treats MLEB relevance as score > 0, i.e. threshold 1 for its
    # integer scores; eval_v2.py infers the threshold from the qrel range.
    threshold = 1 if protocol == "mleb_graded_mean" else None
    return graded_mean_metrics(ranked_all, qrels, n_corpus, threshold)


def is_tainted(model_path: str) -> bool:
    return any(fnmatch.fnmatch(model_path, pat) for pat in TAINTED_MODELS)


def derive_label(model_path: str) -> str:
    parts = Path(model_path).parts
    name = parts[-1]
    if name in {"final", "checkpoint", ""} and len(parts) > 1:
        name = parts[-2]
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Local path or HF id of a ColBERT model.")
    ap.add_argument("--label", default="", help="Name for the output JSON (default: derived from --model).")
    ap.add_argument("--gates", default="", help=f"Comma-separated subset of: {','.join(GATES)}")
    ap.add_argument("--allow-leaked", action="store_true", help="Score tainted models on leaked gates anyway.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output-dir", default=str(HERE / "output" / "evals"))
    args = ap.parse_args()

    gates = [g.strip() for g in args.gates.split(",") if g.strip()] or list(GATES)
    unknown = [g for g in gates if g not in GATES]
    if unknown:
        ap.error(f"unknown gate(s): {', '.join(unknown)}. Known: {', '.join(GATES)}")

    label = args.label or derive_label(args.model)
    tainted = is_tainted(args.model)

    from pylate import models

    print(f"Loading model: {args.model}", flush=True)
    model = models.ColBERT(model_name_or_path=args.model)

    results: dict[str, dict] = {}
    skipped: list[str] = []
    for name in gates:
        cfg = GATES[name]
        if tainted and name in TAINTED_GATES and not args.allow_leaked:
            results[name] = {
                "protocol": cfg["protocol"],
                "skipped": "leakage: model trained on eval query strings",
            }
            skipped.append(name)
            print(f"[{name}] SKIPPED — leakage: model trained on eval query strings", flush=True)
            continue

        print(f"[{name}] encoding + scoring ({cfg['protocol']})...", flush=True)
        started = time.time()
        queries, corpus, qrels, categories = load_gate(name, cfg)
        ranked_all = rank_all(model, queries, corpus, args.batch_size)
        entry = {
            "protocol": cfg["protocol"],
            "source": cfg.get("source", f"{cfg.get('dir')}/{cfg.get('split')}"),
            "headline_metric": cfg["headline"],
            "metrics": compute(cfg["protocol"], ranked_all, qrels, categories, len(corpus)),
            "seconds": round(time.time() - started, 1),
        }
        for extra in cfg.get("extra_protocols", []):
            entry.setdefault("extra_metrics", {})[extra] = compute(
                extra, ranked_all, qrels, categories, len(corpus)
            )
        if tainted and name in TAINTED_GATES:
            entry["leaked"] = True
        results[name] = entry
        print(f"[{name}] {cfg['headline']} = {entry['metrics'][cfg['headline']]}", flush=True)

    payload = {
        "label": label,
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tainted_model": tainted,
        "allow_leaked": args.allow_leaked,
        "skipped_gates": skipped,
        "gates": results,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'=' * 62}\n{label}\n{'=' * 62}")
    print(f"{'gate':<20} {'protocol':<18} {'metric':<16} {'value':>7}")
    print("-" * 62)
    for name in gates:
        entry = results[name]
        if "skipped" in entry:
            print(f"{name:<20} {entry['protocol']:<18} {'-':<16} {'SKIP':>7}")
            continue
        key = entry["headline_metric"]
        flag = " (leaked)" if entry.get("leaked") else ""
        print(f"{name:<20} {entry['protocol']:<18} {key:<16} {entry['metrics'][key]:>7.4f}{flag}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
