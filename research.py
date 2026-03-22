"""
Autoresearch loop for improving legal-colbert.

Each iteration:
  1. Agent proposes a change (data, hyperparams, training strategy)
  2. Trains the model on a GPU instance
  3. Benchmarks on MLEB Contractual Clause Retrieval
  4. Keeps the change if NDCG@10 improved, reverts if not
  5. Logs results to results.tsv

Usage:
    # Run a single experiment (called by the agent)
    python research.py run --experiment "description of change"

    # Run benchmark only (on current model)
    python research.py benchmark

    # Show results log
    python research.py log
"""

import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset

RESULTS_FILE = "results.tsv"
DATASET_ID = "isaacus/contractual-clause-retrieval"
DATASET_REV = "48ed7bcb1f50896a0f71461a04b2df0ca84329d9"


def load_benchmark():
    """Load MLEB contractual clause retrieval dataset."""
    corpus_ds = load_dataset(DATASET_ID, "corpus", revision=DATASET_REV, split="corpus")
    queries_ds = load_dataset(DATASET_ID, "queries", revision=DATASET_REV, split="queries")
    qrels_ds = load_dataset(DATASET_ID, "default", revision=DATASET_REV, split="test")

    corpus = {row["_id"]: row["text"] for row in corpus_ds}
    queries = {row["_id"]: row["text"] for row in queries_ds}
    qrels = {}
    for row in qrels_ds:
        qid = row["query-id"]
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][row["corpus-id"]] = row["score"]

    return corpus, queries, qrels


def benchmark_model(model_path: str = "model") -> dict:
    """Benchmark a ColBERT model on MLEB Contractual Clause Retrieval.

    Returns dict with NDCG@10, MAP, Recall@1/5/10.
    """
    import torch
    from pylate import models

    corpus, queries, qrels = load_benchmark()

    model = models.ColBERT(model_name_or_path=model_path)

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    c_embs = model.encode(corpus_texts, batch_size=32, is_query=False, show_progress_bar=False)
    q_embs = model.encode(query_texts, batch_size=32, is_query=True, show_progress_bar=False)

    def colbert_score(q_emb, d_emb):
        q = torch.tensor(q_emb) if not isinstance(q_emb, torch.Tensor) else q_emb
        d = torch.tensor(d_emb) if not isinstance(d_emb, torch.Tensor) else d_emb
        return torch.matmul(q, d.T).max(dim=1).values.sum().item()

    def dcg_at_k(rels, k=10):
        rels = np.array(rels[:k])
        return np.sum(rels / np.log2(np.arange(2, rels.size + 2))) if rels.size else 0.0

    def ndcg_at_k(rels, k=10):
        dcg = dcg_at_k(rels, k)
        ideal = dcg_at_k(sorted(rels, reverse=True), k)
        return dcg / ideal if ideal > 0 else 0.0

    ndcg_scores = []
    ap_scores = []
    recall_at = {1: [], 5: [], 10: []}

    for qi, qid in enumerate(query_ids):
        scores = [(ci, colbert_score(q_embs[qi], c_embs[ci])) for ci in range(len(corpus_ids))]
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked = [corpus_ids[ci] for ci, _ in scores]
        relevances = [qrels.get(qid, {}).get(cid, 0) for cid in ranked]
        relevant = set(did for did, s in qrels.get(qid, {}).items() if s > 0)

        ndcg_scores.append(ndcg_at_k(relevances, 10))

        if relevant:
            hits = 0
            sum_prec = 0
            for rank, cid in enumerate(ranked, 1):
                if cid in relevant:
                    hits += 1
                    sum_prec += hits / rank
            ap_scores.append(sum_prec / len(relevant))
            for k in [1, 5, 10]:
                recall_at[k].append(len(relevant & set(ranked[:k])) / len(relevant))

    return {
        "ndcg@10": round(float(np.mean(ndcg_scores)), 4),
        "map": round(float(np.mean(ap_scores)), 4),
        "recall@1": round(float(np.mean(recall_at[1])), 4),
        "recall@5": round(float(np.mean(recall_at[5])), 4),
        "recall@10": round(float(np.mean(recall_at[10])), 4),
    }


def get_best_ndcg() -> float:
    """Read best NDCG@10 from results log."""
    if not Path(RESULTS_FILE).exists():
        return 0.0
    best = 0.0
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["status"] == "kept":
                best = max(best, float(row["ndcg@10"]))
    return best


def log_result(
    commit: str,
    ndcg: float,
    metrics: dict,
    status: str,
    description: str,
    train_time: float = 0.0,
):
    """Append a result row to results.tsv."""
    file_exists = Path(RESULTS_FILE).exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        fieldnames = ["timestamp", "commit", "ndcg@10", "map", "recall@1", "recall@10",
                       "train_time_s", "status", "description"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "commit": commit[:8],
            "ndcg@10": metrics["ndcg@10"],
            "map": metrics["map"],
            "recall@1": metrics["recall@1"],
            "recall@10": metrics["recall@10"],
            "train_time_s": round(train_time, 1),
            "status": status,
            "description": description,
        })


def get_commit_hash() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()


def run_experiment(description: str):
    """Run a single experiment iteration.

    Assumes the agent has already made changes to training code/data/config.
    This function:
      1. Runs prepare_data.py (if data/ doesn't exist)
      2. Runs train.py
      3. Benchmarks the result
      4. Keeps or reverts based on NDCG@10
    """
    best_ndcg = get_best_ndcg()
    print(f"Current best NDCG@10: {best_ndcg}")
    print(f"Experiment: {description}")

    # Prepare data if needed
    if not Path("data/train").exists():
        print("\nPreparing training data...")
        result = subprocess.run(
            [sys.executable, "prepare_data.py"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"Data prep failed:\n{result.stderr[-500:]}")
            log_result(get_commit_hash(), 0, {"ndcg@10": 0, "map": 0, "recall@1": 0, "recall@10": 0},
                      "crash", f"[data prep failed] {description}")
            return

    # Train
    print("\nTraining...")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "train.py", "--batch-size", "16", "--num-epochs", "3",
         "--bf16", "--no-fp16"],
        capture_output=True, text=True, timeout=600,
        env={**__import__("os").environ, "WANDB_MODE": "disabled"},
    )
    train_time = time.time() - t0

    if result.returncode != 0:
        print(f"Training failed:\n{result.stderr[-500:]}")
        log_result(get_commit_hash(), 0, {"ndcg@10": 0, "map": 0, "recall@1": 0, "recall@10": 0},
                  "crash", f"[train failed] {description}", train_time)
        return

    print(f"Training complete in {train_time:.0f}s")

    # Copy trained model to model/
    print("\nCopying model...")
    import shutil
    model_src = Path("output/legal-colbert/final")
    model_dst = Path("model")
    if model_src.exists():
        if model_dst.exists():
            shutil.rmtree(model_dst)
        shutil.copytree(model_src, model_dst)

    # Benchmark
    print("\nBenchmarking...")
    metrics = benchmark_model("model")
    ndcg = metrics["ndcg@10"]
    print(f"NDCG@10: {ndcg} (best: {best_ndcg})")

    # Decision
    commit = get_commit_hash()
    if ndcg > best_ndcg:
        print(f"\n>>> IMPROVED: {best_ndcg} -> {ndcg} (+{ndcg - best_ndcg:.4f})")
        log_result(commit, ndcg, metrics, "kept", description, train_time)
    else:
        print(f"\n>>> NO IMPROVEMENT: {ndcg} <= {best_ndcg}")
        log_result(commit, ndcg, metrics, "discarded", description, train_time)


def show_log():
    """Display the results log."""
    if not Path(RESULTS_FILE).exists():
        print("No results yet.")
        return
    with open(RESULTS_FILE) as f:
        print(f.read())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python research.py benchmark          # Benchmark current model")
        print("  python research.py run --experiment '...'  # Run experiment")
        print("  python research.py log                # Show results")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "benchmark":
        model_path = sys.argv[2] if len(sys.argv) > 2 else "model"
        metrics = benchmark_model(model_path)
        print(f"\n{'='*40}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    elif cmd == "run":
        desc = " ".join(sys.argv[2:]).replace("--experiment", "").strip()
        if not desc:
            desc = input("Experiment description: ").strip()
        run_experiment(desc)

    elif cmd == "log":
        show_log()

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
