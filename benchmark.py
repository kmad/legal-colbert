"""
Benchmark legal-colbert-v1 on MLEB Contractual Clause Retrieval.

Compares our fine-tuned ColBERT against:
  - The base model (GTE-ModernColBERT-v1, no fine-tuning)
  - General-purpose bi-encoders (BGE-large, MiniLM)

Usage:
    python benchmark.py
"""

import numpy as np
from datasets import load_dataset

DATASET_ID = "isaacus/contractual-clause-retrieval"
DATASET_REV = "48ed7bcb1f50896a0f71461a04b2df0ca84329d9"


def load_benchmark():
    """Load the MLEB contractual clause retrieval dataset."""
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


def dcg_at_k(rels, k=10):
    rels = np.array(rels[:k])
    return np.sum(rels / np.log2(np.arange(2, rels.size + 2))) if rels.size else 0.0


def ndcg_at_k(rels, k=10):
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg / ideal if ideal > 0 else 0.0


def eval_colbert(model_name, corpus, queries, qrels, label):
    """Evaluate a ColBERT model using MaxSim scoring."""
    import torch
    from pylate import models

    print(f"\nEvaluating: {label}")
    model = models.ColBERT(model_name_or_path=model_name)

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

    return _compute_metrics(query_ids, corpus_ids, q_embs, c_embs, qrels, colbert_score, label)


def eval_biencoder(model_name, corpus, queries, qrels, label):
    """Evaluate a bi-encoder model using cosine similarity."""
    from sentence_transformers import SentenceTransformer

    print(f"\nEvaluating: {label}")
    model = SentenceTransformer(model_name)

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    c_embs = model.encode(corpus_texts, show_progress_bar=False, normalize_embeddings=True)
    q_embs = model.encode(query_texts, show_progress_bar=False, normalize_embeddings=True)

    def cosine_score(q_emb, d_emb):
        return float(np.dot(q_emb, d_emb))

    return _compute_metrics(query_ids, corpus_ids, q_embs, c_embs, qrels, cosine_score, label)


def _compute_metrics(query_ids, corpus_ids, q_embs, c_embs, qrels, score_fn, label):
    """Compute NDCG@10, MAP, Recall@1/5/10."""
    ndcg_scores = []
    ap_scores = []
    recall_at = {1: [], 5: [], 10: []}

    for qi, qid in enumerate(query_ids):
        scores = [(ci, score_fn(q_embs[qi], c_embs[ci])) for ci in range(len(corpus_ids))]
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

    results = {
        "NDCG@10": np.mean(ndcg_scores),
        "MAP": np.mean(ap_scores),
        "Recall@1": np.mean(recall_at[1]),
        "Recall@5": np.mean(recall_at[5]),
        "Recall@10": np.mean(recall_at[10]),
    }

    print(f"  NDCG@10:  {results['NDCG@10']:.4f}")
    print(f"  MAP:      {results['MAP']:.4f}")
    print(f"  Recall@1: {results['Recall@1']:.4f}")
    print(f"  Recall@5: {results['Recall@5']:.4f}")
    print(f"  Recall@10:{results['Recall@10']:.4f}")

    return label, results


if __name__ == "__main__":
    corpus, queries, qrels = load_benchmark()
    print(f"Loaded: {len(queries)} queries, {len(corpus)} passages, {len(qrels)} qrels")

    all_results = []

    # Our fine-tuned model
    all_results.append(eval_colbert("model", corpus, queries, qrels, "legal-colbert-v1 (ours)"))

    # Base model
    all_results.append(eval_colbert(
        "lightonai/GTE-ModernColBERT-v1", corpus, queries, qrels,
        "GTE-ModernColBERT-v1 (base)",
    ))

    # General-purpose bi-encoders
    all_results.append(eval_biencoder(
        "BAAI/bge-large-en-v1.5", corpus, queries, qrels,
        "BGE-large-en-v1.5",
    ))
    all_results.append(eval_biencoder(
        "all-MiniLM-L6-v2", corpus, queries, qrels,
        "all-MiniLM-L6-v2",
    ))

    # Summary table
    print(f"\n{'='*70}")
    print(f"MLEB Contractual Clause Retrieval — Summary")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'NDCG@10':>8} {'MAP':>8} {'R@1':>8} {'R@10':>8}")
    print(f"{'-'*70}")
    for label, r in all_results:
        print(f"{label:<35} {r['NDCG@10']:>8.4f} {r['MAP']:>8.4f} {r['Recall@1']:>8.4f} {r['Recall@10']:>8.4f}")
