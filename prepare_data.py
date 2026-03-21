"""
Prepare training data from CUAD and ACORD datasets with BM25 hard negative mining.

Outputs a triplet dataset (query, positive, negative) for ColBERT contrastive training.
"""

import json
import random
from pathlib import Path

from datasets import Dataset, load_dataset
from rank_bm25 import BM25Okapi


def load_cuad_pairs() -> tuple[list[str], list[str], list[str]]:
    """Load CUAD QA dataset and extract (question, positive_context) pairs.

    Returns:
        queries: list of questions
        positives: list of positive contexts (paragraphs containing the answer)
        corpus: deduplicated list of all contexts for negative mining
    """
    print("Loading CUAD QA dataset...")
    ds = load_dataset("theatticusproject/cuad-qa", split="train")

    queries = []
    positives = []
    seen = set()

    for example in ds:
        question = example["question"]
        context = example["context"]
        answers = example["answers"]

        # Skip examples with no answer
        if not answers or not answers.get("text") or len(answers["text"]) == 0:
            continue

        key = (question, context[:200])
        if key in seen:
            continue
        seen.add(key)

        queries.append(question)
        positives.append(context)

    # Build corpus from all unique contexts
    corpus = list({ctx for ctx in positives})
    print(f"  CUAD: {len(queries)} QA pairs, {len(corpus)} unique contexts")
    return queries, positives, corpus


def load_acord_pairs() -> tuple[list[str], list[str], list[str]]:
    """Load ACORD retrieval dataset and extract (query, relevant_clause) pairs.

    Uses clauses with relevance >= 3 as positives.

    Returns:
        queries: list of queries
        positives: list of relevant clauses
        corpus: deduplicated list of all clauses for negative mining
    """
    print("Loading ACORD dataset...")
    ds = load_dataset("theatticusproject/acord", split="train")

    queries = []
    positives = []
    all_clauses = set()

    for example in ds:
        query = example.get("query") or example.get("question", "")
        clause = example.get("clause") or example.get("passage") or example.get("text", "")
        relevance = example.get("relevance") or example.get("score") or example.get("label", 0)

        if not query or not clause:
            continue

        all_clauses.add(clause)

        # Use highly relevant clauses as positives
        if relevance >= 3:
            queries.append(query)
            positives.append(clause)

    corpus = list(all_clauses)
    print(f"  ACORD: {len(queries)} relevant pairs, {len(corpus)} unique clauses")
    return queries, positives, corpus


def mine_hard_negatives_bm25(
    queries: list[str],
    positives: list[str],
    corpus: list[str],
    n_negatives: int = 1,
    top_k: int = 20,
) -> list[str]:
    """Mine hard negatives using BM25.

    For each query, retrieves top-k BM25 results and picks the highest-ranked
    document that is NOT the positive as the hard negative.

    Args:
        queries: list of query strings
        positives: list of positive document strings (aligned with queries)
        corpus: full corpus to mine negatives from
        n_negatives: number of hard negatives per query
        top_k: number of BM25 candidates to consider

    Returns:
        negatives: list of hard negative documents (aligned with queries)
    """
    print(f"Building BM25 index over {len(corpus)} documents...")
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    negatives = []
    for i, (query, positive) in enumerate(zip(queries, positives)):
        if i % 500 == 0:
            print(f"  Mining negatives: {i}/{len(queries)}")

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]

        # Pick the highest-scoring document that isn't the positive
        hard_neg = None
        for idx in top_indices:
            candidate = corpus[idx]
            if candidate != positive:
                hard_neg = candidate
                break

        # Fallback: random negative
        if hard_neg is None:
            hard_neg = random.choice([c for c in corpus if c != positive])

        negatives.append(hard_neg)

    print(f"  Mined {len(negatives)} hard negatives")
    return negatives


def prepare_training_data(output_dir: str = "data") -> Dataset:
    """Prepare combined training dataset from CUAD and ACORD with BM25 hard negatives.

    Returns:
        HuggingFace Dataset with columns: query, positive, negative
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Load both datasets
    cuad_queries, cuad_positives, cuad_corpus = load_cuad_pairs()
    acord_queries, acord_positives, acord_corpus = load_acord_pairs()

    # Mine hard negatives separately (each dataset has its own corpus)
    print("\nMining BM25 hard negatives for CUAD...")
    cuad_negatives = mine_hard_negatives_bm25(cuad_queries, cuad_positives, cuad_corpus)

    print("\nMining BM25 hard negatives for ACORD...")
    acord_negatives = mine_hard_negatives_bm25(acord_queries, acord_positives, acord_corpus)

    # Combine
    all_queries = cuad_queries + acord_queries
    all_positives = cuad_positives + acord_positives
    all_negatives = cuad_negatives + acord_negatives

    dataset = Dataset.from_dict({
        "query": all_queries,
        "positive": all_positives,
        "negative": all_negatives,
    })

    # Shuffle
    dataset = dataset.shuffle(seed=42)

    # Save
    dataset.save_to_disk(f"{output_dir}/legal_triplets")
    print(f"\nSaved {len(dataset)} triplets to {output_dir}/legal_triplets")
    return dataset


if __name__ == "__main__":
    prepare_training_data()
