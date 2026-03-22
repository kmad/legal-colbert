"""
Prepare training data from CUAD and ACORD datasets with BM25 hard negative mining.

Outputs a triplet dataset (query, positive, negative) for ColBERT contrastive training.
"""

import re
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25, stripping punctuation."""
    return re.findall(r'\b\w+\b', text.lower())


def load_cuad_pairs() -> tuple[list[str], list[str], list[str]]:
    """Load CUAD QA dataset and extract (question, positive_context) pairs.

    Returns:
        queries: list of questions
        positives: list of positive contexts (paragraphs containing the answer)
        corpus: deduplicated list of all contexts for negative mining
    """
    print("Loading CUAD QA dataset...")
    ds = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)

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

        key = (question, hash(context))
        if key in seen:
            continue
        seen.add(key)

        # Extract a window around the answer span rather than using full context
        answer_text = answers["text"][0]
        answer_start = answers["answer_start"][0]
        window = 512  # characters
        start = max(0, answer_start - window)
        end = min(len(context), answer_start + len(answer_text) + window)
        positive_chunk = context[start:end]

        queries.append(question)
        positives.append(positive_chunk)

    # Build corpus from all unique contexts
    corpus = list({ctx for ctx in positives})
    print(f"  CUAD: {len(queries)} QA pairs, {len(corpus)} unique contexts")
    return queries, positives, corpus


def load_acord_pairs() -> tuple[list[str], list[str], list[str]]:
    """Load ACORD retrieval dataset (BEIR format) and extract (query, relevant_clause) pairs.

    Uses clauses with relevance >= 3 as positives.

    Returns:
        queries: list of queries
        positives: list of relevant clauses
        corpus: deduplicated list of all clauses for negative mining
    """
    import csv
    import json
    import tempfile
    import zipfile
    from huggingface_hub import hf_hub_download

    print("Loading ACORD dataset (BEIR format from ZIP)...")
    try:
        zip_path = hf_hub_download(
            "theatticusproject/acord", "ACORD Dataset & ReadMe.zip", repo_type="dataset"
        )
    except Exception as e:
        print(f"  ACORD: could not download ZIP ({e}), skipping")
        return [], [], []

    # Extract to temp dir
    extract_dir = tempfile.mkdtemp(prefix="acord_")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    base = f"{extract_dir}/ACORD Dataset & ReadMe (external)"
    corpus_path = f"{base}/corpus.jsonl"
    queries_path = f"{base}/queries.jsonl"
    # Combine train + test qrels for maximum data
    qrels_paths = [f"{base}/qrels/train.tsv", f"{base}/qrels/test.tsv", f"{base}/qrels/valid.tsv"]

    # Parse corpus: {doc_id: text}
    corpus_map = {}
    with open(corpus_path) as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc.get("_id", ""))
            text = doc.get("text", "")
            title = doc.get("title", "")
            full_text = f"{title}\n{text}".strip() if title else text
            corpus_map[doc_id] = full_text

    # Parse queries: {query_id: text}
    query_map = {}
    with open(queries_path) as f:
        for line in f:
            q = json.loads(line)
            q_id = str(q.get("_id", ""))
            query_map[q_id] = q.get("text", "")

    # Parse qrels from all splits: query_id, corpus_id, score
    qrels = []
    for qrels_path in qrels_paths:
        try:
            with open(qrels_path) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    q_id = str(row.get("query-id", ""))
                    d_id = str(row.get("corpus-id", ""))
                    score = int(row.get("score", 0))
                    qrels.append((q_id, d_id, score))
        except FileNotFoundError:
            continue

    queries = []
    positives = []
    all_clauses = set(corpus_map.values())

    for q_id, d_id, score in qrels:
        if score >= 3 and q_id in query_map and d_id in corpus_map:
            queries.append(query_map[q_id])
            positives.append(corpus_map[d_id])

    corpus = list(all_clauses)
    print(f"  ACORD: {len(queries)} relevant pairs, {len(corpus)} unique clauses")
    return queries, positives, corpus


def load_justia_pairs(dataset_path: str = "justia_dataset.json") -> tuple[list[str], list[str], list[str]]:
    """Load Justia clause pairs: category description as query, clause text as positive.

    Uses the clause category name as a natural language query (e.g., "Termination",
    "Limitation of Liability"). Hard negatives are clauses from DIFFERENT categories
    that BM25 thinks are similar — these are especially valuable because they teach
    the model to distinguish between clause types.

    Returns:
        queries: list of category-based queries
        positives: list of clause texts
        corpus: deduplicated list of all clauses for negative mining
    """
    import json

    if not Path(dataset_path).exists():
        print(f"  Justia dataset not found at {dataset_path}, skipping")
        return [], [], []

    print(f"Loading Justia clauses from {dataset_path}...")
    with open(dataset_path) as f:
        clauses = json.load(f)

    # Category descriptions for richer queries
    CATEGORY_QUERIES = {
        "Termination": "This clause describes when and how the agreement can be terminated by either party.",
        "Indemnification": "This clause requires one party to compensate the other for losses, damages, or liabilities.",
        "Governing Law": "This clause specifies which jurisdiction's laws govern the interpretation of the agreement.",
        "Limitation of Liability": "This clause limits the maximum amount of damages one party can recover from the other.",
        "Confidentiality": "This clause restricts the disclosure of confidential or proprietary information.",
        "Force Majeure": "This clause excuses performance when extraordinary events beyond the parties' control occur.",
        "Assignment": "This clause restricts or permits the transfer of rights or obligations under the agreement.",
        "Waiver of Jury Trial": "This clause requires both parties to waive their right to a jury trial in any dispute.",
        "Change of Control": "This clause addresses what happens when ownership or control of a party changes.",
        "Non-Competition": "This clause restricts a party from competing with the other party during or after the agreement.",
        "Severability": "This clause ensures that if one provision is found unenforceable, the rest of the agreement remains valid.",
        "Entire Agreement": "This clause states that the written agreement is the complete understanding between the parties.",
        "Dispute Resolution": "This clause specifies how disputes between the parties will be resolved, such as through arbitration or mediation.",
        "Representations and Warranties": "This clause contains statements of fact and promises about the condition of the parties and the subject matter.",
        "Events of Default": "This clause defines specific events that constitute a breach or default under the agreement.",
    }

    queries = []
    positives = []
    all_clauses = set()

    for clause in clauses:
        text = clause.get("clause_text", "")
        clause_type = clause.get("clause_type", "")
        if not text or len(text) < 50:
            continue

        all_clauses.add(text)

        # Use rich category description as query, falling back to category name
        query = CATEGORY_QUERIES.get(clause_type, clause_type)
        queries.append(query)
        positives.append(text)

    corpus = list(all_clauses)
    print(f"  Justia: {len(queries)} clause pairs, {len(corpus)} unique clauses across {len(set(c.get('clause_type') for c in clauses))} categories")
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
    if not queries:
        return []
    if len(corpus) < 2:
        raise ValueError("Need at least 2 documents in corpus to mine negatives.")

    print(f"Building BM25 index over {len(corpus)} documents...")
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    negatives = []
    for i, (query, positive) in enumerate(zip(queries, positives)):
        if i % 500 == 0:
            print(f"  Mining negatives: {i}/{len(queries)}")

        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[: min(top_k, len(scores))]

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


def prepare_training_data(output_dir: str = "data") -> DatasetDict:
    """Prepare combined training dataset from CUAD and ACORD with BM25 hard negatives.

    Returns:
        HuggingFace DatasetDict with train/test splits, columns: query, positive, negative
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Load all datasets
    cuad_queries, cuad_positives, cuad_corpus = load_cuad_pairs()
    acord_queries, acord_positives, acord_corpus = load_acord_pairs()
    justia_queries, justia_positives, justia_corpus = load_justia_pairs()

    # Mine hard negatives separately (each dataset has its own corpus)
    print("\nMining BM25 hard negatives for CUAD...")
    cuad_negatives = mine_hard_negatives_bm25(cuad_queries, cuad_positives, cuad_corpus)

    acord_negatives = []
    if acord_queries:
        print("\nMining BM25 hard negatives for ACORD...")
        acord_negatives = mine_hard_negatives_bm25(acord_queries, acord_positives, acord_corpus)

    justia_negatives = []
    if justia_queries:
        print("\nMining BM25 hard negatives for Justia...")
        justia_negatives = mine_hard_negatives_bm25(justia_queries, justia_positives, justia_corpus)

    # Combine
    all_queries = cuad_queries + acord_queries + justia_queries
    all_positives = cuad_positives + acord_positives + justia_positives
    all_negatives = cuad_negatives + acord_negatives + justia_negatives

    dataset = Dataset.from_dict({
        "query": all_queries,
        "positive": all_positives,
        "negative": all_negatives,
    })

    # Shuffle and split
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # Save
    dataset["train"].save_to_disk(f"{output_dir}/train")
    dataset["test"].save_to_disk(f"{output_dir}/test")
    print(f"\nSaved {len(dataset['train'])} train / {len(dataset['test'])} test triplets to {output_dir}/")
    return dataset


if __name__ == "__main__":
    prepare_training_data()
