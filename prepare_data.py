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


def load_generated_qa(dataset_path: str = "generated_qa.json") -> tuple[list[str], list[str], list[str]]:
    """Load LLM-generated QA pairs from full contracts.

    Returns:
        queries, positives, corpus
    """
    import json

    if not Path(dataset_path).exists():
        print(f"  Generated QA not found at {dataset_path}, skipping")
        return [], [], []

    print(f"Loading generated QA from {dataset_path}...")
    with open(dataset_path) as f:
        pairs = json.load(f)

    queries = [p["query"] for p in pairs if p.get("query")]
    positives = [p["positive"] for p in pairs if p.get("positive")]
    corpus = list({p["positive"] for p in pairs if p.get("positive")})
    print(f"  Generated QA: {len(queries)} pairs, {len(corpus)} unique passages")
    return queries, positives, corpus


def load_synthetic_pairs(dataset_path: str = "synthetic_queries.json") -> tuple[list[str], list[str], list[str]]:
    """Load LLM-generated synthetic query-clause pairs.

    These have three query styles per clause (natural question, keyword, semantic)
    matching real production usage patterns.

    Returns:
        queries, positives, corpus
    """
    import json

    if not Path(dataset_path).exists():
        print(f"  Synthetic queries not found at {dataset_path}, skipping")
        return [], [], []

    print(f"Loading synthetic queries from {dataset_path}...")
    with open(dataset_path) as f:
        pairs = json.load(f)

    queries = [p["query"] for p in pairs]
    positives = [p["positive"] for p in pairs]
    corpus = list({p["positive"] for p in pairs})
    print(f"  Synthetic: {len(queries)} pairs, {len(corpus)} unique clauses")
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


def _query_keywords(query: str) -> set[str]:
    """Extract meaningful keywords from a query, stripping stopwords and CUAD boilerplate."""
    stopwords = {
        # English stopwords
        'the', 'a', 'an', 'of', 'in', 'to', 'for', 'and', 'or', 'is', 'are',
        'that', 'this', 'by', 'be', 'as', 'if', 'any', 'such', 'its', 'it',
        'should', 'have', 'has', 'been', 'not', 'all', 'with', 'from', 'on',
        'at', 'which', 'does', 'do', 'can', 'may', 'will', 'would', 'could',
        'there', 'what', 'how', 'when', 'where', 'who', 'why', 'other', 'each',
        'both', 'than', 'no', 'nor', 'so', 'too', 'very', 'just', 'about',
        'between', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'up', 'down', 'out', 'off', 'over', 'then', 'once', 'here',
        'only', 'own', 'same', 'but', 'because', 'until', 'while',
        # CUAD query boilerplate
        'highlight', 'parts', 'contract', 'related', 'reviewed', 'lawyer',
        'details', 'one', 'right',
        # Generic legal terms (appear everywhere in contracts)
        'clause', 'agreement', 'party', 'parties', 'upon', 'under',
        'section', 'shall', 'herein', 'hereof', 'thereof', 'pursuant',
        'provided', 'respect', 'written', 'notice', 'provide', 'whether',
        'including', 'without', 'limitation', 'connection', 'accordance',
        'event', 'otherwise', 'provisions', 'provision', 'terms', 'term',
        'set', 'forth', 'made', 'date', 'prior', 'following', 'subject',
    }
    words = set(re.findall(r'\b\w+\b', query.lower()))
    return words - stopwords


def _keyword_overlap(query_kw: set[str], text: str) -> float:
    """Fraction of query keywords found in text."""
    if not query_kw:
        return 0.0
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    return len(query_kw & text_words) / len(query_kw)


def mine_hard_negatives_bm25(
    queries: list[str],
    positives: list[str],
    corpus: list[str],
    n_negatives: int = 1,
    top_k: int = 100,
    max_keyword_overlap: float = 0.6,
) -> list[str]:
    """Mine hard negatives using BM25, filtering out false negatives.

    For each query, retrieves top-k BM25 results and picks the highest-ranked
    document that:
    1. Is not a known positive for that query (across all contracts)
    2. Has keyword overlap with the query below max_keyword_overlap, to avoid
       selecting passages that are clearly about the same topic

    Args:
        queries: list of query strings
        positives: list of positive document strings (aligned with queries)
        corpus: full corpus to mine negatives from
        n_negatives: number of hard negatives per query
        top_k: number of BM25 candidates to consider
        max_keyword_overlap: reject candidates with query keyword overlap above this

    Returns:
        negatives: list of hard negative documents (aligned with queries)
    """
    if not queries:
        return []
    if len(corpus) < 2:
        raise ValueError("Need at least 2 documents in corpus to mine negatives.")

    from collections import defaultdict

    # Build a set of ALL positives for each unique query, so we exclude
    # documents that are valid answers from other contracts
    query_to_positives = defaultdict(set)
    for q, p in zip(queries, positives):
        query_to_positives[q].add(p)

    print(f"Building BM25 index over {len(corpus)} documents...")
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    negatives = []
    overlap_filtered = 0
    fallback_count = 0
    for i, (query, positive) in enumerate(zip(queries, positives)):
        if i % 500 == 0:
            print(f"  Mining negatives: {i}/{len(queries)}")

        all_positives_for_query = query_to_positives[query]
        query_kw = _query_keywords(query)

        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[: min(top_k, len(scores))]

        # Extract quoted clause type from CUAD-style queries for exact-match filtering
        ct_match = re.search(r'"([^"]+)"', query)
        clause_type_pattern = (
            re.compile(re.escape(ct_match.group(1)), re.I) if ct_match else None
        )

        # Pick the highest-scoring document that isn't a positive and doesn't
        # have excessive keyword overlap (which signals it's about the same topic)
        hard_neg = None
        for idx in top_indices:
            candidate = corpus[idx]
            if candidate in all_positives_for_query:
                continue
            # Reject if candidate contains the exact clause type phrase
            if clause_type_pattern and clause_type_pattern.search(candidate):
                overlap_filtered += 1
                continue
            if _keyword_overlap(query_kw, candidate) > max_keyword_overlap:
                overlap_filtered += 1
                continue
            hard_neg = candidate
            break

        # Fallback: random negative from corpus, preferring ones that don't
        # match the clause type or have low keyword overlap
        if hard_neg is None:
            fallback_count += 1
            non_positives = [c for c in corpus if c not in all_positives_for_query]
            if non_positives:
                # Filter out candidates containing the exact clause type
                filtered = non_positives
                if clause_type_pattern:
                    filtered = [c for c in filtered if not clause_type_pattern.search(c)]
                # Further prefer low keyword overlap
                low_overlap = [c for c in filtered
                               if _keyword_overlap(query_kw, c) <= max_keyword_overlap]
                pool = low_overlap or filtered or non_positives
                hard_neg = random.choice(pool)
            else:
                hard_neg = random.choice([c for c in corpus if c != positive])

        negatives.append(hard_neg)

    print(f"  Mined {len(negatives)} hard negatives")
    print(f"  Filtered {overlap_filtered} candidates for keyword overlap > {max_keyword_overlap}")
    if fallback_count:
        print(f"  {fallback_count} queries fell back to random negatives")
    return negatives


def mine_hard_negatives_model(
    queries: list[str],
    positives: list[str],
    corpus: list[str],
    model_path: str = "model",
    batch_size: int = 64,
) -> list[str]:
    """Mine hard negatives using the trained ColBERT model.

    For each query, scores ALL corpus passages with the model and picks
    the highest-scoring passage that isn't a known positive. These are
    the passages the model currently confuses — exactly what it needs
    to learn from in the next round of training.

    This is a second-pass refinement: run BM25 mining first to train an
    initial model, then use that model to find harder negatives.
    """
    import torch
    from collections import defaultdict
    from pylate import models

    if not queries:
        return []

    import torch as _torch
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    print(f"Loading ColBERT model from {model_path} for negative mining (device={device})...")
    model = models.ColBERT(model_name_or_path=model_path, device=device)

    # Build positive lookup
    query_to_positives = defaultdict(set)
    for q, p in zip(queries, positives):
        query_to_positives[q].add(p)

    # Encode corpus once
    print(f"Encoding {len(corpus)} corpus passages...")
    corpus_embs = model.encode(corpus, batch_size=batch_size, is_query=False, show_progress_bar=True)

    # Encode queries
    unique_queries = list(set(queries))
    print(f"Encoding {len(unique_queries)} unique queries...")
    query_embs_map = {}
    for i in range(0, len(unique_queries), batch_size):
        batch = unique_queries[i:i + batch_size]
        embs = model.encode(batch, batch_size=batch_size, is_query=True, show_progress_bar=False)
        for q, emb in zip(batch, embs):
            query_embs_map[q] = emb

    # Build positive index set for fast lookup: for each query, which corpus indices are positive
    corpus_lookup = {text: idx for idx, text in enumerate(corpus)}
    query_pos_indices = defaultdict(set)
    for q, p in zip(queries, positives):
        if p in corpus_lookup:
            query_pos_indices[q].add(corpus_lookup[p])

    # Batch score: for each unique query, compute MaxSim against all corpus docs
    # We score in chunks to fit in GPU memory
    print("Batch scoring queries against corpus...")
    query_top_negs = {}  # query → list of (corpus_idx, score) sorted desc, non-positives only
    scored = 0

    for q in unique_queries:
        q_emb = torch.tensor(query_embs_map[q])
        pos_indices = query_pos_indices[q]

        # Score against all corpus — do in chunks of 500 for memory
        all_scores = []
        for start in range(0, len(corpus), 500):
            end = min(start + 500, len(corpus))
            batch_scores = []
            for ci in range(start, end):
                d = torch.tensor(corpus_embs[ci])
                s = torch.matmul(q_emb, d.T).max(dim=1).values.sum().item()
                batch_scores.append((ci, s))
            all_scores.extend(batch_scores)

        # Sort descending, find best non-positive
        all_scores.sort(key=lambda x: x[1], reverse=True)
        best_neg_idx = None
        for ci, score in all_scores:
            if ci not in pos_indices:
                best_neg_idx = ci
                break
        query_top_negs[q] = best_neg_idx

        scored += 1
        if scored % 100 == 0:
            print(f"  Scored {scored}/{len(unique_queries)} unique queries")

    # Map back to per-example negatives
    print("Assembling negatives...")
    negatives = []
    for query, positive in zip(queries, positives):
        neg_idx = query_top_negs.get(query)
        if neg_idx is not None:
            negatives.append(corpus[neg_idx])
        else:
            negatives.append(random.choice([c for c in corpus if c != positive]))

    print(f"  Mined {len(negatives)} model-based hard negatives")
    return negatives


def prepare_training_data(output_dir: str = "data", use_model_negatives: bool = False, model_path: str = "model") -> DatasetDict:
    """Prepare combined training dataset with hard negatives.

    Args:
        output_dir: where to save the dataset
        use_model_negatives: if True, use the trained ColBERT model for negative mining
            instead of BM25. Requires a trained model at model_path.
        model_path: path to trained model for model-based negative mining

    Returns:
        HuggingFace DatasetDict with train/test splits, columns: query, positive, negative
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Load all datasets
    cuad_queries, cuad_positives, cuad_corpus = load_cuad_pairs()
    acord_queries, acord_positives, acord_corpus = load_acord_pairs()
    synth_queries, synth_positives, synth_corpus = load_synthetic_pairs()
    genqa_queries, genqa_positives, genqa_corpus = load_generated_qa()

    if use_model_negatives:
        # Model-based hard negatives: find passages the model currently confuses
        print(f"\nMining model-based hard negatives (model: {model_path})...")

        # Combine all corpora for cross-dataset mining
        all_corpus = list(set(cuad_corpus + acord_corpus + synth_corpus + genqa_corpus))
        all_queries_combined = cuad_queries + acord_queries + synth_queries + genqa_queries
        all_positives_combined = cuad_positives + acord_positives + synth_positives + genqa_positives

        all_negatives = mine_hard_negatives_model(
            all_queries_combined, all_positives_combined, all_corpus,
            model_path=model_path,
        )
    else:
        # BM25 hard negatives (first pass, no trained model needed)
        print("\nMining BM25 hard negatives for CUAD...")
        cuad_negatives = mine_hard_negatives_bm25(cuad_queries, cuad_positives, cuad_corpus)

        acord_negatives = []
        if acord_queries:
            print("\nMining BM25 hard negatives for ACORD...")
            acord_negatives = mine_hard_negatives_bm25(acord_queries, acord_positives, acord_corpus)

        synth_negatives = []
        if synth_queries:
            print("\nMining BM25 hard negatives for synthetic queries...")
            synth_negatives = mine_hard_negatives_bm25(synth_queries, synth_positives, synth_corpus)

        genqa_negatives = []
        if genqa_queries:
            print("\nMining BM25 hard negatives for generated QA...")
            genqa_negatives = mine_hard_negatives_bm25(genqa_queries, genqa_positives, genqa_corpus)

        all_negatives = cuad_negatives + acord_negatives + synth_negatives + genqa_negatives

    # Combine queries and positives
    all_queries_combined = cuad_queries + acord_queries + synth_queries + genqa_queries
    all_positives_combined = cuad_positives + acord_positives + synth_positives + genqa_positives

    dataset = Dataset.from_dict({
        "query": all_queries_combined,
        "positive": all_positives_combined,
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-negatives", action="store_true",
                        help="Use trained ColBERT model for hard negative mining (requires model/)")
    parser.add_argument("--model-path", default="model",
                        help="Path to trained model for model-based mining")
    args = parser.parse_args()
    prepare_training_data(use_model_negatives=args.model_negatives, model_path=args.model_path)
