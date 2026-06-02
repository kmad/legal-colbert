"""
Prepare V2 training/evaluation data for legal ColBERT.

Design goals:
  - keep MLEB out of training;
  - preserve ACORD graded relevance and original splits;
  - split CUAD by source contract/title where possible;
  - emit PyLate-compatible triplets by expanding multi-negative records.

Outputs:
  data_v2/train/                 HF Dataset with query, positive, negative
  data_v2/acord_valid_qrels.json graded ACORD valid qrels
  data_v2/acord_valid_corpus.json
  data_v2/acord_valid_queries.json
  data_v2/acord_test_qrels.json  graded ACORD test qrels
  data_v2/cuad_dev_qrels.json    held-out CUAD contract qrels
  data_v2/cuad_test_qrels.json   held-out CUAD contract qrels
  data_v2/manifest.json          counts and source details
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import random
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from rank_bm25 import BM25Okapi


SEED = 42
STOPWORDS = {
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "are",
    "that", "this", "by", "be", "as", "if", "any", "such", "its", "it",
    "should", "have", "has", "been", "not", "all", "with", "from", "on",
    "at", "which", "does", "do", "can", "may", "will", "would", "could",
    "there", "what", "how", "when", "where", "who", "why", "other", "each",
    "both", "than", "no", "nor", "so", "too", "very", "just", "about",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "over", "then", "once", "here",
    "only", "own", "same", "but", "because", "until", "while",
    "clause", "agreement", "party", "parties", "upon", "under", "section",
    "shall", "herein", "hereof", "thereof", "pursuant", "provided",
    "respect", "written", "notice", "provide", "whether", "including",
    "without", "limitation", "connection", "accordance", "event",
    "otherwise", "provisions", "provision", "terms", "term",
}


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def keyword_set(text: str) -> set[str]:
    return {w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2}


def keyword_overlap(query: str, text: str) -> float:
    q = keyword_set(query)
    if not q:
        return 0.0
    words = set(tokenize(text))
    return len(q & words) / len(q)


def normalize_label(label: str) -> str:
    label = re.sub(r"[_\-]+", " ", label.strip().lower())
    label = re.sub(r"\s+", " ", label)
    replacements = {
        "assignments": "assignment",
        "applicable law": "governing law",
        "choice of law": "governing law",
        "term and termination": "termination",
        "limitation of liabilities": "limitation of liability",
        "change in control": "change of control",
        "payment terms": "payment",
    }
    return replacements.get(label, label)


def sample_bm25_negatives(
    query: str,
    positives: set[str],
    corpus: list[str],
    bm25: BM25Okapi,
    n: int,
    rng: random.Random,
    max_overlap: float = 0.75,
    excluded: set[str] | None = None,
    top_k: int = 500,
) -> list[str]:
    excluded = excluded or set()
    scores = bm25.get_scores(tokenize(query))
    candidate_count = min(top_k, len(scores))
    ranked = heapq.nlargest(candidate_count, range(len(scores)), key=lambda i: scores[i])
    negatives = []
    seen = set()
    for idx in ranked:
        cand = corpus[idx]
        if cand in positives or cand in excluded or cand in seen:
            continue
        if keyword_overlap(query, cand) > max_overlap:
            continue
        negatives.append(cand)
        seen.add(cand)
        if len(negatives) >= n:
            return negatives

    pool = [c for c in corpus if c not in positives and c not in excluded and c not in seen]
    rng.shuffle(pool)
    negatives.extend(pool[: max(0, n - len(negatives))])
    return negatives


def load_acord_beir() -> dict:
    """Load ACORD BEIR files with original train/valid/test qrels."""
    zip_path = hf_hub_download(
        "theatticusproject/acord",
        "ACORD Dataset & ReadMe.zip",
        repo_type="dataset",
    )
    extract_dir = tempfile.mkdtemp(prefix="acord_")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    base = Path(extract_dir) / "ACORD Dataset & ReadMe (external)"

    corpus = {}
    with open(base / "corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text", "")
            title = row.get("title", "")
            corpus[str(row["_id"])] = f"{title}\n{text}".strip() if title else text

    queries = {}
    with open(base / "queries.jsonl") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row.get("text", "")

    qrels = {}
    for split in ("train", "valid", "test"):
        rows = []
        path = base / "qrels" / f"{split}.tsv"
        if not path.exists():
            qrels[split] = rows
            continue
        with open(path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append((
                    str(row.get("query-id", "")),
                    str(row.get("corpus-id", "")),
                    int(row.get("score", 0)),
                ))
        qrels[split] = rows

    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def acord_training_records(
    acord: dict,
    train_splits: Iterable[str],
    negatives_per_positive: int,
    min_positive_grade: int = 4,
) -> list[dict]:
    corpus = acord["corpus"]
    queries = acord["queries"]
    qrels = acord["qrels"]
    records = []

    for split in train_splits:
        by_query = defaultdict(list)
        for qid, did, score in qrels.get(split, []):
            if qid in queries and did in corpus:
                by_query[qid].append((did, score))

        for qid, rows in by_query.items():
            # Use grade >= min_positive_grade as positives (lower threshold keeps
            # more of ACORD's expert graded signal); grade <= 2 are hard negatives.
            positives = [(did, s) for did, s in rows if s >= min_positive_grade]
            soft_positives = [(did, s) for did, s in rows if s == 3]
            hard_negs = [(did, s) for did, s in rows if s <= 2]
            if not positives:
                positives = soft_positives
            pos_ids = {did for did, _ in positives + soft_positives}
            neg_ids = [did for did, _ in sorted(hard_negs, key=lambda x: x[1], reverse=True)]

            for did, score in positives:
                selected = [corpus[nid] for nid in neg_ids if nid not in pos_ids][:negatives_per_positive]
                if selected:
                    records.append({
                        "source": f"acord:{split}",
                        "query": queries[qid],
                        "positive": corpus[did],
                        "negatives": selected,
                        "positive_grade": score,
                    })
    return records


def acord_dev_files(acord: dict, split: str = "valid") -> tuple[dict, dict, dict]:
    qrels_rows = acord["qrels"].get(split) or acord["qrels"].get("test") or []
    qids = {qid for qid, _, _ in qrels_rows}
    dids = {did for _, did, _ in qrels_rows}
    queries = {qid: acord["queries"][qid] for qid in qids if qid in acord["queries"]}
    corpus = {did: acord["corpus"][did] for did in dids if did in acord["corpus"]}
    qrels = defaultdict(dict)
    for qid, did, score in qrels_rows:
        if qid in queries and did in corpus:
            qrels[qid][did] = score
    return dict(queries), corpus, {k: dict(v) for k, v in qrels.items()}


def build_cuad_eval_files(examples: list[dict]) -> tuple[dict, dict, dict]:
    queries = {}
    corpus = {}
    qrels = defaultdict(dict)
    for ex in examples:
        qid = stable_id("cuad-q", ex["query"])
        did = stable_id("cuad-d", ex["positive"])
        queries[qid] = ex["query"]
        corpus[did] = ex["positive"]
        qrels[qid][did] = 1
    return queries, corpus, {qid: dict(rows) for qid, rows in qrels.items()}


def clean_maud_question(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[-: ]*Answer$", "", text)
    text = re.sub(r"\(Y/N\)", "", text)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def maud_query(row: dict) -> str:
    question = clean_maud_question(str(row.get("question") or ""))
    subquestion = str(row.get("subquestion") or "").strip()
    text_type = clean_maud_question(str(row.get("text_type") or ""))
    category = clean_maud_question(str(row.get("category") or ""))

    parts = [question]
    if subquestion and subquestion != "<NONE>":
        parts.append(f"focus: {clean_maud_question(subquestion)}")
    if text_type and text_type.lower() not in question.lower():
        parts.append(f"clause type: {text_type}")
    if category and category.lower() not in " ".join(parts).lower():
        parts.append(f"category: {category}")
    return ". ".join(part for part in parts if part)


def load_maud_records(
    negatives_per_positive: int,
    rng: random.Random,
    max_records: int,
) -> tuple[list[dict], dict[str, tuple[dict, dict, dict]]]:
    ds = load_dataset("theatticusproject/maud")

    train_examples = []
    for row in ds["train"]:
        query = maud_query(row)
        positive = str(row.get("text") or "").strip()
        if not query or len(positive.split()) < 8:
            continue
        train_examples.append({
            "query": query,
            "positive": positive,
            "contract_id": str(row.get("contract_name") or ""),
            "data_type": str(row.get("data_type") or ""),
            "question": str(row.get("question") or ""),
            "subquestion": str(row.get("subquestion") or ""),
        })

    if max_records > 0 and len(train_examples) > max_records:
        by_query = defaultdict(list)
        for ex in train_examples:
            by_query[ex["query"]].append(ex)
        sampled = []
        query_order = list(by_query)
        rng.shuffle(query_order)
        while len(sampled) < max_records and query_order:
            next_round = []
            for query in query_order:
                rows = by_query[query]
                if not rows:
                    continue
                sampled.append(rows.pop(rng.randrange(len(rows))))
                if rows:
                    next_round.append(query)
                if len(sampled) >= max_records:
                    break
            query_order = next_round
        train_examples = sampled

    corpus = list({ex["positive"] for ex in train_examples})
    bm25 = BM25Okapi([tokenize(doc) for doc in corpus]) if corpus else None
    query_to_pos = defaultdict(set)
    for ex in train_examples:
        query_to_pos[ex["query"]].add(ex["positive"])

    query_to_negatives = {}
    for i, query in enumerate(query_to_pos):
        if bm25 is None:
            continue
        if i % 25 == 0:
            print(f"  MAUD BM25 negatives: {i}/{len(query_to_pos)} unique queries", flush=True)
        query_to_negatives[query] = sample_bm25_negatives(
            query,
            query_to_pos[query],
            corpus,
            bm25,
            negatives_per_positive,
            rng,
        )

    records = []
    for ex in train_examples:
        negatives = query_to_negatives.get(ex["query"], [])
        if negatives:
            records.append({
                "source": "maud",
                "query": ex["query"],
                "positive": ex["positive"],
                "negatives": negatives,
                "positive_grade": 1,
            })

    eval_files = {
        "validation": build_maud_eval_files(ds["validation"]),
        "test": build_maud_eval_files(ds["test"]),
    }
    return records, eval_files


def build_maud_eval_files(rows: Iterable[dict]) -> tuple[dict, dict, dict]:
    queries = {}
    corpus = {}
    qrels = defaultdict(dict)
    for row in rows:
        query = maud_query(row)
        positive = str(row.get("text") or "").strip()
        if not query or len(positive.split()) < 8:
            continue
        qid = stable_id("maud-q", query)
        did = stable_id("maud-d", positive)
        queries[qid] = query
        corpus[did] = positive
        qrels[qid][did] = 1
    return queries, corpus, {qid: dict(doc_scores) for qid, doc_scores in qrels.items()}


def aligned_span(context: str, answer_start: int, answer_text: str, window: int) -> str:
    """Extract a window around the answer, snapped to whitespace boundaries.

    Unlike the raw ``answer_start +/- 512`` slice, this never truncates mid-word
    and uses a tighter window so the positive is centered on the labeled clause
    rather than padded with unrelated surrounding context (CUAD label noise).
    """
    start = max(0, answer_start - window)
    end = min(len(context), answer_start + len(answer_text) + window)
    if start > 0:
        boundary = max(context.rfind(" ", 0, start), context.rfind("\n", 0, start))
        if boundary != -1:
            start = boundary + 1
    if end < len(context):
        cands = [c for c in (context.find(" ", end), context.find("\n", end)) if c != -1]
        if cands:
            end = min(cands)
    return context[start:end].strip()


def load_cuad_records(
    dev_fraction: float,
    test_fraction: float,
    negatives_per_positive: int,
    rng: random.Random,
    train_window: int = 0,
) -> tuple[list[dict], dict[str, tuple[dict, dict, dict]]]:
    ds = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)
    examples = []
    contract_to_examples = defaultdict(list)
    for i, ex in enumerate(ds):
        answers = ex.get("answers") or {}
        texts = answers.get("text") or []
        starts = answers.get("answer_start") or []
        if not texts or not starts:
            continue
        context = ex.get("context", "")
        if not context:
            continue
        answer_text = texts[0]
        answer_start = int(starts[0])
        start = max(0, answer_start - 512)
        end = min(len(context), answer_start + len(answer_text) + 512)
        positive = context[start:end].strip()
        if len(positive.split()) < 8:
            continue
        # Training positive: optionally a tighter, boundary-aligned span. Eval
        # files always use ``positive`` (locked +/-512) so splits stay comparable.
        train_positive = positive
        if train_window > 0:
            candidate = aligned_span(context, answer_start, answer_text, train_window)
            if len(candidate.split()) >= 8:
                train_positive = candidate
        contract_id = (
            ex.get("title")
            or ex.get("contract_name")
            or str(ex.get("id", "")).split("_")[0]
            or stable_id("cuad-contract", context[:2000])
        )
        row = {
            "query": ex["question"],
            "positive": positive,
            "train_positive": train_positive,
            "contract_id": contract_id,
        }
        examples.append(row)
        contract_to_examples[contract_id].append(row)

    contracts = list(contract_to_examples)
    rng.shuffle(contracts)
    n_dev = max(1, int(len(contracts) * dev_fraction))
    n_test = max(1, int(len(contracts) * test_fraction))
    dev_contracts = set(contracts[:n_dev])
    test_contracts = set(contracts[n_dev : n_dev + n_test])
    heldout_contracts = dev_contracts | test_contracts
    train_examples = [ex for ex in examples if ex["contract_id"] not in heldout_contracts]
    dev_examples = [ex for ex in examples if ex["contract_id"] in dev_contracts]
    test_examples = [ex for ex in examples if ex["contract_id"] in test_contracts]

    corpus = list({ex["train_positive"] for ex in train_examples})
    bm25 = BM25Okapi([tokenize(doc) for doc in corpus]) if corpus else None
    query_to_pos = defaultdict(set)
    for ex in train_examples:
        query_to_pos[ex["query"]].add(ex["train_positive"])

    query_to_negatives = {}
    for i, query in enumerate(query_to_pos):
        if bm25 is None:
            continue
        if i % 25 == 0:
            print(f"  CUAD BM25 negatives: {i}/{len(query_to_pos)} unique queries", flush=True)
        query_to_negatives[query] = sample_bm25_negatives(
            query,
            query_to_pos[query],
            corpus,
            bm25,
            negatives_per_positive,
            rng,
        )

    records = []
    for ex in train_examples:
        negatives = query_to_negatives.get(ex["query"], [])
        if negatives:
            records.append({
                "source": "cuad",
                "query": ex["query"],
                "positive": ex["train_positive"],
                "negatives": negatives,
                "positive_grade": 1,
            })
    eval_files = {
        "dev": build_cuad_eval_files(dev_examples),
        "test": build_cuad_eval_files(test_examples),
    }
    return records, eval_files


def load_labeled_clause_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    rows = []
    for item in data:
        text = (
            item.get("clause_text")
            or item.get("text")
            or item.get("clause")
            or ""
        ).strip()
        label = (
            item.get("clause_type")
            or item.get("category")
            or item.get("type")
            or ""
        ).strip()
        if len(text.split()) >= 12 and label:
            rows.append({"label": normalize_label(label), "text": text})
    return rows


def extra_json_records(
    path: Path,
    corpus: list[str],
    negatives_per_positive: int,
    repeat: int,
    rng: random.Random,
) -> list[dict]:
    if repeat < 1 or not path.exists():
        return []
    rows = load_labeled_clause_json(path)
    if not rows:
        with open(path) as f:
            data = json.load(f)
        rows = [
            {"query": item.get("query", ""), "text": item.get("positive", "")}
            for item in data
            if item.get("query") and item.get("positive")
        ]
    if not rows or not corpus:
        return []

    bm25 = BM25Okapi([tokenize(doc) for doc in corpus])
    records = []
    for row in rows:
        query = row.get("query") or row.get("label") or ""
        positive = row.get("positive") or row.get("text") or ""
        if not query or not positive:
            continue
        negatives = sample_bm25_negatives(
            query,
            {positive},
            corpus,
            bm25,
            negatives_per_positive,
            rng,
        )
        if negatives:
            for _ in range(repeat):
                records.append({
                    "source": f"extra_json:{path.name}",
                    "query": query,
                    "positive": positive,
                    "negatives": negatives,
                    "positive_grade": 1,
                })
    return records


def cuad_category(query: str) -> str:
    """Extract the CUAD clause-category label from a templated query."""
    match = re.search(r'related to "([^"]+)"', query)
    return match.group(1) if match else query.strip()[:60]


def balance_records_by_category(
    records: list[dict],
    ceiling: int,
    floor: int,
    rng: random.Random,
) -> tuple[list[dict], dict[str, int]]:
    """Cap frequent clause categories and oversample rare ones.

    Compresses the per-category record count into roughly [floor, ceiling] so a
    single fine-tuning epoch does not let frequent categories (e.g. Parties)
    dominate gradients while rare categories (e.g. Price Restrictions) starve.
    Oversampling repeats whole records; downsampling samples without replacement.
    """
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_cat[cuad_category(rec["query"])].append(rec)

    balanced: list[dict] = []
    counts: dict[str, int] = {}
    for cat in sorted(by_cat):
        rows = list(by_cat[cat])
        rng.shuffle(rows)
        if ceiling > 0 and len(rows) > ceiling:
            rows = rows[:ceiling]
        elif floor > 0 and len(rows) < floor:
            repeated: list[dict] = []
            while len(repeated) < floor:
                repeated.extend(rows)
            rows = repeated[:floor]
        counts[cat] = len(rows)
        balanced.extend(rows)

    rng.shuffle(balanced)
    return balanced, counts


def expand_triplets(records: list[dict], rng: random.Random) -> Dataset:
    rows = []
    for rec in records:
        for neg in rec["negatives"]:
            if neg and neg != rec["positive"]:
                rows.append({
                    "query": rec["query"],
                    "positive": rec["positive"],
                    "negative": neg,
                    "source": rec["source"],
                    "positive_grade": rec.get("positive_grade", 1),
                })
    rng.shuffle(rows)
    return Dataset.from_list(rows)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data_v2")
    parser.add_argument("--negatives-per-positive", type=int, default=6)
    parser.add_argument("--cuad-dev-fraction", type=float, default=0.10)
    parser.add_argument("--cuad-test-fraction", type=float, default=0.10)
    parser.add_argument(
        "--acord-repeat",
        type=int,
        default=1,
        help="Repeat ACORD train records before triplet expansion; 0 excludes ACORD from training.",
    )
    parser.add_argument(
        "--acord-min-grade",
        type=int,
        default=4,
        help="Minimum ACORD relevance grade treated as a positive (3 keeps more graded signal).",
    )
    parser.add_argument(
        "--cuad-max-records",
        type=int,
        default=0,
        help="Cap CUAD train records after negative mining; 0 keeps all records.",
    )
    parser.add_argument(
        "--cuad-balance-ceiling",
        type=int,
        default=0,
        help="Max CUAD train records per clause category; 0 disables capping.",
    )
    parser.add_argument(
        "--cuad-balance-floor",
        type=int,
        default=0,
        help="Min CUAD train records per clause category via oversampling; 0 disables.",
    )
    parser.add_argument(
        "--cuad-paraphrase-json",
        default="",
        help="JSON {category: [paraphrases]} to diversify CUAD training queries "
        "(replaces the rigid template with a sampled paraphrase per record). "
        "Eval splits keep their own queries, so this stays leakage-free.",
    )
    parser.add_argument(
        "--cuad-train-window",
        type=int,
        default=0,
        help="Char window each side of the CUAD answer for TRAINING positives, "
        "snapped to word boundaries; 0 keeps the locked +/-512 span. Eval splits "
        "always use the +/-512 span for comparability.",
    )
    parser.add_argument(
        "--extra-json",
        default="",
        help="Optional list of {'query','positive'} records to add as clause-extraction training examples.",
    )
    parser.add_argument("--extra-repeat", type=int, default=1)
    parser.add_argument("--include-maud", action="store_true")
    parser.add_argument(
        "--maud-repeat",
        type=int,
        default=1,
        help="Repeat MAUD train records before triplet expansion.",
    )
    parser.add_argument(
        "--maud-max-records",
        type=int,
        default=0,
        help="Cap MAUD train records after query-balanced sampling; 0 keeps all records.",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading ACORD...")
    acord = load_acord_beir()
    acord_records = []
    if args.acord_repeat > 0:
        acord_records = acord_training_records(
            acord,
            train_splits=("train",),
            negatives_per_positive=args.negatives_per_positive,
            min_positive_grade=args.acord_min_grade,
        )
    if args.acord_repeat < 0:
        raise ValueError("--acord-repeat must be >= 0")
    records = acord_records * args.acord_repeat
    print(f"  ACORD training records: {len(acord_records)}")
    if args.acord_repeat > 1:
        print(f"  ACORD repeated records: {len(records)}")

    print("Loading CUAD...")
    cuad, cuad_eval = load_cuad_records(
        args.cuad_dev_fraction,
        args.cuad_test_fraction,
        args.negatives_per_positive,
        rng,
        train_window=args.cuad_train_window,
    )
    if args.cuad_max_records > 0 and len(cuad) > args.cuad_max_records:
        rng.shuffle(cuad)
        cuad = cuad[: args.cuad_max_records]
    cuad_balance_counts = {}
    if args.cuad_balance_ceiling > 0 or args.cuad_balance_floor > 0:
        cuad, cuad_balance_counts = balance_records_by_category(
            cuad,
            args.cuad_balance_ceiling,
            args.cuad_balance_floor,
            rng,
        )
        vals = sorted(cuad_balance_counts.values())
        print(
            f"  CUAD category balance: {len(cuad_balance_counts)} categories, "
            f"per-category records min={vals[0]} max={vals[-1]} "
            f"(ceiling={args.cuad_balance_ceiling}, floor={args.cuad_balance_floor})"
        )
    cuad_paraphrase_used = 0
    if args.cuad_paraphrase_json:
        with open(args.cuad_paraphrase_json) as f:
            paraphrases = json.load(f)
        for rec in cuad:
            options = paraphrases.get(cuad_category(rec["query"]))
            if options:
                rec["query"] = rng.choice(options)
                cuad_paraphrase_used += 1
        print(
            f"  CUAD query paraphrasing: {cuad_paraphrase_used}/{len(cuad)} records "
            f"rephrased from {args.cuad_paraphrase_json}"
        )
    records.extend(cuad)
    print(f"  CUAD training records: {len(cuad)}")

    maud_count = 0
    maud_eval = {}
    if args.include_maud:
        if args.maud_repeat < 1:
            raise ValueError("--maud-repeat must be >= 1 when --include-maud is set")
        print("Loading MAUD...")
        maud, maud_eval = load_maud_records(
            args.negatives_per_positive,
            rng,
            args.maud_max_records,
        )
        records.extend(maud * args.maud_repeat)
        maud_count = len(maud) * args.maud_repeat
        print(f"  MAUD training records: {len(maud)}")
        if args.maud_repeat > 1:
            print(f"  MAUD repeated records: {maud_count}")

    extra_count = 0
    if args.extra_json:
        print("Loading extra JSON training records...")
        extra_corpus = list({rec["positive"] for rec in records})
        extra = extra_json_records(
            Path(args.extra_json),
            extra_corpus,
            args.negatives_per_positive,
            args.extra_repeat,
            rng,
        )
        records.extend(extra)
        extra_count = len(extra)
        print(f"  Extra JSON training records: {extra_count}")

    dataset = expand_triplets(records, rng)
    dataset.save_to_disk(str(out / "train"))

    eval_manifest = {}
    for split in ("valid", "test"):
        queries, corpus, qrels = acord_dev_files(acord, split)
        prefix = f"acord_{split}"
        write_json(out / f"{prefix}_queries.json", queries)
        write_json(out / f"{prefix}_corpus.json", corpus)
        write_json(out / f"{prefix}_qrels.json", qrels)
        eval_manifest[prefix] = {
            "source": f"ACORD {split} split",
            "queries": len(queries),
            "corpus": len(corpus),
            "qrels": sum(len(v) for v in qrels.values()),
        }

    # Backward-compatible aliases for earlier commands.
    valid_queries, valid_corpus, valid_qrels = acord_dev_files(acord, "valid")
    write_json(out / "dev_queries.json", valid_queries)
    write_json(out / "dev_corpus.json", valid_corpus)
    write_json(out / "dev_qrels.json", valid_qrels)

    for split, files in cuad_eval.items():
        queries, corpus, qrels = files
        prefix = f"cuad_{split}"
        write_json(out / f"{prefix}_queries.json", queries)
        write_json(out / f"{prefix}_corpus.json", corpus)
        write_json(out / f"{prefix}_qrels.json", qrels)
        eval_manifest[prefix] = {
            "source": f"CUAD held-out {split} contracts",
            "queries": len(queries),
            "corpus": len(corpus),
            "qrels": sum(len(v) for v in qrels.values()),
        }

    for split, files in maud_eval.items():
        queries, corpus, qrels = files
        prefix = f"maud_{split}"
        write_json(out / f"{prefix}_queries.json", queries)
        write_json(out / f"{prefix}_corpus.json", corpus)
        write_json(out / f"{prefix}_qrels.json", qrels)
        eval_manifest[prefix] = {
            "source": f"MAUD {split} split",
            "queries": len(queries),
            "corpus": len(corpus),
            "qrels": sum(len(v) for v in qrels.values()),
        }

    manifest = {
        "seed": SEED,
        "triplets": len(dataset),
        "multi_negative_records": len(records),
        "negatives_per_positive": args.negatives_per_positive,
        "acord_repeat": args.acord_repeat,
        "cuad_max_records": args.cuad_max_records,
        "cuad_balance_ceiling": args.cuad_balance_ceiling,
        "cuad_balance_floor": args.cuad_balance_floor,
        "cuad_train_window": args.cuad_train_window,
        "cuad_paraphrase_json": args.cuad_paraphrase_json,
        "cuad_paraphrase_used": cuad_paraphrase_used,
        "cuad_balance_counts": cuad_balance_counts,
        "include_maud": args.include_maud,
        "maud_repeat": args.maud_repeat,
        "maud_max_records": args.maud_max_records,
        "extra_json": args.extra_json,
        "extra_repeat": args.extra_repeat,
        "sources": {
            "acord_records": len([r for r in records if r["source"].startswith("acord")]),
            "cuad_records": len(cuad),
            "maud_records": maud_count,
            "extra_json_records": extra_count,
        },
        "eval_splits": eval_manifest,
        "dev": eval_manifest["acord_valid"],
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
