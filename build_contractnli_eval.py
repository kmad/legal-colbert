"""Build an external ContractNLI blind retrieval diagnostic.

LegalBench exposes ContractNLI as separate binary tasks with evidence text and a
Yes/No answer. This converts those tasks into a BEIR-style retrieval set: one
query per NLI hypothesis, Yes rows as relevant evidence spans, and No rows as
hard distractors. It is intended as an external diagnostic, not a training set.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

from prepare_v2_data import SEED, stable_id


QUERY_TEXT = {
    "contract_nli_confidentiality_of_agreement": "Find provisions showing that the existence or terms of the agreement must be kept confidential.",
    "contract_nli_explicit_identification": "Find provisions requiring confidential information to be marked, designated, or otherwise explicitly identified as confidential.",
    "contract_nli_inclusion_of_verbally_conveyed_information": "Find provisions stating that orally or verbally disclosed information can be confidential information.",
    "contract_nli_limited_use": "Find provisions limiting use of confidential information to a permitted purpose.",
    "contract_nli_no_licensing": "Find provisions stating that disclosure of confidential information does not grant a license or intellectual property rights.",
    "contract_nli_notice_on_compelled_disclosure": "Find provisions requiring notice before compelled disclosure of confidential information.",
    "contract_nli_permissible_acquirement_of_similar_information": "Find provisions allowing use of similar information acquired lawfully from another source or already known.",
    "contract_nli_permissible_copy": "Find provisions permitting or restricting copies of confidential information.",
    "contract_nli_permissible_development_of_similar_information": "Find provisions allowing independent development of similar information without using confidential information.",
    "contract_nli_permissible_post-agreement_possession": "Find provisions allowing possession or retention of confidential information after the agreement ends.",
    "contract_nli_return_of_confidential_information": "Find provisions requiring confidential information to be returned or destroyed.",
    "contract_nli_sharing_with_employees": "Find provisions permitting confidential information to be shared with employees, affiliates, or representatives.",
    "contract_nli_sharing_with_third-parties": "Find provisions permitting confidential information to be shared with third parties.",
    "contract_nli_survival_of_obligations": "Find provisions stating that confidentiality or other obligations survive termination or expiration.",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def task_label(config: str) -> str:
    return config.removeprefix("contract_nli_").replace("_", " ").replace("-", " ").title()


def sample_rows(rows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:cap] if cap > 0 else rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="eval_contractnli_blind")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-positive-per-task", type=int, default=14)
    parser.add_argument("--max-distractor-per-task", type=int, default=14)
    parser.add_argument("--min-words", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(SEED)
    configs = [
        name
        for name in get_dataset_config_names("nguha/legalbench")
        if name.startswith("contract_nli_")
    ]
    missing = sorted(set(configs) - set(QUERY_TEXT))
    if missing:
        raise ValueError(f"Missing query text for configs: {missing}")

    queries = {}
    corpus = {}
    qrels = defaultdict(dict)
    categories = {}
    manifest_tasks = {}

    for config in sorted(configs):
        ds = load_dataset("nguha/legalbench", config, split=args.split)
        positives = []
        distractors = []
        seen = set()
        for row in ds:
            text = normalize_text(str(row.get("text") or ""))
            if len(text.split()) < args.min_words or text in seen:
                continue
            seen.add(text)
            answer = str(row.get("answer") or "").strip().lower()
            item = {
                "text": text,
                "document_name": str(row.get("document_name") or ""),
                "answer": answer,
            }
            if answer == "yes":
                positives.append(item)
            else:
                distractors.append(item)

        positives = sample_rows(positives, args.max_positive_per_task, rng)
        distractors = sample_rows(distractors, args.max_distractor_per_task, rng)

        query = QUERY_TEXT[config]
        qid = stable_id("contractnli-q", config)
        queries[qid] = query
        categories[qid] = task_label(config)

        for item in positives + distractors:
            did = stable_id("contractnli-d", f"{config}\n{item['text']}")
            corpus[did] = item["text"]
            if item["answer"] == "yes":
                qrels[qid][did] = 1

        manifest_tasks[config] = {
            "query": query,
            "positive_rows": len(positives),
            "distractor_rows": len(distractors),
        }

    out = Path(args.output_dir)
    prefix = "contractnli_test"
    write_json(out / f"{prefix}_queries.json", queries)
    write_json(out / f"{prefix}_corpus.json", corpus)
    write_json(out / f"{prefix}_qrels.json", {qid: dict(rows) for qid, rows in qrels.items()})
    write_json(out / f"{prefix}_categories.json", categories)

    manifest = {
        "seed": SEED,
        "source": "nguha/legalbench ContractNLI tasks",
        "split": args.split,
        "queries": len(queries),
        "corpus": len(corpus),
        "qrels": sum(len(rows) for rows in qrels.values()),
        "max_positive_per_task": args.max_positive_per_task,
        "max_distractor_per_task": args.max_distractor_per_task,
        "min_words": args.min_words,
        "tasks": manifest_tasks,
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
