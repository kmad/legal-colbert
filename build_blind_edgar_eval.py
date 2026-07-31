"""Build a heading-labeled blind clause retrieval eval from SEC contracts.

The labels are intentionally weak but independent: a span becomes relevant only
when the contract itself has a section heading matching the target clause type.
This is useful for overfitting diagnostics because the tuned retriever does not
generate labels or choose positives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


CLAUSE_TYPES = {
    "termination": {
        "query": "Find contractual provisions governing termination rights, termination for cause, termination for convenience, or the consequences of ending the agreement.",
        "patterns": [r"termination", r"term and termination", r"termination for cause", r"termination for convenience"],
    },
    "assignment": {
        "query": "Find contractual provisions restricting or permitting assignment, delegation, transfer, or succession of rights and obligations.",
        "patterns": [r"assignment", r"assignments", r"successors and assigns", r"transfer"],
    },
    "notices": {
        "query": "Find contractual provisions specifying how notices must be delivered, addressed, deemed given, or sent to the parties.",
        "patterns": [r"notices?", r"notice requirements", r"communications"],
    },
    "governing_law": {
        "query": "Find contractual provisions specifying the governing law, choice of law, jurisdiction, venue, or forum for disputes.",
        "patterns": [r"governing law", r"choice of law", r"jurisdiction", r"venue"],
    },
    "confidentiality": {
        "query": "Find contractual provisions requiring confidentiality, restricted disclosure, protection of confidential information, or permitted disclosures.",
        "patterns": [r"confidentiality", r"confidential information", r"non[- ]disclosure"],
    },
    "indemnification": {
        "query": "Find contractual provisions requiring indemnification, defense, reimbursement, or holding another party harmless from claims or losses.",
        "patterns": [r"indemnification", r"indemnity", r"hold harmless", r"defense"],
    },
    "limitation_of_liability": {
        "query": "Find contractual provisions limiting liability, excluding damages, capping damages, or disclaiming consequential damages.",
        "patterns": [r"limitation of liability", r"limitations on liability", r"liability cap", r"exclusion of damages", r"consequential damages"],
    },
    "intellectual_property": {
        "query": "Find contractual provisions governing intellectual property ownership, licenses, inventions, work product, or IP rights.",
        "patterns": [r"intellectual property", r"ownership of intellectual property", r"license grant", r"work product", r"inventions"],
    },
    "payment": {
        "query": "Find contractual provisions governing fees, payment terms, invoices, billing, taxes, expenses, or late payment.",
        "patterns": [r"payment", r"fees", r"invoices?", r"billing", r"taxes"],
    },
    "insurance": {
        "query": "Find contractual provisions requiring insurance coverage, policy limits, certificates of insurance, or additional insured status.",
        "patterns": [r"insurance", r"insurance requirements"],
    },
    "change_of_control": {
        "query": "Find contractual provisions addressing change of control, merger, acquisition, sale of substantially all assets, or ownership changes.",
        "patterns": [r"change of control", r"merger", r"sale of substantially all assets"],
    },
    "force_majeure": {
        "query": "Find contractual provisions excusing or delaying performance due to force majeure, acts of God, disasters, strikes, or events beyond control.",
        "patterns": [r"force majeure", r"acts of god", r"events beyond.*control"],
    },
    "dispute_resolution": {
        "query": "Find contractual provisions requiring arbitration, mediation, dispute resolution procedures, litigation forum, or jury trial waiver.",
        "patterns": [r"dispute resolution", r"arbitration", r"mediation", r"jury trial", r"waiver of jury"],
    },
    "survival": {
        "query": "Find contractual provisions stating which obligations survive termination or expiration of the agreement.",
        "patterns": [r"survival", r"survival of obligations"],
    },
    "representations_warranties": {
        "query": "Find contractual provisions containing representations, warranties, warranty disclaimers, authority representations, or compliance representations.",
        "patterns": [r"representations and warranties", r"representations", r"warranties", r"warranty disclaimer"],
    },
}


HEADING_RE = re.compile(
    r"(?im)^\s*(?:"
    r"(?:section|article|clause)\s+)?"
    r"(?:[ivxlcdm]+|\d+(?:\.\d+)*|[a-z])"
    r"[\).\s:-]{1,5}"
    r"(?P<title>[A-Z][A-Za-z0-9 ,/&'().\-]{2,90})"
    r"\s*$"
)


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_contracts(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def heading_matches(title: str, clause: dict) -> bool:
    normalized = normalize_space(title).lower().strip(".:;- ")
    if len(normalized.split()) > 9:
        return False
    return any(re.fullmatch(pattern, normalized) or re.search(rf"\b(?:{pattern})\b", normalized) for pattern in clause["patterns"])


def section_spans(text: str, max_chars: int) -> list[dict]:
    matches = list(HEADING_RE.finditer(text))
    spans = []
    for idx, match in enumerate(matches):
        title = normalize_space(match.group("title"))
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(text), match.end() + max_chars)
        section = text[start:end].strip()
        if len(section) > max_chars:
            section = section[:max_chars].rsplit(" ", 1)[0].strip()
        words = len(re.findall(r"\b\w+\b", section))
        if words >= 25:
            spans.append({"title": title, "text": section, "start": start, "end": min(end, start + len(section))})
    return spans


def build_eval(contracts: list[dict], max_chars: int, min_queries: int) -> tuple[dict, dict, dict, dict, dict]:
    queries = {stable_id("edgar-q", key): value["query"] for key, value in CLAUSE_TYPES.items()}
    qid_by_clause = {key: stable_id("edgar-q", key) for key in CLAUSE_TYPES}
    categories = {qid_by_clause[key]: key for key in CLAUSE_TYPES}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    evidence = []

    for contract in contracts:
        for span in section_spans(contract["text"], max_chars):
            did = stable_id("edgar-d", f"{contract['sha1']}\n{span['start']}\n{span['text']}")
            corpus[did] = span["text"]
            for clause_name, clause in CLAUSE_TYPES.items():
                if heading_matches(span["title"], clause):
                    qid = qid_by_clause[clause_name]
                    qrels[qid][did] = 1
                    evidence.append({
                        "query_id": qid,
                        "clause_type": clause_name,
                        "doc_id": did,
                        "heading": span["title"],
                        "contract_id": contract["contract_id"],
                        "ticker": contract.get("ticker"),
                        "filing_date": contract.get("filing_date"),
                        "url": contract.get("url"),
                    })

    active_qids = {qid for qid, rels in qrels.items() if len(rels) >= min_queries}
    queries = {qid: text for qid, text in queries.items() if qid in active_qids}
    categories = {qid: cat for qid, cat in categories.items() if qid in active_qids}
    qrels = {qid: dict(rows) for qid, rows in qrels.items() if qid in active_qids}
    evidence = [row for row in evidence if row["query_id"] in active_qids]
    return queries, corpus, qrels, categories, {"evidence": evidence}


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="eval_blind_edgar/raw_contracts.jsonl")
    parser.add_argument("--output-dir", default="eval_blind_edgar")
    parser.add_argument("--split", default="blind_edgar")
    parser.add_argument("--max-section-chars", type=int, default=2500)
    parser.add_argument("--min-positives-per-query", type=int, default=1)
    args = parser.parse_args()

    contracts = load_contracts(Path(args.input_jsonl))
    queries, corpus, qrels, categories, extra = build_eval(
        contracts,
        max_chars=args.max_section_chars,
        min_queries=args.min_positives_per_query,
    )

    out = Path(args.output_dir)
    write_json(out / f"{args.split}_queries.json", queries)
    write_json(out / f"{args.split}_corpus.json", corpus)
    write_json(out / f"{args.split}_qrels.json", qrels)
    write_json(out / f"{args.split}_categories.json", categories)
    write_json(out / "heading_evidence.json", extra["evidence"])
    manifest = {
        "source": "heading-labeled SEC EDGAR Exhibit 10 blind eval",
        "contracts": len(contracts),
        "queries": len(queries),
        "corpus": len(corpus),
        "qrels": sum(len(rows) for rows in qrels.values()),
        "max_section_chars": args.max_section_chars,
        "min_positives_per_query": args.min_positives_per_query,
        "labeling": "positive spans come from contract section headings only; no retriever-generated labels",
    }
    write_json(out / "eval_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
