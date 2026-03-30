"""
Extract specific clause types from a legal contract.

Usage:
    # Extract all common clause types from a PDF
    python extract_clauses.py example_contract.pdf

    # Extract specific clause types
    python extract_clauses.py example_contract.pdf termination "governing law" indemnification

    # From raw text
    python extract_clauses.py --text contract.txt assignment
"""

import sys
from pathlib import Path

from pylate import models

from chunk import Chunk, chunk_document, chunk_pdf, estimate_tokens
from retrieve import colbert_score

# Clause type → expanded query variants for better retrieval
CLAUSE_QUERIES = {
    "termination": [
        "termination of this agreement for cause or convenience",
        "right to terminate the agreement upon notice or breach",
        "circumstances under which either party may end this contract",
    ],
    "assignment": [
        "assignment or transfer of rights and obligations under this agreement",
        "restrictions on assigning this contract to a third party",
        "consent required for assignment of this agreement",
    ],
    "governing law": [
        "governing law and choice of law provision",
        "which state or jurisdiction law governs this agreement",
        "this agreement shall be governed by the laws of",
    ],
    "indemnification": [
        "indemnification and hold harmless obligations",
        "party shall indemnify defend and hold harmless against claims losses damages",
        "indemnity for third party claims and liabilities",
    ],
    "limitation of liability": [
        "limitation of liability and cap on damages",
        "maximum aggregate liability under this agreement",
        "exclusion of consequential incidental or punitive damages",
    ],
    "confidentiality": [
        "confidentiality and non-disclosure obligations",
        "protection of confidential and proprietary information",
        "restrictions on disclosure of confidential information",
    ],
    "events of default": [
        "events of default under this agreement",
        "circumstances constituting a default or breach",
        "failure to pay or perform constitutes an event of default",
    ],
    "representations and warranties": [
        "representations and warranties of the parties",
        "each party represents and warrants that it is duly organized",
        "representations regarding authority power and enforceability",
    ],
    "force majeure": [
        "force majeure and excused performance",
        "events beyond reasonable control including acts of god war pandemic",
        "neither party liable for failure to perform due to force majeure",
    ],
    "severability": [
        "severability of provisions",
        "if any provision is held invalid the remainder shall continue in effect",
        "invalid or unenforceable provision shall be modified to minimum extent",
    ],
    "entire agreement": [
        "entire agreement and integration clause",
        "this agreement constitutes the entire understanding between the parties",
        "supersedes all prior agreements negotiations and understandings",
    ],
    "waiver of jury trial": [
        "waiver of right to jury trial",
        "each party irrevocably waives any right to trial by jury",
        "disputes shall be resolved without a jury",
    ],
    "non-competition": [
        "non-competition and non-compete restrictions",
        "shall not compete with the company during and after employment",
        "restrictive covenant prohibiting competitive activities",
    ],
    "dispute resolution": [
        "dispute resolution and arbitration",
        "disputes shall be resolved through binding arbitration or mediation",
        "arbitration under the rules of the American Arbitration Association",
    ],
    "change of control": [
        "change of control provision",
        "transfer of ownership or controlling interest triggers rights",
        "merger acquisition or change in majority ownership",
    ],
    "notices": [
        "notices and communications under this agreement",
        "all notices shall be in writing and delivered to the addresses specified",
        "notice is effective upon receipt by registered mail or overnight courier",
    ],
    "payment terms": [
        "payment terms and schedule",
        "amounts due and payable under this agreement",
        "invoicing payment due dates and late payment interest",
    ],
    "insurance": [
        "insurance requirements and coverage",
        "party shall maintain insurance policies with minimum coverage",
        "proof of insurance and certificate of insurance",
    ],
    "intellectual property": [
        "intellectual property rights and ownership",
        "ownership of inventions patents copyrights and trade secrets",
        "license grant for intellectual property created under this agreement",
    ],
    "subordination": [
        "subordination of debt and priority of payments",
        "junior debt subordinate to senior obligations",
        "subordinated lender rights subject to prior payment of senior debt",
    ],
}


def extract_clauses(
    chunks: list[Chunk],
    model: models.ColBERT,
    clause_types: list[str] | None = None,
    top_k: int = 3,
    context_window: int = 1,
    max_overlap: float = 0.5,
) -> dict[str, list[tuple[str, float, int]]]:
    """Extract specific clause types from pre-chunked document.

    For each clause type, runs multiple query variants and returns
    diverse, high-scoring results with surrounding context. Handles
    clauses that appear in multiple sections of the contract.

    Args:
        chunks: document chunks to search
        model: ColBERT model
        clause_types: list of clause types to extract (default: all)
        top_k: number of distinct mentions to return per clause type
        context_window: number of chunks before/after to include as context
        max_overlap: maximum Jaccard word overlap between results (0-1).
            Results with higher overlap than this are considered duplicates
            and skipped. 0.5 means two chunks sharing >50% of their words
            are treated as the same content.

    Returns:
        dict mapping clause type → list of (text_with_context, score, chunk_index) tuples
    """
    if clause_types is None:
        clause_types = list(CLAUSE_QUERIES.keys())

    # Encode all chunks once
    doc_embs = model.encode(
        [c.full_text for c in chunks],
        batch_size=32, is_query=False, show_progress_bar=False,
    )

    results = {}
    for clause_type in clause_types:
        # Normalize clause type name
        ct = clause_type.lower().strip()
        query_variants = CLAUSE_QUERIES.get(ct)
        if not query_variants:
            # Fallback: use the clause type name directly with expansions
            query_variants = [
                f"{ct} clause provision",
                f"{ct} rights and obligations under this agreement",
                f"provisions relating to {ct}",
            ]

        # Score all chunks against all query variants, take max per chunk
        chunk_best_scores = {}
        for query in query_variants:
            q_emb = model.encode([query], batch_size=1, is_query=True, show_progress_bar=False)[0]
            for i, d_emb in enumerate(doc_embs):
                score = colbert_score(q_emb, d_emb)
                if i not in chunk_best_scores or score > chunk_best_scores[i]:
                    chunk_best_scores[i] = score

        # Rank all chunks by score
        ranked = sorted(chunk_best_scores.items(), key=lambda x: x[1], reverse=True)

        # Select top-k DIVERSE results using content deduplication
        # Skip results whose text overlaps heavily with already-selected results
        selected = []
        selected_texts = []
        for idx, score in ranked:
            if len(selected) >= top_k:
                break

            candidate_text = chunks[idx].text
            # Check word overlap with already-selected results
            if selected_texts:
                candidate_words = set(candidate_text.lower().split())
                is_duplicate = False
                for prev_text in selected_texts:
                    prev_words = set(prev_text.lower().split())
                    if not candidate_words or not prev_words:
                        continue
                    intersection = len(candidate_words & prev_words)
                    union = len(candidate_words | prev_words)
                    jaccard = intersection / union if union > 0 else 0
                    if jaccard > max_overlap:
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue

            selected.append((idx, score))
            selected_texts.append(candidate_text)

        # Build context: include +/- context_window chunks around each hit
        clause_results = []
        for idx, score in selected:
            start = max(0, idx - context_window)
            end = min(len(chunks), idx + context_window + 1)
            context_texts = [chunks[i].text for i in range(start, end)]
            full_text = "\n\n".join(context_texts)
            clause_results.append((full_text, score, idx))

        results[clause_type] = clause_results

    return results


def print_extraction(results: dict[str, list[tuple[str, float, int]]]):
    """Pretty-print extracted clauses."""
    for clause_type, matches in results.items():
        print(f"\n{'='*80}")
        print(f"  {clause_type.upper()}")
        print(f"{'='*80}")

        if not matches:
            print("  No matches found.")
            continue

        for i, (text, score, chunk_idx) in enumerate(matches):
            tokens = estimate_tokens(text)
            print(f"\n  [{i+1}] Score: {score:.1f}  (~{tokens} tokens, chunk #{chunk_idx})")
            # Show full text for top result, preview for others
            if i == 0:
                display = text[:800].replace("\n", "\n      ")
                print(f"      {display}")
                if len(text) > 800:
                    print(f"      ...")
            else:
                display = text[:300].replace("\n", "\n      ")
                print(f"      {display}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract clauses from a contract")
    parser.add_argument("document", help="PDF or text file path")
    parser.add_argument("clauses", nargs="*", help="Clause types to extract (default: all common types)")
    parser.add_argument("--text", action="store_true", help="Input is a text file, not PDF")
    parser.add_argument("--top-k", type=int, default=3, help="Results per clause type")
    args = parser.parse_args()

    # Load document
    if args.text:
        with open(args.document) as f:
            text = f.read()
        chunks = chunk_document(text)
    else:
        chunks = chunk_pdf(args.document)

    chunks = [c for c in chunks if len(c.text.split()) >= 5]
    print(f"Document: {args.document} ({len(chunks)} chunks)")

    # Load model
    local_model = Path("model")
    if local_model.exists() and (local_model / "model.safetensors").exists():
        model = models.ColBERT(model_name_or_path=str(local_model))
    else:
        model = models.ColBERT(model_name_or_path="kmad00/legal-colbert-v1")

    # Determine clause types
    if args.clauses:
        clause_types = args.clauses
    else:
        # Default: extract the most common clause types
        clause_types = [
            "termination", "governing law", "indemnification",
            "assignment", "confidentiality", "limitation of liability",
            "events of default", "representations and warranties",
            "severability", "entire agreement",
        ]

    # Extract and print
    results = extract_clauses(chunks, model, clause_types, top_k=args.top_k)
    print_extraction(results)
