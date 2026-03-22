"""
Generate synthetic queries for contract clauses using Claude.

For each clause, generates three query styles:
  1. Natural question (how a user would ask)
  2. Short keyword phrase (how a user would search)
  3. Semantic description (what the clause does functionally)

Usage:
    ANTHROPIC_API_KEY=... python generate_queries.py

Input: justia_dataset.json
Output: synthetic_queries.json
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic


SYSTEM_PROMPT = """You are a legal contract analyst. Given a contract clause, generate three different queries that someone might use to find this clause in a contract search system.

Rules:
- Do NOT copy phrases verbatim from the clause
- Each query should be findable by meaning, not keyword matching
- Be specific to what THIS clause says, not generic to the clause type
- Keep queries concise (1-2 sentences max)

Output ONLY valid JSON with no other text."""

USER_TEMPLATE = """Given this contract clause:

---
{clause_text}
---

Generate exactly 3 queries in this JSON format:
{{
  "natural_question": "A question a business person or lawyer would ask, like 'What happens if we terminate early?' or 'How much notice do we need to give?'",
  "keyword_search": "A short search phrase, 2-6 words, like 'early termination penalties' or 'notice period cancellation'",
  "semantic_description": "A one-sentence description of what this provision does functionally, like 'This provision specifies the financial consequences of terminating the agreement before its natural expiration date.'"
}}"""


def generate_queries_for_clause(client: anthropic.Anthropic, clause_text: str) -> dict | None:
    """Generate three query styles for a single clause."""
    # Truncate very long clauses
    if len(clause_text) > 2000:
        clause_text = clause_text[:2000] + "..."

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": USER_TEMPLATE.format(clause_text=clause_text),
            }],
        )
        text = response.content[0].text.strip()

        # Parse JSON from response
        # Handle potential markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        if text.endswith("```"):
            text = text[:-3]

        return json.loads(text.strip())
    except Exception as e:
        print(f"  Error: {e}")
        return None


def generate_all(
    input_path: str = "justia_dataset.json",
    output_path: str = "synthetic_queries.json",
    max_clauses: int | None = None,
):
    """Generate synthetic queries for all clauses."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=api_key)

    with open(input_path) as f:
        clauses = json.load(f)

    if max_clauses:
        clauses = clauses[:max_clauses]

    print(f"Generating queries for {len(clauses)} clauses...")

    results = []
    for i, clause in enumerate(clauses):
        if i % 10 == 0:
            print(f"  [{i}/{len(clauses)}]")

        clause_text = clause["clause_text"]
        queries = generate_queries_for_clause(client, clause_text)

        if queries:
            for style in ["natural_question", "keyword_search", "semantic_description"]:
                if style in queries and queries[style]:
                    results.append({
                        "query": queries[style],
                        "query_style": style,
                        "positive": clause_text,
                        "clause_type": clause.get("clause_type", ""),
                        "contract": clause.get("contract", ""),
                    })

    print(f"\nGenerated {len(results)} query-clause pairs")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")

    # Summary
    from collections import Counter
    style_counts = Counter(r["query_style"] for r in results)
    type_counts = Counter(r["clause_type"] for r in results)
    print(f"\nBy style: {dict(style_counts)}")
    print(f"By type: {dict(type_counts)}")

    return results


if __name__ == "__main__":
    max_clauses = int(sys.argv[1]) if len(sys.argv) > 1 else None
    generate_all(max_clauses=max_clauses)
