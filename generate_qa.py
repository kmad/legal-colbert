"""
Generate CUAD-style QA pairs from full contract text using an LLM.

For each contract, identifies clause sections and generates
question-passage pairs similar to the CUAD dataset format.

Usage:
    ANTHROPIC_API_KEY=... python generate_qa.py contracts.json

Input: JSON file mapping contract URL/name → full text
Output: generated_qa.json
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic

SYSTEM_PROMPT = """You are a legal contract analyst. Given a passage from a contract, generate question-answer pairs that a lawyer or business person might ask when reviewing this contract.

Rules:
- Generate questions that someone would actually ask when reviewing a contract
- Questions should be answerable from the passage provided
- Mix question styles: "What...", "Is there...", "How much...", "When...", "Who..."
- Be specific to the actual terms in the passage (names, dates, amounts, conditions)
- Do NOT generate generic questions that could apply to any contract
- Output ONLY valid JSON"""

USER_TEMPLATE = """Given this contract passage:

---
{passage}
---

Generate 3-5 question-answer pairs. Output as JSON:
[
  {{"question": "...", "answer_span": "the exact text from the passage that answers the question"}}
]"""


def generate_qa_for_passage(client: anthropic.Anthropic, passage: str) -> list[dict] | None:
    """Generate QA pairs for a single passage."""
    if len(passage) > 3000:
        passage = passage[:3000] + "..."

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": USER_TEMPLATE.format(passage=passage)}],
        )
        text = response.content[0].text.strip()
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


def chunk_contract(text: str, target_size: int = 1500) -> list[str]:
    """Split contract text into passage-sized chunks for QA generation.

    Uses paragraph boundaries, targeting ~1500 chars per chunk.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > target_size and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    # Filter out very short chunks (headers, page numbers, etc.)
    return [c for c in chunks if len(c) > 200]


def generate_qa_from_contracts(
    contracts_path: str = "justia_contracts.json",
    output_path: str = "generated_qa.json",
    max_contracts: int | None = None,
    max_passages_per_contract: int = 20,
):
    """Generate QA pairs from contract texts."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=api_key)

    with open(contracts_path) as f:
        contracts = json.load(f)

    if isinstance(contracts, list):
        # List of contract dicts
        contract_items = [(c.get("contract", f"contract_{i}"), c.get("text", "")) for i, c in enumerate(contracts)]
    else:
        # Dict mapping URL/name → text
        contract_items = list(contracts.items())

    if max_contracts:
        contract_items = contract_items[:max_contracts]

    print(f"Generating QA from {len(contract_items)} contracts...")
    all_pairs = []

    for ci, (name, text) in enumerate(contract_items):
        if not text or len(text) < 500:
            continue

        print(f"\n[{ci+1}/{len(contract_items)}] {name[:60]}... ({len(text)} chars)")
        passages = chunk_contract(text)[:max_passages_per_contract]
        print(f"  {len(passages)} passages")

        for pi, passage in enumerate(passages):
            qa_pairs = generate_qa_for_passage(client, passage)
            if qa_pairs:
                for qa in qa_pairs:
                    if "question" in qa:
                        all_pairs.append({
                            "query": qa["question"],
                            "positive": passage,
                            "contract": name,
                            "answer_span": qa.get("answer_span", ""),
                        })
            time.sleep(0.5)  # Rate limit

        print(f"  {sum(1 for p in all_pairs if p['contract'] == name)} QA pairs generated")

    print(f"\nTotal: {len(all_pairs)} QA pairs from {len(contract_items)} contracts")

    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"Saved to {output_path}")

    return all_pairs


if __name__ == "__main__":
    contracts_path = sys.argv[1] if len(sys.argv) > 1 else "justia_contracts.json"
    max_contracts = int(sys.argv[2]) if len(sys.argv) > 2 else None
    generate_qa_from_contracts(contracts_path, max_contracts=max_contracts)
