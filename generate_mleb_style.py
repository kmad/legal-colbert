"""
Generate MLEB-style training data: functional clause descriptions paired with actual clauses.

The MLEB Contractual Clause Retrieval benchmark uses queries like:
  "This is a contractual provision that permits a contracting party to
   deduct liabilities owed to it by the counterparty."

This script generates training pairs in that exact style by:
1. Taking real contract passages from CUAD
2. Having an LLM generate functional descriptions of what each passage does
3. Outputting (description, passage) pairs for contrastive training

Uses OpenRouter API with google/gemma-4-27b-it.

Usage:
    OPENROUTER_API_KEY=... python generate_mleb_style.py [max_passages]
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-3-27b-it"

SYSTEM_PROMPT = """You are a legal contract analyst. Given a contract clause or passage, generate functional descriptions of what the provision does.

Each description MUST follow this exact format:
"This is a contractual provision that [verb phrase describing what the clause does]."

Rules:
- Start every description with "This is a contractual provision that"
- Describe the FUNCTION of the clause, not its content
- Use legal terminology naturally (contracting party, counterparty, obligations, etc.)
- Be specific to what THIS clause does — not generic
- Do NOT quote or copy text from the clause
- Generate 2-4 descriptions per passage, each highlighting a different aspect
- Output ONLY a JSON array of strings, no other text"""

USER_TEMPLATE = """Given this contract passage, generate 2-4 functional descriptions.

Passage:
---
{passage}
---

Output a JSON array of strings, each starting with "This is a contractual provision that":"""


def generate_descriptions(passage: str, api_key: str) -> list[str]:
    """Generate MLEB-style functional descriptions for a passage."""
    if len(passage) > 3000:
        passage = passage[:3000] + "..."

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(passage=passage)},
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        },
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    # Parse JSON array from response
    json_match = re.search(r'\[[\s\S]*\]', text)
    if json_match:
        descriptions = json.loads(json_match.group())
    else:
        # Fallback: extract lines starting with "This is"
        descriptions = [line.strip().strip('"').strip(',')
                       for line in text.split('\n')
                       if line.strip().startswith('"This is') or line.strip().startswith('This is')]

    # Validate format
    valid = [d for d in descriptions
             if isinstance(d, str) and d.startswith("This is a contractual provision that") and len(d) > 50]

    return valid


def load_cuad_passages(max_passages: int = 200) -> list[dict]:
    """Load diverse contract passages from CUAD."""
    from datasets import load_dataset

    print("Loading CUAD passages...")
    ds = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)

    # Get unique passages with their questions for context
    seen = set()
    passages = []
    for ex in ds:
        key = ex["context"][:200]
        if key in seen:
            continue
        seen.add(key)

        # Only use passages with actual answers (substantive content)
        answers = ex.get("answers", {})
        if not answers or not answers.get("text") or not answers["text"][0]:
            continue

        passages.append({
            "text": ex["context"],
            "title": ex["title"],
            "question": ex["question"],
        })

        if len(passages) >= max_passages:
            break

    print(f"  {len(passages)} passages loaded")
    return passages


def generate_training_data(
    output_path: str = "mleb_style_training.json",
    max_passages: int = 200,
):
    """Generate MLEB-style training pairs."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    passages = load_cuad_passages(max_passages=max_passages)

    print(f"\nGenerating MLEB-style descriptions for {len(passages)} passages...")
    results = []
    errors = 0

    for i, passage in enumerate(passages):
        if i % 20 == 0:
            print(f"  [{i}/{len(passages)}] ({len(results)} pairs so far, {errors} errors)")

        try:
            descriptions = generate_descriptions(passage["text"], api_key)
            for desc in descriptions:
                results.append({
                    "query": desc,
                    "positive": passage["text"],
                    "contract": passage["title"],
                    "source": "mleb_style_generation",
                })
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error on passage {i}: {e}")
            continue

        time.sleep(2)  # Rate limit for OpenRouter free tier

    print(f"\nGenerated {len(results)} MLEB-style pairs from {len(passages)} passages ({errors} errors)")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")

    return results


if __name__ == "__main__":
    max_passages = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    generate_training_data(max_passages=max_passages)
