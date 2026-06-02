"""
Generate augmented training data from contract texts using text-albumentations.

Runs QA pair generation and retrieval augmentation on contract passages
to produce diverse training triplets for ColBERT fine-tuning.

Usage:
    ANTHROPIC_API_KEY=... python augment_data.py

Input: Contract texts from CUAD (via HuggingFace) or local files
Output: augmented_training.json
"""

import json
import os
import sys
from pathlib import Path

import anthropic
from pydantic import BaseModel

from text_albumentations import run_augmentation, AugmentationRunner
from text_albumentations.tasks.qa_pairs import QaPairAugmentation
from text_albumentations.tasks.retrieval import RetrievalAugmentation
from text_albumentations.runtime import ModelRuntime


class AnthropicRuntime(ModelRuntime):
    """Anthropic Claude adapter for text-albumentations ModelRuntime."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate_structured(
        self,
        messages: list[dict[str, str]],
        output_type: type,
        *,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ):
        # Extract system message if present
        system = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)

        # Force JSON output by appending instruction
        schema = output_type.model_json_schema()
        json_instruction = f"\n\nRespond with ONLY valid JSON matching this schema (no markdown, no explanation):\n{json.dumps(schema, indent=2)}"

        if user_messages:
            user_messages[-1]["content"] += json_instruction

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system if system else "You are a helpful assistant that outputs only valid JSON.",
            messages=user_messages,
        )
        text = response.content[0].text.strip()

        # Clean up response — extract JSON
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            text = json_match.group(1).strip()

        # Find first { or [
        for start in ['{', '[']:
            idx = text.find(start)
            if idx >= 0:
                text = text[idx:]
                break

        return output_type.model_validate_json(text)

    def generate_variation(
        self,
        output,
        output_type: type,
        *,
        context: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 5000,
    ):
        if isinstance(output, BaseModel):
            output_text = output.model_dump_json()
        else:
            output_text = str(output)

        messages = [{"role": "user", "content": f"Generate a variation of this output:\n{output_text}"}]
        if context:
            messages[0]["content"] = f"Context: {context}\n\n{messages[0]['content']}"

        return self.generate_structured(messages, output_type, temperature=temperature, max_tokens=max_tokens)

    async def agenerate_structured(self, messages, output_type, **kwargs):
        # Sync fallback
        return self.generate_structured(messages, output_type, **kwargs)

    async def agenerate_variation(self, output, output_type, **kwargs):
        return self.generate_variation(output, output_type, **kwargs)


def load_contract_passages(max_contracts: int = 30) -> list[str]:
    """Load contract passages from CUAD for augmentation."""
    from datasets import load_dataset

    print("Loading CUAD contracts...")
    ds = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)

    # Get unique contexts grouped by contract
    seen = set()
    passages = []
    contract_count = 0
    last_title = None

    for ex in ds:
        title = ex["title"]
        context = ex["context"]
        key = context[:200]

        if key in seen:
            continue
        seen.add(key)

        if title != last_title:
            contract_count += 1
            last_title = title
            if contract_count > max_contracts:
                break

        # Only use passages with substantial content
        if len(context) > 500:
            passages.append(context)

    print(f"  {len(passages)} passages from {min(contract_count, max_contracts)} contracts")
    return passages


def extract_questions_from_output(output: str) -> list[str]:
    """Extract individual questions from QA augmentation output."""
    import re
    questions = []

    # Pattern 1: "**Question:** How much..." or "Q1: How much..."
    for match in re.finditer(r'(?:\*\*Question:\*\*|Q\d+[:.]\s*(?:\*\*Question:\*\*)?\s*)(.*?)(?:\n|$)', output):
        q = match.group(1).strip().strip('*').strip()
        if q and len(q) > 10:
            questions.append(q)

    # Pattern 2: Numbered list "1. How much..."
    if not questions:
        for match in re.finditer(r'^\s*\d+\.\s+(.+?)$', output, re.MULTILINE):
            q = match.group(1).strip()
            if q and len(q) > 10 and q.endswith('?'):
                questions.append(q)

    # Pattern 3: Lines ending with ?
    if not questions:
        for line in output.split('\n'):
            line = line.strip().strip('*').strip('-').strip()
            if line.endswith('?') and len(line) > 10:
                questions.append(line)

    return questions


def run_qa_augmentation(passages: list[str], runtime: ModelRuntime, max_passages: int = 100) -> list[dict]:
    """Generate QA pairs from passages."""
    print(f"\nGenerating QA pairs from {min(len(passages), max_passages)} passages...")

    augmentation = QaPairAugmentation(max_qa_pairs=5)
    results = []
    seen_questions = set()

    for i, passage in enumerate(passages[:max_passages]):
        if i % 10 == 0:
            print(f"  [{i}/{min(len(passages), max_passages)}]")
        try:
            rows = run_augmentation(
                data=passage,
                augmentation=augmentation,
                runtime=runtime,
            )
            for row in rows:
                # Extract questions from the output field
                questions = extract_questions_from_output(row.output)
                for q in questions:
                    if q not in seen_questions:
                        seen_questions.add(q)
                        results.append({
                            "query": q,
                            "positive": passage,
                            "source": "qa_augmentation",
                        })
        except Exception as e:
            print(f"  Error on passage {i}: {e}")
            continue

    print(f"  Generated {len(results)} unique QA pairs")
    return results


def run_retrieval_augmentation(passages: list[str], runtime: ModelRuntime, max_pairs: int = 50) -> list[dict]:
    """Generate retrieval pairs from passage groups."""
    import random

    print(f"\nGenerating retrieval pairs from passage groups...")

    augmentation = RetrievalAugmentation()
    results = []

    # Create passage groups (3-5 passages each)
    random.seed(42)
    random.shuffle(passages)

    for i in range(0, min(len(passages), max_pairs * 4), 4):
        group = passages[i:i+4]
        if len(group) < 2:
            break

        if len(results) >= max_pairs:
            break

        try:
            rows = run_augmentation(
                data=group,
                augmentation=augmentation,
                runtime=runtime,
            )
            for row in rows:
                if hasattr(row, 'input') and hasattr(row, 'output'):
                    # The input is a question, output references a passage
                    results.append({
                        "query": row.input if row.input else row.instruction,
                        "positive": row.output,
                        "source": "retrieval_augmentation",
                    })
        except Exception as e:
            print(f"  Error on group {i//4}: {e}")
            continue

        if (i // 4) % 5 == 0:
            print(f"  [{i//4}] {len(results)} pairs so far")

    print(f"  Generated {len(results)} retrieval pairs")
    return results


def augment_training_data(
    output_path: str = "augmented_training.json",
    max_contracts: int = 30,
    max_qa_passages: int = 100,
    max_retrieval_pairs: int = 50,
):
    """Main: generate augmented training data."""
    runtime = AnthropicRuntime()

    # Load passages
    passages = load_contract_passages(max_contracts=max_contracts)

    # Run augmentations
    qa_results = run_qa_augmentation(passages, runtime, max_passages=max_qa_passages)
    retrieval_results = run_retrieval_augmentation(passages, runtime, max_pairs=max_retrieval_pairs)

    # Combine
    all_results = qa_results + retrieval_results
    print(f"\nTotal: {len(all_results)} augmented pairs ({len(qa_results)} QA + {len(retrieval_results)} retrieval)")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {output_path}")

    return all_results


if __name__ == "__main__":
    max_qa = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_ret = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    augment_training_data(max_qa_passages=max_qa, max_retrieval_pairs=max_ret)
