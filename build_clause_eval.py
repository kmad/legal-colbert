"""Build a trustworthy clause-retrieval eval that tracks MLEB.

Two problems with the legacy CUAD eval (`cuad_dev`/`cuad_test`):
  1. Positives are raw ``answer_start +/- 512`` char windows (label-noisy,
     truncated mid-word).
  2. Queries are a rigid template ("Highlight the parts ... related to 'X'"),
     unlike MLEB's natural-language clause definitions. The legacy eval even
     anti-correlates with MLEB across our own models.

This builder reproduces the SAME held-out CUAD contract split used by the
training runs (SEED=42, 10%/10%), then emits BEIR-style eval files with:
  - clean, word-boundary-aligned clause spans (`aligned_span`);
  - a choice of query style: ``mleb`` (descriptive definitions, matching the
    real benchmark) or ``template`` (legacy CUAD wording, for ablation);
  - a per-query category map so eval can report macro metrics and exclude
    tiny-n categories that otherwise dominate the noise.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from prepare_v2_data import SEED, aligned_span, stable_id

# MLEB-style natural-language definitions for each CUAD clause category.
CATEGORY_DEFINITIONS = {
    "Affiliate License-Licensee": "This is a contractual provision that grants the licensee's affiliates the right to use the licensed materials.",
    "Affiliate License-Licensor": "This is a contractual provision under which a party licenses intellectual property on behalf of or including its affiliates.",
    "Agreement Date": "This is the date on which the contract is signed or entered into.",
    "Anti-Assignment": "This is a contractual provision that restricts a party from assigning or transferring its rights or obligations under the contract without consent.",
    "Audit Rights": "This is a contractual provision that grants a party the right to inspect or audit the books, records, or premises of the counterparty.",
    "Cap On Liability": "This is a contractual provision that limits the maximum amount of liability a party can incur under the contract.",
    "Change Of Control": "This is a contractual provision addressing the consequences of a change in ownership or control of a contracting party.",
    "Competitive Restriction Exception": "This is a contractual provision that carves out exceptions to a non-compete or other competitive restriction.",
    "Covenant Not To Sue": "This is a contractual provision in which a party agrees not to sue or bring legal action against the counterparty.",
    "Document Name": "This is the title or name of the contract document.",
    "Effective Date": "This is the date on which the contract becomes effective.",
    "Exclusivity": "This is a contractual provision that grants exclusive rights or imposes exclusive dealing obligations on a party.",
    "Expiration Date": "This is the date on which the contract expires or terminates.",
    "Governing Law": "This is a contractual provision specifying which jurisdiction's laws govern the contract.",
    "Insurance": "This is a contractual provision that requires a party to maintain insurance coverage.",
    "Ip Ownership Assignment": "This is a contractual provision that assigns ownership of intellectual property created under the contract.",
    "Irrevocable Or Perpetual License": "This is a contractual provision granting a license that is irrevocable or perpetual in duration.",
    "Joint Ip Ownership": "This is a contractual provision that provides for joint or shared ownership of intellectual property.",
    "License Grant": "This is a contractual provision that grants a license to use intellectual property or other rights.",
    "Liquidated Damages": "This is a contractual provision specifying a predetermined amount of damages payable upon breach.",
    "Minimum Commitment": "This is a contractual provision that obligates a party to a minimum purchase, sales, or usage commitment.",
    "Most Favored Nation": "This is a contractual provision entitling a party to terms at least as favorable as those offered to any third party.",
    "No-Solicit Of Customers": "This is a contractual provision that restricts a party from soliciting the counterparty's customers.",
    "No-Solicit Of Employees": "This is a contractual provision that restricts a party from soliciting or hiring the counterparty's employees.",
    "Non-Compete": "This is a contractual provision that restricts a party from competing with the counterparty.",
    "Non-Disparagement": "This is a contractual provision that prohibits a party from making disparaging statements about the counterparty.",
    "Non-Transferable License": "This is a contractual provision granting a license that cannot be transferred or assigned.",
    "Notice Period To Terminate Renewal": "This is a contractual provision specifying the notice period required to terminate or prevent automatic renewal of the contract.",
    "Parties": "This identifies the parties who enter into the contract.",
    "Post-Termination Services": "This is a contractual provision governing obligations or services that continue after the contract terminates.",
    "Price Restrictions": "This is a contractual provision that restricts or controls the prices a party may charge.",
    "Renewal Term": "This is a contractual provision specifying the length or terms of contract renewal.",
    "Revenue/Profit Sharing": "This is a contractual provision requiring a party to share revenue or profits with the counterparty.",
    "Rofr/Rofo/Rofn": "This is a contractual provision granting a right of first refusal, first offer, or first negotiation.",
    "Source Code Escrow": "This is a contractual provision requiring software source code to be deposited into escrow.",
    "Termination For Convenience": "This is a contractual provision that allows a party to terminate the contract for convenience, without cause.",
    "Third Party Beneficiary": "This is a contractual provision that grants rights to a third party who is not a signatory to the contract.",
    "Uncapped Liability": "This is a contractual provision specifying that a party's liability is not capped or limited.",
    "Unlimited/All-You-Can-Eat-License": "This is a contractual provision granting an unlimited or all-you-can-eat license to use the materials.",
    "Volume Restriction": "This is a contractual provision that restricts the volume or quantity a party may purchase or use.",
    "Warranty Duration": "This is a contractual provision specifying the duration of a warranty.",
}


def category_from_question(question: str) -> str | None:
    import re

    match = re.search(r'related to "([^"]+)"', question)
    return match.group(1) if match else None


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="eval_clause")
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--query-style", choices=["mleb", "template"], default="mleb")
    args = parser.parse_args()

    ds = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)

    # Reproduce prepare_v2_data.load_cuad_records' example filtering and split
    # EXACTLY so the held-out contracts match the training runs (disjoint from
    # train). The +/-512 positive is computed only to mirror the legacy length
    # filter; the eval uses the clean aligned span instead.
    examples = []
    contract_to_examples = defaultdict(list)
    for ex in ds:
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
        legacy_positive = context[start:end].strip()
        if len(legacy_positive.split()) < 8:
            continue
        contract_id = (
            ex.get("title")
            or ex.get("contract_name")
            or str(ex.get("id", "")).split("_")[0]
            or stable_id("cuad-contract", context[:2000])
        )
        category = category_from_question(ex["question"])
        clean = aligned_span(context, answer_start, answer_text, args.window)
        if category is None or len(clean.split()) < 8:
            # Keep the row for split fidelity but mark it unusable for eval.
            clean = ""
        examples.append({"category": category, "span": clean, "contract_id": contract_id})
        contract_to_examples[contract_id].append(examples[-1])

    rng = random.Random(SEED)
    contracts = list(contract_to_examples)
    rng.shuffle(contracts)
    n_dev = max(1, int(len(contracts) * args.dev_fraction))
    n_test = max(1, int(len(contracts) * args.test_fraction))
    dev_contracts = set(contracts[:n_dev])
    test_contracts = set(contracts[n_dev : n_dev + n_test])

    out = Path(args.output_dir)
    for split, contract_set in (("clause_dev", dev_contracts), ("clause_test", test_contracts)):
        queries, corpus, qrels, qid_category = {}, {}, defaultdict(dict), {}
        for ex in examples:
            if ex["contract_id"] not in contract_set or not ex["span"] or not ex["category"]:
                continue
            cat = ex["category"]
            if cat not in CATEGORY_DEFINITIONS:
                continue
            query_text = (
                CATEGORY_DEFINITIONS[cat]
                if args.query_style == "mleb"
                else f'Highlight the parts (if any) of this contract related to "{cat}" that should be reviewed by a lawyer.'
            )
            qid = stable_id("clause-q", cat)  # one query per category
            did = stable_id("clause-d", ex["span"])
            queries[qid] = query_text
            corpus[did] = ex["span"]
            qrels[qid][did] = 1
            qid_category[qid] = cat

        write_json(out / f"{split}_queries.json", queries)
        write_json(out / f"{split}_corpus.json", corpus)
        write_json(out / f"{split}_qrels.json", {q: dict(v) for q, v in qrels.items()})
        write_json(out / f"{split}_categories.json", qid_category)
        print(
            f"{split}: {len(queries)} queries, {len(corpus)} docs, "
            f"{sum(len(v) for v in qrels.values())} qrels "
            f"(style={args.query_style}, window={args.window})"
        )

    write_json(out / "build_config.json", vars(args))


if __name__ == "__main__":
    main()
