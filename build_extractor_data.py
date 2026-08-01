"""Assemble the P8 extractor dataset from mined provisions (pod-side, GPU).

Consumes extractor_provisions.json (from mine_ledgar_families.py) and builds
train triplets with, per positive:
  - within-family hard negatives (guard-filtered sibling subtypes), plus
  - MODEL-MINED global hard negatives (--hard-negative-model, e.g. p7b):
    top-ranked pool provisions for the subtype's query that fail the subtype
    guard — the confusables the current model actually gets wrong. This was
    the single biggest V1-era win (model-based negatives) applied to the
    extractor line. Falls back to BM25 negatives without a model.
Plus the standard CUAD anchor (V9-style) at roughly 1:1.

Usage (pod):
    python build_extractor_data.py --provisions extractor_provisions.json \
        --hard-negative-model output/legal-colbert-p7b-depth-200/final \
        --output-dir data_p8_extractor
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from prepare_v2_data import (
    SEED,
    balance_records_by_category,
    expand_triplets,
    load_cuad_records,
    tokenize,
    write_json,
)
from build_ledgar_depth_data import depth_query


def model_mine_candidates(model_path: str, queries: dict[str, str], pool: list[str], top_k: int = 250):
    """Rank the pool per query with the ColBERT model; return top_k indices."""
    import torch
    from pylate import models

    model = models.ColBERT(model_name_or_path=model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_embs = model.encode(pool, batch_size=128, is_query=False, show_progress_bar=True)
    q_embs = model.encode(list(queries.values()), batch_size=32, is_query=True, show_progress_bar=False)
    d_tensors = [torch.tensor(d, device=device) for d in d_embs]

    out: dict[str, list[int]] = {}
    for qi, name in enumerate(queries):
        q = torch.tensor(q_embs[qi], device=device)
        scores = torch.empty(len(d_tensors))
        for start in range(0, len(d_tensors), 2048):
            chunk = d_tensors[start : start + 2048]
            for j, d in enumerate(chunk):
                scores[start + j] = torch.matmul(q, d.T).max(dim=1).values.sum()
        out[name] = torch.argsort(scores, descending=True)[:top_k].tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provisions", required=True)
    ap.add_argument("--output-dir", default="data_p8_extractor")
    ap.add_argument("--hard-negative-model", default="")
    ap.add_argument("--negatives-per-positive", type=int, default=6)
    ap.add_argument("--max-within-family", type=int, default=3)
    ap.add_argument("--mined-top-k", type=int, default=250)
    ap.add_argument("--cuad-records-json", default="", help="Pre-built CUAD anchor records (bypasses HF dataset loading — needed on pods where datasets>=4 rejects script datasets).")
    ap.add_argument("--cuad-dev-fraction", type=float, default=0.10)
    ap.add_argument("--cuad-test-fraction", type=float, default=0.10)
    ap.add_argument("--cuad-train-window", type=int, default=256)
    ap.add_argument("--cuad-balance-ceiling", type=int, default=400)
    args = ap.parse_args()

    rng = random.Random(SEED)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.provisions) as f:
        buckets = json.load(f)
    for b in buckets.values():
        b["guard_re"] = re.compile(b["guard"], re.I) if b.get("guard") else None

    pool = [(name, t) for name, b in buckets.items() for t in b["texts"]]
    pool_texts = [t for _n, t in pool]
    pool_subtype = [n for n, _t in pool]
    pool_family = [buckets[n]["family"] for n, _t in pool]
    print(f"pool: {len(pool_texts):,} provisions, {len(buckets)} subtypes")

    queries = {name: depth_query(name) for name in buckets}

    mined: dict[str, list[int]] = {}
    if args.hard_negative_model:
        print(f"model-mining hard negatives with {args.hard_negative_model}...")
        mined = model_mine_candidates(
            args.hard_negative_model, queries, pool_texts, args.mined_top_k
        )
    bm25 = None
    if not mined:
        print("building BM25 fallback...")
        bm25 = BM25Okapi([tokenize(t) for t in pool_texts])

    records = []
    for name, b in buckets.items():
        family = b["family"]
        guard = b["guard_re"]
        query = queries[name]
        positives = set(b["texts"])

        # within-family sibling candidates failing this subtype's guard
        siblings = []
        if guard is not None:
            for other, ob in buckets.items():
                if other == name or ob["family"] != family:
                    continue
                siblings.extend(t for t in ob["texts"] if not guard.search(t))

        # global candidate index list for this subtype
        if mined:
            cand_idx = mined[name]
        else:
            scores = bm25.get_scores(tokenize(query))
            cand_idx = sorted(range(len(pool_texts)), key=lambda i: -scores[i])[: args.mined_top_k]
        global_cands = []
        for i in cand_idx:
            t = pool_texts[i]
            if t in positives or pool_subtype[i] == name:
                continue
            if guard is not None and guard.search(t):
                continue
            if guard is None and pool_family[i] == family:
                continue  # umbrella subtype: exclude own family entirely
            global_cands.append(t)

        for positive in b["texts"]:
            negs = []
            if siblings:
                negs.extend(rng.sample(siblings, min(args.max_within_family, len(siblings))))
            pool_slice = [t for t in global_cands if t not in negs]
            need = args.negatives_per_positive - len(negs)
            if need > 0 and pool_slice:
                # sample from top candidates so negatives vary across positives
                take = min(need, len(pool_slice))
                window = pool_slice[: max(40, take * 4)]
                negs.extend(rng.sample(window, take))
            if negs:
                records.append({
                    "source": f"extractor:{family}:{name}",
                    "query": query,
                    "positive": positive,
                    "negatives": negs,
                    "positive_grade": 1,
                })
    print(f"extractor records: {len(records)}")

    print("CUAD anchor...")
    if args.cuad_records_json:
        with open(args.cuad_records_json) as f:
            cuad_records = json.load(f)
        cuad_eval = {}
    else:
        cuad_records, cuad_eval = load_cuad_records(
            args.cuad_dev_fraction, args.cuad_test_fraction,
            args.negatives_per_positive, rng, train_window=args.cuad_train_window,
        )
        if args.cuad_balance_ceiling > 0:
            cuad_records, _ = balance_records_by_category(
                cuad_records, args.cuad_balance_ceiling, 0, rng
            )
    print(f"CUAD records: {len(cuad_records)}")

    all_records = cuad_records + records
    dataset = expand_triplets(all_records, rng)
    dataset.save_to_disk(str(out / "train"))

    for split, files in cuad_eval.items():
        q, c, r = files
        for kind, obj in zip(("queries", "corpus", "qrels"), (q, c, r)):
            write_json(out / f"cuad_{split}_{kind}.json", obj)

    manifest = {
        "seed": SEED,
        "triplets": len(dataset),
        "records": len(all_records),
        "hard_negative_model": args.hard_negative_model or "bm25",
        "sources": {"cuad": len(cuad_records), "extractor": len(records)},
        "subtype_counts": {n: len(b["texts"]) for n, b in sorted(buckets.items())},
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "subtype_counts"}, indent=2))


if __name__ == "__main__":
    main()
