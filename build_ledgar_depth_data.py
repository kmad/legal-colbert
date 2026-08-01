"""Build the P7 depth-mined LEDGAR dataset for priority clause families.

Strategy (2026-07-31 dataset research + user direction): be VERY good at a few
core clause types — assignment, termination, notices, cure — rather than broad
label coverage. Mines the FULL original LEDGAR corpus (raw section-heading
labels, SEC EDGAR 2016-2019) for curated subtypes within those families, keeps
the rigid query template that P3-P6 proved safe, and adds WITHIN-FAMILY hard
negatives so the model learns subtype discrimination (termination-for-cause vs
for-convenience etc. — exactly what the failing MLEB queries test).

Query side is untouched by design: `Find provisions related to "X".` only.
Do NOT switch to MLEB-style descriptive queries (P1/P2 regression).

Usage:
    python build_ledgar_depth_data.py --ledgar-zip /path/to/ledgar.zip \
        --output-dir data_p7_depth_priority
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import struct
import zlib
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from prepare_v2_data import (
    SEED,
    balance_records_by_category,
    expand_triplets,
    load_cuad_records,
    sample_bm25_negatives,
    stable_id,
    tokenize,
    write_json,
)

# (family, subtype display name, label regex, text guard regex)
# Order matters: specific subtypes before family umbrellas (first match wins).
# The guard regex marks text that could plausibly satisfy the subtype's query;
# a provision matching the guard is never used as a negative for that subtype.
SUBTYPES = [
    # --- termination: specific first ---
    ("termination", "Termination for Cause", r"terminat.*\bfor cause|\bwith cause\b.*terminat|terminat.*\bwith cause\b", r"\bfor cause\b|\bwith cause\b"),
    ("termination", "Termination Without Cause", r"terminat.*without cause|without cause.*terminat", r"without cause"),
    ("termination", "Termination for Convenience", r"terminat.*convenience", r"convenience"),
    ("termination", "Termination for Insolvency", r"terminat.*(insolven|bankrupt)|(insolven|bankrupt).*terminat", r"insolven|bankrupt"),
    ("termination", "Termination upon Change of Control", r"terminat.*change (of|in) control|change (of|in) control.*terminat", r"change (of|in) control"),
    ("termination", "Effect of Termination", r"^effects? of termination", r"effect of termination|upon termination|following termination"),
    ("termination", "Notice of Termination", r"^notices? of termination", r"notice of termination|written notice of.{0,30}terminat"),
    ("termination", "Early Termination", r"^early termination", r"early termination"),
    ("termination", "Termination", r"^terminations?$|^termination of (this )?agreement$|^term and termination$", None),
    # --- assignment ---
    ("assignment", "No Assignment", r"^no assignments?\b|^non-?assignment|^prohibition on assignment", r"shall not (be )?assign|may not (be )?assign|no assignment|not assignable|without the prior written consent"),
    ("assignment", "Successors and Assigns", r"^successors? and assigns", r"successors and assigns|inure to the benefit"),
    ("assignment", "Assignment and Assumption", r"^assignment and assumption", r"assignment and assumption|assumes"),
    ("assignment", "Assignment", r"^assignments?$|^assignability$", None),
    # --- notices ---
    ("notices", "Notice of Default", r"^notices? of default", r"notice of .{0,25}default"),
    ("notices", "Notice Addresses", r"^notice addresses?$|^addresses for notices?$", r"address"),
    ("notices", "Notices", r"^notices?$|^notices, etc\.?$|^notices generally$|^notice provisions?$|^manner of (giving )?notices?$", None),
    # --- cure ---
    ("cure", "Cure Period", r"cure period|opportunity to cure|right to cure|^notice and cure$|^cure$|^cure provisions?$|obligation to cure", r"\bcure\b|remedy such|period to remedy"),
]

FAMILY_TEXT_GUARD = {
    "termination": r"terminat",
    "assignment": r"assign",
    "notices": r"notice",
    "cure": r"\bcure\b",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def depth_query(subtype: str) -> str:
    return f'Find provisions related to "{subtype}".'


def stream_ledgar_rows(zip_path: Path):
    """Yield rows from the (possibly truncated) nested LEDGAR zip.

    Outer zip stores LEDGAR/LEDGAR_2016-2019.jsonl.zip uncompressed; the inner
    zip deflates sec_corpus_2016-2019.jsonl. Deflate decompresses front-to-back,
    so a partially downloaded archive still yields every complete row.
    """
    f = open(zip_path, "rb")

    def lfh():
        sig = f.read(4)
        if sig != b"PK\x03\x04":
            raise ValueError(f"unexpected zip signature {sig!r}")
        _v, _fl, comp, _mt, _md, _crc, _cs, _us, nlen, elen = struct.unpack(
            "<HHHHHIIIHH", f.read(26)
        )
        name = f.read(nlen).decode()
        f.read(elen)
        return name, comp

    lfh()  # LEDGAR/ directory entry
    lfh()  # inner zip entry (stored)
    _name, comp = lfh()  # sec_corpus_2016-2019.jsonl
    decomp = zlib.decompressobj(-15) if comp == 8 else None
    tail = b""
    while True:
        chunk = f.read(1 << 20)
        if not chunk:
            break
        try:
            data = decomp.decompress(chunk) if decomp else chunk
        except zlib.error:
            break  # truncated tail of a partial download
        tail += data
        lines = tail.split(b"\n")
        tail = lines.pop()
        for line in lines:
            try:
                yield json.loads(line)
            except Exception:
                continue


def mine_subtypes(zip_path: Path, min_words: int, max_words: int) -> dict[str, dict]:
    """Return subtype -> {family, texts} with first-match-wins label routing."""
    compiled = [
        (family, name, re.compile(label_re, re.I), re.compile(guard, re.I) if guard else None)
        for family, name, label_re, guard in SUBTYPES
    ]
    buckets: dict[str, dict] = {
        name: {"family": family, "texts": [], "seen": set()}
        for family, name, _, _ in SUBTYPES
    }
    n_rows = 0
    for row in stream_ledgar_rows(zip_path):
        n_rows += 1
        labels = row.get("label") or []
        if isinstance(labels, str):
            labels = [labels]
        text = normalize_text(str(row.get("provision") or ""))
        n_words = len(text.split())
        if n_words < min_words or n_words > max_words:
            continue
        for label in labels:
            hit = next(
                ((fam, name) for fam, name, lre, _ in compiled if lre.search(label)),
                None,
            )
            if hit is None:
                continue
            _, name = hit
            b = buckets[name]
            if text not in b["seen"]:
                b["seen"].add(text)
                b["texts"].append(text)
            break
    print(f"  scanned {n_rows:,} LEDGAR rows")
    for name, b in buckets.items():
        del b["seen"]
    return buckets


def load_mleb_corpus_texts() -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(
        "isaacus/contractual-clause-retrieval",
        "corpus",
        revision="48ed7bcb1f50896a0f71461a04b2df0ca84329d9",
        split="corpus",
    )
    return [normalize_text(r["text"]).lower() for r in ds]


def drop_contaminated(buckets: dict[str, dict], mleb_texts: list[str]) -> int:
    dropped = 0
    for b in buckets.values():
        kept = []
        for t in b["texts"]:
            tl = t.lower()
            if any(tl in m or m in tl for m in mleb_texts):
                dropped += 1
            else:
                kept.append(t)
        b["texts"] = kept
    return dropped


def build_depth_records(
    buckets: dict[str, dict],
    negatives_per_positive: int,
    max_within_family: int,
    rng: random.Random,
) -> list[dict]:
    guards = {name: (re.compile(g, re.I) if g else None) for _f, name, _l, g in SUBTYPES}
    family_guard = {f: re.compile(p, re.I) for f, p in FAMILY_TEXT_GUARD.items()}
    umbrella = {name for _f, name, _l, g in SUBTYPES if g is None}

    # global BM25 pool: every mined provision, tagged with its family
    pool = [(b["family"], t) for name, b in buckets.items() for t in b["texts"]]
    corpus = [t for _f, t in pool]
    corpus_family = [f for f, _t in pool]
    bm25 = BM25Okapi([tokenize(t) for t in corpus])

    records = []
    for name, b in buckets.items():
        family = b["family"]
        query = depth_query(name)
        positives = set(b["texts"])
        guard = guards[name]

        # within-family candidates: sibling subtypes, text fails this subtype's guard
        siblings = []
        if name not in umbrella and guard is not None:
            for other, ob in buckets.items():
                if other == name or ob["family"] != family:
                    continue
                for t in ob["texts"]:
                    if not guard.search(t):
                        siblings.append(t)

        for positive in b["texts"]:
            negs = []
            if siblings:
                negs = rng.sample(siblings, min(max_within_family, len(siblings)))
            need = negatives_per_positive - len(negs)
            if need > 0:
                # global BM25 negatives; for umbrella queries exclude the whole family
                scores = bm25.get_scores(tokenize(query + " " + positive[:200]))
                order = sorted(range(len(corpus)), key=lambda i: -scores[i])
                for i in order:
                    t = corpus[i]
                    if t in positives or t in negs:
                        continue
                    if name in umbrella and corpus_family[i] == family:
                        continue
                    if guard is not None and guard.search(t):
                        continue
                    if name in umbrella and family_guard[family].search(t):
                        continue
                    negs.append(t)
                    if len(negs) >= negatives_per_positive:
                        break
            if negs:
                records.append({
                    "source": f"ledgar_depth:{family}:{name}",
                    "query": query,
                    "positive": positive,
                    "negatives": negs,
                    "positive_grade": 1,
                })
    return records


def build_depth_eval(buckets: dict[str, dict]) -> tuple[dict, dict, dict, dict]:
    queries, corpus, categories = {}, {}, {}
    qrels: dict[str, dict] = defaultdict(dict)
    for name, b in buckets.items():
        if not b["texts"]:
            continue
        query = depth_query(name)
        qid = stable_id("depth-q", query)
        queries[qid] = query
        categories[qid] = name
        for t in b["texts"]:
            did = stable_id("depth-d", t)
            corpus[did] = t
            qrels[qid][did] = 1
    return queries, corpus, {q: dict(d) for q, d in qrels.items()}, categories


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledgar-zip", required=True)
    ap.add_argument("--output-dir", default="data_p7_depth_priority")
    ap.add_argument("--negatives-per-positive", type=int, default=6)
    ap.add_argument("--max-within-family", type=int, default=4)
    ap.add_argument("--max-per-subtype", type=int, default=600)
    ap.add_argument("--eval-per-subtype", type=int, default=30)
    ap.add_argument("--min-subtype-size", type=int, default=30)
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--max-words", type=int, default=320)
    ap.add_argument("--cuad-dev-fraction", type=float, default=0.10)
    ap.add_argument("--cuad-test-fraction", type=float, default=0.10)
    ap.add_argument("--cuad-train-window", type=int, default=256)
    ap.add_argument("--cuad-balance-ceiling", type=int, default=400)
    args = ap.parse_args()

    rng = random.Random(SEED)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Mining priority-family subtypes from full LEDGAR...")
    buckets = mine_subtypes(Path(args.ledgar_zip), args.min_words, args.max_words)

    print("Contamination guard vs MLEB corpus...")
    dropped = drop_contaminated(buckets, load_mleb_corpus_texts())
    print(f"  dropped {dropped} provisions overlapping MLEB passages")

    # drop tiny subtypes, cap + hold out eval slice
    eval_buckets: dict[str, dict] = {}
    for name in list(buckets):
        b = buckets[name]
        rng.shuffle(b["texts"])
        if len(b["texts"]) < args.min_subtype_size:
            print(f"  dropping subtype {name!r}: only {len(b['texts'])} provisions")
            del buckets[name]
            continue
        n_eval = min(args.eval_per_subtype, len(b["texts"]) // 5)
        eval_buckets[name] = {"family": b["family"], "texts": b["texts"][:n_eval]}
        b["texts"] = b["texts"][n_eval : n_eval + args.max_per_subtype]

    print("Building depth records with within-family hard negatives...")
    depth_records = build_depth_records(
        buckets, args.negatives_per_positive, args.max_within_family, rng
    )
    print(f"  depth records: {len(depth_records)}")

    print("Loading CUAD V9-style anchor records...")
    cuad_records, cuad_eval = load_cuad_records(
        args.cuad_dev_fraction,
        args.cuad_test_fraction,
        args.negatives_per_positive,
        rng,
        train_window=args.cuad_train_window,
    )
    if args.cuad_balance_ceiling > 0:
        cuad_records, _ = balance_records_by_category(
            cuad_records, args.cuad_balance_ceiling, 0, rng
        )
    print(f"  CUAD records: {len(cuad_records)}")

    records = cuad_records + depth_records
    dataset = expand_triplets(records, rng)
    dataset.save_to_disk(str(out / "train"))

    for split, files in cuad_eval.items():
        queries, corpus, qrels = files
        for kind, obj in zip(("queries", "corpus", "qrels"), (queries, corpus, qrels)):
            write_json(out / f"cuad_{split}_{kind}.json", obj)

    q, c, qr, cats = build_depth_eval(eval_buckets)
    for kind, obj in zip(
        ("queries", "corpus", "qrels", "categories"), (q, c, qr, cats)
    ):
        write_json(out / f"depth_validation_{kind}.json", obj)

    manifest = {
        "seed": SEED,
        "triplets": len(dataset),
        "multi_negative_records": len(records),
        "mleb_contamination_dropped": dropped,
        "sources": {
            "cuad_records": len(cuad_records),
            "depth_records": len(depth_records),
        },
        "subtype_train_counts": {n: len(b["texts"]) for n, b in sorted(buckets.items())},
        "subtype_eval_counts": {n: len(b["texts"]) for n, b in sorted(eval_buckets.items())},
        "depth_validation": {
            "queries": len(q),
            "corpus": len(c),
            "qrels": sum(len(v) for v in qr.values()),
        },
        "params": {
            k: getattr(args, k)
            for k in (
                "negatives_per_positive", "max_within_family", "max_per_subtype",
                "eval_per_subtype", "min_words", "max_words",
                "cuad_train_window", "cuad_balance_ceiling",
            )
        },
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
