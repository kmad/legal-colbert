"""Mine the full LEDGAR corpus for the EXTRACTOR line's clause families (P8).

Extends the P7 priority mining (assignment/termination/notices/cure) with the
common commercial types the extractor should be VERY good at: governing law,
indemnification, confidentiality, limitation of liability, IP, reps &
warranties, payment, insurance, force majeure, survival, dispute resolution,
audit. Emits a provisions JSON consumed by build_extractor_data.py (locally or
on a GPU pod for model-mined hard negatives) plus a held-out subtype
diagnostic (depth_validation_v2).

Usage:
    python mine_ledgar_families.py --ledgar-zip /path/ledgar.zip \
        --output extractor_provisions.json --eval-dir eval_depth_v2
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from build_ledgar_depth_data import (
    SUBTYPES as P7_SUBTYPES,
    normalize_text,
    stream_ledgar_rows,
    depth_query,
    load_mleb_corpus_texts,
)
from prepare_v2_data import SEED, stable_id, write_json

# Additional common-type subtypes for the extractor line.
# (family, subtype display name, label regex, text guard regex)
COMMON_SUBTYPES = [
    ("governing_law", "Governing Law", r"^governing laws?$|^applicable laws?$|^choice of law$", r"governed by|laws of the"),
    ("governing_law", "Consent to Jurisdiction", r"^consent to jurisdiction|^submission to jurisdiction|^jurisdictions?$|^venue$|^forum", r"jurisdiction|venue|forum"),
    ("governing_law", "Waiver of Jury Trial", r"^waiver of jury trial|^jury trial waiver|^waiver of trial by jury", r"jury"),
    ("indemnification", "Indemnification", r"^indemnifications?$|^indemnity$|^indemnification and", r"indemnif|hold harmless"),
    ("indemnification", "Indemnification by the Company", r"^indemnification by", r"indemnif"),
    ("indemnification", "Indemnification Procedures", r"^indemnification procedures?$|^procedures? for indemnification|^notice of claims?$", r"indemnif|claim"),
    ("confidentiality", "Confidentiality", r"^confidentiality$|^confidential information$|^non-?disclosure$|^confidentiality obligations$", r"confidential"),
    ("confidentiality", "Permitted Disclosures", r"^permitted disclosures?$|^required disclosures?$|^compelled disclosure", r"disclos"),
    ("confidentiality", "Return of Confidential Information", r"^return of confidential|^return or destruction", r"return|destroy|destruction"),
    ("liability", "Limitation of Liability", r"^limitations? o[fn] liability$|^liability cap|^cap on liability|^maximum liability", r"liab"),
    ("liability", "Exclusion of Consequential Damages", r"consequential damages|^exclusion of damages|^waiver of consequential|^no consequential", r"consequential|indirect|special|punitive"),
    ("liability", "Disclaimer of Warranties", r"^disclaimer of warrant|^warranty disclaimer|^no other warranties|^disclaimer$", r"disclaim|as is|no warrant"),
    ("intellectual_property", "Intellectual Property", r"^intellectual property$|^intellectual property rights$|^proprietary rights$", r"intellectual property|proprietary"),
    ("intellectual_property", "License Grant", r"^license grants?$|^grant of license|^licenses? granted|^grant of rights$", r"licens|grant"),
    ("intellectual_property", "Ownership of Work Product", r"^work product$|^ownership of (work product|inventions|intellectual property|developments)|^inventions$|^assignment of inventions", r"work product|invention|develop"),
    ("reps_warranties", "Representations and Warranties", r"^representations and warranties( of| by)?|^representations$|^warranties$|^mutual representations", r"represents? and warrants?|representations"),
    ("reps_warranties", "Authority", r"^authority$|^authorizations?$|^power and authority|^corporate (power|authority)|^due authorization", r"authori[tz]|power"),
    ("reps_warranties", "Organization and Good Standing", r"^organization( and good standing)?$|^good standing$|^due organization|^existence", r"organized|good standing|existence"),
    ("payment", "Payment Terms", r"^payments?$|^payment terms$|^payment of fees$|^fees and payments?$|^invoic", r"pay|fee|invoice"),
    ("payment", "Late Payment Interest", r"^late (payments?|charges?|fees?)|^interest on (late|overdue)|^default interest$|^overdue", r"interest|late|overdue"),
    ("payment", "Taxes", r"^taxes$|^withholding( tax(es)?)?$|^tax matters$|^sales tax", r"tax"),
    ("insurance", "Insurance", r"^insurances?$|^insurance requirements?$|^insurance coverage$", r"insur"),
    ("force_majeure", "Force Majeure", r"^force majeure", r"force majeure|beyond.{0,25}control|acts? of god"),
    ("survival", "Survival", r"^survival$|^survival of", r"surviv"),
    ("dispute_resolution", "Arbitration", r"^arbitrations?$|^binding arbitration|^arbitration procedures?$", r"arbitrat"),
    ("dispute_resolution", "Dispute Resolution", r"^dispute resolution|^disputes?$|^resolution of disputes", r"dispute"),
    ("audit", "Audit Rights", r"^audits?( rights?)?$|^books and records$|^inspection( rights?)?$|^right to audit", r"audit|inspect|books and records"),
]

ALL_SUBTYPES = P7_SUBTYPES + COMMON_SUBTYPES
PRIORITY_FAMILIES = {"assignment", "termination", "notices", "cure"}


def mine(zip_path: Path, min_words: int, max_words: int) -> dict[str, dict]:
    compiled = [
        (fam, name, re.compile(lre, re.I), guard)
        for fam, name, lre, guard in ALL_SUBTYPES
    ]
    buckets = {name: {"family": fam, "guard": guard, "texts": [], "seen": set()}
               for fam, name, _l, guard in ALL_SUBTYPES}
    n = 0
    for row in stream_ledgar_rows(zip_path):
        n += 1
        labels = row.get("label") or []
        if isinstance(labels, str):
            labels = [labels]
        text = normalize_text(str(row.get("provision") or ""))
        w = len(text.split())
        if w < min_words or w > max_words:
            continue
        for label in labels:
            hit = next((name for _f, name, lre, _g in compiled if lre.search(label)), None)
            if hit is None:
                continue
            b = buckets[hit]
            if text not in b["seen"]:
                b["seen"].add(text)
                b["texts"].append(text)
            break
    print(f"scanned {n:,} rows")
    for b in buckets.values():
        del b["seen"]
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledgar-zip", required=True)
    ap.add_argument("--output", default="extractor_provisions.json")
    ap.add_argument("--eval-dir", default="eval_depth_v2")
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--max-words", type=int, default=320)
    ap.add_argument("--priority-cap", type=int, default=2000)
    ap.add_argument("--common-cap", type=int, default=1000)
    ap.add_argument("--eval-per-subtype", type=int, default=30)
    ap.add_argument("--min-subtype-size", type=int, default=30)
    args = ap.parse_args()

    rng = random.Random(SEED)
    buckets = mine(Path(args.ledgar_zip), args.min_words, args.max_words)

    print("MLEB contamination guard...")
    mleb = load_mleb_corpus_texts()
    dropped = 0
    for b in buckets.values():
        kept = []
        for t in b["texts"]:
            tl = t.lower()
            if any(tl in m or m in tl for m in mleb):
                dropped += 1
            else:
                kept.append(t)
        b["texts"] = kept
    print(f"dropped {dropped} MLEB-overlapping provisions")

    eval_q, eval_c, eval_r, eval_cat = {}, {}, defaultdict(dict), {}
    out_buckets = {}
    for name in list(buckets):
        b = buckets[name]
        rng.shuffle(b["texts"])
        if len(b["texts"]) < args.min_subtype_size:
            print(f"  dropping {name!r}: {len(b['texts'])}")
            continue
        n_eval = min(args.eval_per_subtype, len(b["texts"]) // 5)
        cap = args.priority_cap if b["family"] in PRIORITY_FAMILIES else args.common_cap
        query = depth_query(name)
        qid = stable_id("depthv2-q", query)
        eval_q[qid] = query
        eval_cat[qid] = name
        for t in b["texts"][:n_eval]:
            did = stable_id("depthv2-d", t)
            eval_c[did] = t
            eval_r[qid][did] = 1
        out_buckets[name] = {
            "family": b["family"],
            "guard": b["guard"],
            "texts": b["texts"][n_eval : n_eval + cap],
        }

    ev = Path(args.eval_dir)
    write_json(ev / "depth_v2_queries.json", eval_q)
    write_json(ev / "depth_v2_corpus.json", eval_c)
    write_json(ev / "depth_v2_qrels.json", {q: dict(d) for q, d in eval_r.items()})
    write_json(ev / "depth_v2_categories.json", eval_cat)

    with open(args.output, "w") as f:
        json.dump(out_buckets, f)

    counts = {n: len(b["texts"]) for n, b in sorted(out_buckets.items())}
    manifest = {
        "subtypes": len(out_buckets),
        "total_provisions": sum(counts.values()),
        "mleb_dropped": dropped,
        "eval": {"queries": len(eval_q), "corpus": len(eval_c)},
        "counts": counts,
    }
    write_json(ev / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
