"""Compare tuned MLEB metrics against independent blind EDGAR metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = {
    "v1": {
        "mleb": "output/mleb_v1_remote.json",
        "blind": "output/legal-colbert-v1.blind_edgar.json",
    },
    "v9": {
        "mleb": "output/legal-colbert-v9-spans-capped.mleb.json",
        "blind": "output/legal-colbert-v9-spans-capped.blind_edgar.json",
    },
    "p5": {
        "mleb": "output/legal-colbert-p5-v9-ledgar-400.mleb.json",
        "blind": "output/legal-colbert-p5-v9-ledgar-400.blind_edgar.json",
    },
    "p6b": {
        "mleb": "output/legal-colbert-p6b-p5-anchor-300.mleb.json",
        "blind": "output/legal-colbert-p6b-p5-anchor-300.blind_edgar.json",
    },
}


def load_metric(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    keys = {key, key.upper(), key.replace("@", "@").upper()}
    for candidate in keys:
        if candidate in data:
            return float(data[candidate])
    for value in data.values():
        if isinstance(value, dict):
            for candidate in keys:
                if candidate in value:
                    return float(value[candidate])
    return None


BM25_BASELINE = "output/bm25_baseline.json"


def load_bm25_row(path: Path) -> dict | None:
    """Lexical baseline row (eval_bm25.py output, stopword-filtered variant)."""
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    best = data.get("bm25_stopfiltered") or data.get("bm25_plain")
    if not best:
        return None
    return {
        "model": "bm25",
        "mleb_ndcg@10": best.get("mleb", {}).get("ndcg@10"),
        "blind_ndcg@10": best.get("blind_edgar", {}).get("macro_ndcg@10"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default="output/blind_overfit_assessment.json")
    args = parser.parse_args()

    rows = []
    for name, paths in MODELS.items():
        mleb = load_metric(Path(paths["mleb"]), "ndcg@10")
        blind = load_metric(Path(paths["blind"]), "ndcg@10")
        rows.append({"model": name, "mleb_ndcg@10": mleb, "blind_ndcg@10": blind})

    bm25 = load_bm25_row(Path(BM25_BASELINE))
    if bm25:
        rows.append(bm25)
        for row in rows:
            if row["model"] != "bm25" and row["blind_ndcg@10"] is not None and bm25["blind_ndcg@10"] is not None:
                row["blind_beats_bm25"] = row["blind_ndcg@10"] > bm25["blind_ndcg@10"]

    baseline = next((row for row in rows if row["model"] == "v1"), None)
    for row in rows:
        if baseline and row["mleb_ndcg@10"] is not None and baseline["mleb_ndcg@10"] is not None:
            row["mleb_delta_vs_v1"] = round(row["mleb_ndcg@10"] - baseline["mleb_ndcg@10"], 4)
        if baseline and row["blind_ndcg@10"] is not None and baseline["blind_ndcg@10"] is not None:
            row["blind_delta_vs_v1"] = round(row["blind_ndcg@10"] - baseline["blind_ndcg@10"], 4)

    p5 = next((row for row in rows if row["model"] == "p5"), None)
    p6b = next((row for row in rows if row["model"] == "p6b"), None)
    p6_assessment = None
    if p5 and p6b and p5["mleb_ndcg@10"] is not None and p6b["mleb_ndcg@10"] is not None and p5["blind_ndcg@10"] is not None and p6b["blind_ndcg@10"] is not None:
        mleb_delta = round(p6b["mleb_ndcg@10"] - p5["mleb_ndcg@10"], 4)
        blind_delta = round(p6b["blind_ndcg@10"] - p5["blind_ndcg@10"], 4)
        if mleb_delta > 0 and blind_delta < 0:
            verdict = "p6b_mleb_uplift_does_not_transfer_to_blind"
        elif mleb_delta > 0 and blind_delta >= 0:
            verdict = "p6b_mleb_uplift_has_some_blind_support"
        else:
            verdict = "p6b_not_mleb_improved"
        p6_assessment = {"mleb_delta_p6b_vs_p5": mleb_delta, "blind_delta_p6b_vs_p5": blind_delta, "verdict": verdict}

    result = {
        "metric": "ndcg@10",
        "rows": rows,
        "p6_assessment": p6_assessment,
        "caveat": "Blind EDGAR labels are heading-derived weak labels. Use them as an overfitting diagnostic, not as final benchmark truth.",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
