"""Plot our legal ColBERT training journey and key metrics.

The chart focuses on the actual progression we ran locally:
V9 -> P5 -> P6a/b/c, with MLEB as the main target and ACORD as the regression
check. This is meant to be a compact, shareable PNG rather than a leaderboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = [
    {
        "label": "V9",
        "mleb": "output/legal-colbert-v9-spans-capped.mleb.json",
        "acord": "output/legal-colbert-v9-spans-capped.acord_test.json",
        "clause": "output/clause_mleb_V9.json",
    },
    {
        "label": "P5",
        "mleb": "output/legal-colbert-p5-v9-ledgar-400.mleb.json",
        "acord": "output/legal-colbert-p5-v9-ledgar-400.acord_test.json",
        "clause": "output/legal-colbert-p5-v9-ledgar-400.clause_mleb.json",
    },
    {
        "label": "P6a",
        "mleb": "output/legal-colbert-p6a-p5-anchor-200.mleb.json",
        "acord": "output/legal-colbert-p6a-p5-anchor-200.acord_test.json",
        "clause": "output/legal-colbert-p6a-p5-anchor-200.clause_mleb.json",
    },
    {
        "label": "P6b",
        "mleb": "output/legal-colbert-p6b-p5-anchor-300.mleb.json",
        "acord": "output/legal-colbert-p6b-p5-anchor-300.acord_test.json",
        "clause": "output/legal-colbert-p6b-p5-anchor-300.clause_mleb.json",
    },
    {
        "label": "P6c",
        "mleb": "output/legal-colbert-p6c-p5-ledgar-200.mleb.json",
        "acord": "output/legal-colbert-p6c-p5-ledgar-200.acord_test.json",
        "clause": "output/legal-colbert-p6c-p5-ledgar-200.clause_mleb.json",
    },
]


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def metric(path: Path, *keys: str) -> float:
    d = load_json(path)
    if len(d) == 1 and isinstance(next(iter(d.values())), dict):
        d = next(iter(d.values()))
    for key in keys:
        if key in d:
            return float(d[key])
        for cand in d.keys():
            if cand.lower() == key.lower():
                return float(d[cand])
    raise KeyError(f"None of {keys} found in {path}")


def main() -> None:
    rows = []
    for item in MODELS:
        rows.append(
            {
                "label": item["label"],
                "mleb_ndcg": metric(Path(item["mleb"]), "NDCG@10"),
                "mleb_map": metric(Path(item["mleb"]), "MAP"),
                "acord_ndcg": metric(Path(item["acord"]), "ndcg@10"),
                "clause_ndcg": metric(Path(item["clause"]), "macro_ndcg@10"),
                "clause_ndcg_n5": metric(Path(item["clause"]), "macro_ndcg@10_n>=5"),
            }
        )

    x = list(range(len(rows)))
    labels = [r["label"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=160, sharex=True)
    fig.patch.set_facecolor("white")
    colors = {
        "mleb": "#c0392b",
        "acord": "#2c7fb8",
        "clause": "#3d9970",
    }

    def draw(ax, key, title, color):
        ys = [r[key] * 100 for r in rows]
        ax.plot(x, ys, marker="o", lw=2.4, color=color)
        ax.scatter(x, ys, s=48, color=color, zorder=3)
        for xi, yi in zip(x, ys):
            ax.text(xi, yi + 0.12, f"{yi:.2f}", ha="center", va="bottom", fontsize=8, color="#2f3a40")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(min(ys) - 0.4, max(ys) + 0.5)
        ax.grid(True, axis="y", ls="--", alpha=0.3)
        ax.set_xticks(x, labels, fontsize=9)

    draw(axes[0], "mleb_ndcg", "MLEB NDCG@10", colors["mleb"])
    draw(axes[1], "acord_ndcg", "ACORD test NDCG@10", colors["acord"])
    draw(axes[2], "clause_ndcg_n5", "Clause eval NDCG@10 (n>=5)", colors["clause"])

    axes[0].set_ylabel("Score")
    axes[1].set_ylabel("Score")
    axes[2].set_ylabel("Score")

    # Highlight the current best point.
    best_idx = max(range(len(rows)), key=lambda i: rows[i]["mleb_ndcg"])
    bx = x[best_idx]
    by = rows[best_idx]["mleb_ndcg"] * 100
    axes[0].annotate(
        "current best",
        xy=(bx, by),
        xytext=(bx - 0.3, by + 0.35),
        fontsize=10,
        fontweight="bold",
        color="#8e1e14",
        arrowprops=dict(arrowstyle="->", color="#8e1e14", lw=1.1),
    )

    fig.suptitle(
        "Legal clause retrieval training journey: V9 -> P5 -> P6 sweep",
        fontsize=15,
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        -0.02,
        "P6b is the current MLEB-best model; ACORD remains the counter-signal.",
        ha="center",
        fontsize=10,
        color="#59656f",
    )
    fig.tight_layout()

    out = Path("output/legal-colbert-results-journey.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
