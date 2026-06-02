"""Param-size vs NDCG@10 chart for MLEB Contractual Clause Retrieval (open models).

NDCG@10 values are pulled live from the official MLEB results.jsonl (not typed by
hand). Parameter counts are verified per-model (HF cards / model papers). Our
point (legal-colbert-p6b, 149,015,808 params) is measured locally (0.8338) with
the same protocol that reproduces the leaderboard's BGE-M3 score exactly (0.7281).
"""

import json
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Verified parameter counts (actual, not rounded model-name sizes).
PARAMS = {
    "Qwen/Qwen3-Embedding-0.6B": 595_000_000,
    "Qwen/Qwen3-Embedding-4B": 4_020_000_000,
    "Qwen/Qwen3-Embedding-8B": 7_570_000_000,
    "Snowflake/snowflake-arctic-embed-l-v2.0": 568_000_000,
    "Snowflake/snowflake-arctic-embed-m-v2.0": 305_000_000,
    "freelawproject/modernbert-embed-base_finetune_512": 149_000_000,
    "google/embeddinggemma-300m": 308_000_000,
    "ibm-granite/granite-embedding-english-r2": 149_000_000,
    "ibm-granite/granite-embedding-small-english-r2": 47_000_000,
    "intfloat/multilingual-e5-large-instruct": 560_000_000,
    "jinaai/jina-embeddings-v4": 3_800_000_000,
    "mixedbread-ai/mxbai-embed-large-v1": 335_000_000,
    "BAAI/bge-m3": 568_000_000,
}
SHORT = {
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-0.6B",
    "Qwen/Qwen3-Embedding-4B": "Qwen3-4B",
    "Qwen/Qwen3-Embedding-8B": "Qwen3-8B",
    "Snowflake/snowflake-arctic-embed-l-v2.0": "Arctic-L v2",
    "Snowflake/snowflake-arctic-embed-m-v2.0": "Arctic-M v2",
    "freelawproject/modernbert-embed-base_finetune_512": "FreeLaw ModernBERT",
    "google/embeddinggemma-300m": "EmbeddingGemma",
    "ibm-granite/granite-embedding-english-r2": "Granite-R2",
    "ibm-granite/granite-embedding-small-english-r2": "Granite-small",
    "intfloat/multilingual-e5-large-instruct": "E5-large-instruct",
    "jinaai/jina-embeddings-v4": "Jina-v4",
    "mixedbread-ai/mxbai-embed-large-v1": "mxbai-large",
    "BAAI/bge-m3": "BGE-M3",
}

# Pull NDCG@10 for contractual-clause-retrieval from the official results.
url = "https://raw.githubusercontent.com/isaacus-dev/mleb/main/results/results.jsonl"
rows = [json.loads(l) for l in urllib.request.urlopen(url).read().decode().splitlines() if l.strip()]
scores = {}
for r in rows:
    mid = r["model"]["id"]
    for d in r["results"]:
        if d["dataset"]["id"] == "contractual-clause-retrieval":
            scores[mid] = d["score"]

pts = []
for mid, p in PARAMS.items():
    if mid in scores:
        pts.append((p, scores[mid] * 100, SHORT[mid]))
    else:
        print("WARN missing score:", mid)

OURS = (149_015_808, 83.38, "legal-colbert (ours)")  # P6b, measured locally

print("plotted points:")
for p, s, n in sorted(pts) + [OURS]:
    print(f"  {n:22} {p/1e6:7.0f}M  {s:.1f}")

fig, ax = plt.subplots(figsize=(12, 7), dpi=120)
fig.patch.set_facecolor("white")

xs = [p for p, _, _ in pts]
ys = [s for _, s, _ in pts]
ax.scatter(xs, ys, s=90, c="#9aa6b2", edgecolors="#5b6770", linewidths=1, zorder=3)
# Custom label offsets (x_pts, y_pts, ha) to avoid collisions where real
# values nearly coincide.
OFFSETS = {
    "Arctic-L v2": (-6, 11, "right"),
    "Qwen3-0.6B": (8, -17, "left"),
    "Arctic-M v2": (-6, -16, "right"),
    "mxbai-large": (4, 10, "left"),
    "BGE-M3": (10, -5, "left"),
    "Granite-R2": (0, -16, "center"),
    "EmbeddingGemma": (0, 10, "center"),
}
for p, s, n in pts:
    ox, oy, ha = OFFSETS.get(n, (0, 10, "center"))
    ax.annotate(n, (p, s), xytext=(ox, oy), textcoords="offset points",
                ha=ha, fontsize=9, color="#3a4750")

# Our model, highlighted.
ax.scatter([OURS[0]], [OURS[1]], s=320, c="#e8543f", edgecolors="#9b2d1c",
           linewidths=2, zorder=5, marker="*")
ax.annotate("legal-colbert (ours)\n149M params",
            (OURS[0], OURS[1]), xytext=(14, -34), textcoords="offset points",
            ha="left", fontsize=11, fontweight="bold", color="#c0392b",
            arrowprops=dict(arrowstyle="-", color="#c0392b", lw=1.2))

ax.set_xscale("log")
ax.set_xlabel("Model size (parameters, log scale)", fontsize=12)
ax.set_ylabel("NDCG@10  (MLEB Contractual Clause Retrieval)", fontsize=12)
ax.set_title("Legal clause retrieval: size vs accuracy — open-source models",
             fontsize=15, fontweight="bold", pad=14)
ax.set_xticks([1e8, 3e8, 1e9, 3e9, 1e10])
ax.set_xticklabels(["100M", "300M", "1B", "3B", "10B"])
ax.set_ylim(45, 92)
ax.grid(True, which="major", ls="--", alpha=0.35, zorder=0)
ax.text(0.99, 0.02,
        "Source: MLEB official results (isaacus-dev/mleb). Ours measured with the same protocol.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#8a939b")
fig.tight_layout()
fig.savefig("output/clause_size_vs_ndcg.png", facecolor="white")
print("saved output/clause_size_vs_ndcg.png")
