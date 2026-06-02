"""Contrastive control for P1: descriptive-query triplets from the same
candidate pool as the distillation set (no teacher). Isolates the effect of
distillation vs the query-style/hard-negative changes."""
import argparse, random
from pathlib import Path
from datasets import Dataset, load_from_disk
from prepare_v2_data import SEED

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data_p1_distill")
ap.add_argument("--out", default="data_p1_contrastive")
ap.add_argument("--negs-per-pos", type=int, default=6)
a = ap.parse_args()
d = Path(a.data_dir)
queries = {r["query_id"]: r["text"] for r in load_from_disk(str(d / "queries"))}
documents = {r["document_id"]: r["text"] for r in load_from_disk(str(d / "documents"))}
train = load_from_disk(str(d / "train"))
rng = random.Random(SEED)
rows = []
for r in train:
    q = queries[r["query_id"]]
    ids = r["document_ids"]
    pos = documents[ids[0]]
    for nid in ids[1 : 1 + a.negs_per_pos]:
        neg = documents[nid]
        if neg != pos:
            rows.append({"query": q, "positive": pos, "negative": neg})
rng.shuffle(rows)
out = Path(a.out)
ds = Dataset.from_list(rows)
ds.save_to_disk(str(out / "train"))
print(f"contrastive control: {len(rows)} triplets -> {out}/train")
