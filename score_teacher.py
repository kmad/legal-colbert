"""Score distillation candidates with a cross-encoder teacher (P1).

Adds a `scores` column to the train dataset: for each (query, candidate-doc)
pair, the cross-encoder relevance score. Scores are flattened, predicted in
batches (GPU-friendly), then regrouped per train row. Run on the pod (GPU).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk
from sentence_transformers import CrossEncoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_p1_distill")
    ap.add_argument("--teacher", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    data = Path(args.data_dir)
    queries = load_from_disk(str(data / "queries"))
    documents = load_from_disk(str(data / "documents"))
    train = load_from_disk(str(data / "train"))

    qmap = {r["query_id"]: r["text"] for r in queries}
    dmap = {r["document_id"]: r["text"] for r in documents}

    pairs, spans = [], []
    for row in train:
        q = qmap[row["query_id"]]
        ids = row["document_ids"]
        spans.append(len(ids))
        pairs.extend((q, dmap[d]) for d in ids)
    print(f"scoring {len(pairs)} pairs over {len(train)} rows with {args.teacher}")

    ce = CrossEncoder(args.teacher, max_length=args.max_length)
    flat = ce.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)

    scores, cursor = [], 0
    for n in spans:
        scores.append([float(x) for x in flat[cursor : cursor + n]])
        cursor += n

    train = train.add_column("scores", scores)
    out = data / "train_scored"
    train.save_to_disk(str(out))
    print(f"wrote {out} ({len(train)} rows)")


if __name__ == "__main__":
    main()
