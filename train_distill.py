"""Train a ColBERT clause retriever by distilling cross-encoder scores (P1)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_from_disk
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

from pylate import losses, models, utils


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_p1_distill")
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--output-dir", default="output/legal-colbert-p1-distill")
    ap.add_argument("--run-name", default="legal-colbert-p1-distill")
    ap.add_argument("--num-epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=2e-6)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--n-ways", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")

    data = Path(args.data_dir)
    queries = load_from_disk(str(data / "queries"))
    documents = load_from_disk(str(data / "documents"))
    train = load_from_disk(str(data / "train_scored"))
    print(f"KD train rows: {len(train)}; queries {len(queries)}; documents {len(documents)}")

    train.set_transform(
        utils.KDProcessing(queries=queries, documents=documents, n_ways=args.n_ways).transform
    )

    model = models.ColBERT(model_name_or_path=args.model_name)
    loss = losses.Distillation(model=model)

    st_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=50,
        save_strategy="no",
        run_name=args.run_name,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=st_args,
        train_dataset=train,
        loss=loss,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )
    trainer.train()

    final = Path(args.output_dir) / "final"
    model.save_pretrained(str(final))
    print(f"saved {final}")
    with open(Path(args.output_dir) / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
