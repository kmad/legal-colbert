"""Train legal-colbert-v2 candidates with PyLate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from datasets import load_from_disk
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

from pylate import evaluation, losses, models, utils


def train(args: argparse.Namespace) -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")

    train_dataset = load_from_disk(str(Path(args.data_dir) / "train"))
    extra_columns = [
        column
        for column in train_dataset.column_names
        if column not in {"query", "positive", "negative"}
    ]
    if extra_columns:
        train_dataset = train_dataset.remove_columns(extra_columns)
    print(f"Loaded train triplets: {len(train_dataset)} from {args.data_dir}")

    model = models.ColBERT(model_name_or_path=args.model_name)
    train_loss = losses.Contrastive(model=model, temperature=args.temperature)

    # Use a small held-out sample from training triplets only for trainer checkpoint
    # selection. Real model selection should use eval_v2.py on ACORD valid.
    eval_size = min(max(args.eval_triplets, 128), len(train_dataset) // 20)
    split = train_dataset.train_test_split(test_size=eval_size, seed=42)
    train_split = split["train"]
    eval_split = split["test"]

    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_split["query"],
        positives=eval_split["positive"],
        negatives=eval_split["negative"],
    )

    st_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        run_name=args.run_name,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=st_args,
        train_dataset=train_split,
        eval_dataset=eval_split,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )

    trainer.train()

    final_path = Path(args.output_dir) / "final"
    model.save_pretrained(str(final_path))
    print(f"Saved final model to {final_path}")

    if args.copy_to_model:
        dst = Path(args.copy_to_model)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(final_path, dst)
        print(f"Copied final model to {dst}")

    config = vars(args).copy()
    config["train_triplets"] = len(train_split)
    config["eval_triplets"] = len(eval_split)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_v2")
    parser.add_argument("--model-name", default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument("--output-dir", default="output/legal-colbert-v2")
    parser.add_argument("--copy-to-model", default="")
    parser.add_argument("--run-name", default="legal-colbert-v2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-triplets", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 or --fp16")
    train(args)


if __name__ == "__main__":
    main()
