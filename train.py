"""
Fine-tune a ColBERT model for legal retrieval using PyLate.

Base model: lightonai/GTE-ModernColBERT-v1
Training data: CUAD + ACORD with BM25 hard negatives (prepared by prepare_data.py)
"""

import torch
from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils


def train(
    data_dir: str = "data",
    model_name: str = "lightonai/GTE-ModernColBERT-v1",
    output_dir: str = "output/legal-colbert",
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 3e-6,
    temperature: float = 0.02,
    use_bf16: bool = False,
    use_fp16: bool = True,
):
    """Fine-tune a ColBERT model on legal triplet data.

    Args:
        data_dir: path to prepared data directory containing train/ and test/ splits
        model_name: base ColBERT model to fine-tune
        output_dir: where to save checkpoints and final model
        batch_size: per-device training batch size
        num_epochs: number of training epochs
        learning_rate: peak learning rate
        temperature: contrastive loss temperature (lower = harder negatives matter more)
        use_bf16: enable bfloat16 training (Ampere+ GPUs)
        use_fp16: enable float16 training
    """
    # Load prepared triplet data (pre-split by prepare_data.py)
    print(f"Loading triplet data from {data_dir}...")
    train_dataset = load_from_disk(f"{data_dir}/train")
    eval_dataset = load_from_disk(f"{data_dir}/test")
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Initialize ColBERT model
    print(f"Loading model: {model_name}")
    model = models.ColBERT(model_name_or_path=model_name)

    # Contrastive loss with low temperature for hard negative training
    train_loss = losses.Contrastive(model=model, temperature=temperature)

    # Triplet evaluator on held-out set
    dev_evaluator = evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
    )

    # Training arguments
    run_name = "legal-colbert-cuad-acord"
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=50,
        run_name=run_name,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Initialize trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=dev_evaluator,
        data_collator=utils.ColBERTCollator(model.tokenize),
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    final_path = f"{output_dir}/final"
    model.save_pretrained(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune legal ColBERT model")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-name", default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument("--output-dir", default="output/legal-colbert")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parsed = parser.parse_args()

    train(
        data_dir=parsed.data_dir,
        model_name=parsed.model_name,
        output_dir=parsed.output_dir,
        batch_size=parsed.batch_size,
        num_epochs=parsed.num_epochs,
        learning_rate=parsed.learning_rate,
        temperature=parsed.temperature,
        use_bf16=parsed.bf16,
        use_fp16=not parsed.no_fp16,
    )
