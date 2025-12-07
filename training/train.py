"""
Training script for fine-tuning translation models.
Shows performance improvement before and after training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from src.evaluation.evaluator import TranslationEvaluator
from src.utils.dataset_loader import load_wmt19, load_flores, EXAMPLE_CUSTOM_SAMPLES
import torch
from typing import List, Tuple
import argparse
import json


def prepare_dataset(source_texts: List[str], target_texts: List[str], tokenizer, max_length: int = 128):
    """
    Prepare dataset for training.
    
    Args:
        source_texts: Source language texts
        target_texts: Target language texts
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
    
    Returns:
        Dataset ready for training
    """
    def tokenize_function(examples):
        # Tokenize source texts
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        
        # Tokenize target texts
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=max_length,
                truncation=True,
                padding="max_length"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    dataset_dict = {
        "source": source_texts,
        "target": target_texts
    }
    dataset = Dataset.from_dict(dataset_dict)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset


def evaluate_model(model, tokenizer, source_texts: List[str], target_texts: List[str], device: str):
    """
    Evaluate a model on test data.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        source_texts: Source texts
        target_texts: Reference translations
        device: Device to run on
    
    Returns:
        Dictionary of evaluation scores
    """
    if not source_texts or not target_texts:
        return {"bleu": 0.0, "rougeL": 0.0, "meteor": 0.0, "chrf": 0.0}
    
    model.eval()
    predictions = []
    
    for text in source_texts:
        if not text or not text.strip():
            predictions.append("")
            continue
            
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred_text)
    
    # Filter out empty predictions/references
    valid_pairs = [(p, t) for p, t in zip(predictions, target_texts) if p and t]
    if not valid_pairs:
        return {"bleu": 0.0, "rougeL": 0.0, "meteor": 0.0, "chrf": 0.0}
    
    predictions, target_texts = zip(*valid_pairs)
    predictions = list(predictions)
    target_texts = list(target_texts)
    
    evaluator = TranslationEvaluator()
    scores = evaluator.evaluate_single(predictions, target_texts)
    
    return scores


def train_model(
    model_name: str,
    train_source: List[str],
    train_target: List[str],
    eval_source: List[str],
    eval_target: List[str],
    output_dir: str = "./models/fine-tuned",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5
):
    """
    Fine-tune a translation model.
    
    Args:
        model_name: Base model name
        train_source: Training source texts
        train_target: Training target texts
        eval_source: Evaluation source texts
        eval_target: Evaluation target texts
        output_dir: Directory to save model
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(train_source, train_target, tokenizer)
    eval_dataset = prepare_dataset(eval_source, eval_target, tokenizer)
    
    # Evaluate before training
    print("\nEvaluating model BEFORE training...")
    before_scores = evaluate_model(model, tokenizer, eval_source[:50], eval_target[:50], device)
    print("Before training scores:")
    for metric, score in before_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    # Handle different transformers versions (eval_strategy vs evaluation_strategy)
    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": f"{output_dir}/logs",
        "logging_steps": 10,
        "load_best_model_at_end": True,
        "push_to_hub": False,
    }
    
    # Check which parameter name is supported
    import inspect
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        training_args_dict["eval_strategy"] = "epoch"
        training_args_dict["save_strategy"] = "epoch"
    elif "evaluation_strategy" in sig.parameters:
        training_args_dict["evaluation_strategy"] = "epoch"
        training_args_dict["save_strategy"] = "epoch"
    else:
        # Fallback: don't set evaluation strategy
        training_args_dict["save_strategy"] = "epoch"
    
    training_args = Seq2SeqTrainingArguments(**training_args_dict)
    
    # Trainer
    # Handle potential accelerate version issues
    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    except TypeError as e:
        if "dispatch_batches" in str(e) or "Accelerator" in str(e):
            print("\n⚠️  Version compatibility issue detected.")
            print("   Please update accelerate: pip install --upgrade accelerate")
            print("   Or install compatible version: pip install accelerate==0.20.0")
            raise Exception("Accelerate version incompatible. Please update: pip install --upgrade accelerate")
        else:
            raise
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate after training
    print("\nEvaluating model AFTER training...")
    after_scores = evaluate_model(model, tokenizer, eval_source[:50], eval_target[:50], device)
    print("After training scores:")
    for metric, score in after_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    # Compare results
    print("\n" + "="*80)
    print("BEFORE vs AFTER TRAINING COMPARISON")
    print("="*80)
    comparison = {}
    for metric in before_scores.keys():
        before = before_scores[metric]
        after = after_scores[metric]
        improvement = after - before
        improvement_pct = (improvement / before * 100) if before > 0 else 0
        comparison[metric] = {
            "before": before,
            "after": after,
            "improvement": improvement,
            "improvement_pct": improvement_pct
        }
        print(f"{metric}:")
        print(f"  Before: {before:.4f}")
        print(f"  After:  {after:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    # Save comparison
    os.makedirs("evaluation/results", exist_ok=True)
    with open("evaluation/results/training_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to evaluation/results/training_comparison.json")
    
    return model, tokenizer, comparison


def main():
    parser = argparse.ArgumentParser(description="Train a translation model")
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-en-de",
                       help="Base model to fine-tune")
    parser.add_argument("--dataset", default="custom", choices=["wmt19", "flores", "opus_books", "custom"],
                       help="Dataset to use")
    parser.add_argument("--lang-pair", default="de-en",
                       help="Language pair")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--output-dir", default="./models/fine-tuned",
                       help="Output directory for fine-tuned model")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "wmt19":
        train_source, train_target = load_wmt19(args.lang_pair, split="train")
        eval_source, eval_target = load_wmt19(args.lang_pair, split="validation")
    elif args.dataset == "flores":
        train_source, train_target = load_flores(args.lang_pair, split="dev")
        eval_source, eval_target = load_flores(args.lang_pair, split="devtest")
    elif args.dataset == "opus_books":
        from src.utils.dataset_loader import load_opus_books
        train_source, train_target = load_opus_books(args.lang_pair, split="train")
        eval_source, eval_target = load_opus_books(args.lang_pair, split="validation")
    else:  # custom
        custom_data = EXAMPLE_CUSTOM_SAMPLES
        train_source = custom_data["source"][:3]
        train_target = custom_data["target"][:3]
        eval_source = custom_data["source"][3:]
        eval_target = custom_data["target"][3:]
    
    # Limit dataset size for demo (remove in production)
    train_source = train_source[:100] if len(train_source) > 100 else train_source
    train_target = train_target[:100] if len(train_target) > 100 else train_target
    
    print(f"Training samples: {len(train_source)}")
    print(f"Evaluation samples: {len(eval_source)}")
    
    # Train
    train_model(
        model_name=args.model,
        train_source=train_source,
        train_target=train_target,
        eval_source=eval_source,
        eval_target=eval_target,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )


if __name__ == "__main__":
    main()

