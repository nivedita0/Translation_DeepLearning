"""
Script to compare multiple translation models on a dataset.
Generates comparison tables and figures.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.model_loader import load_model, list_available_models
from src.evaluation.evaluator import TranslationEvaluator, format_comparison_table
from src.utils.dataset_loader import (
    load_wmt19, load_flores, load_opus_ted, 
    load_custom_dataset, EXAMPLE_CUSTOM_SAMPLES
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import argparse


def compare_models_on_dataset(
    model_names: List[str],
    dataset_name: str = "wmt19",
    language_pair: str = "de-en",
    max_samples: int = 100
):
    """
    Compare multiple models on a dataset.
    
    Args:
        model_names: List of model keys to compare
        dataset_name: Name of dataset ("wmt19", "flores", "opus_ted", "custom")
        language_pair: Language pair for the dataset
        max_samples: Maximum number of samples to evaluate
    """
    print(f"Comparing models: {model_names}")
    print(f"Dataset: {dataset_name}, Language pair: {language_pair}")
    
    # Load dataset
    if dataset_name == "wmt19":
        source_texts, target_texts = load_wmt19(language_pair)
    elif dataset_name == "flores":
        source_texts, target_texts = load_flores(language_pair)
    elif dataset_name == "opus_ted":
        source_texts, target_texts = load_opus_ted(language_pair)
    elif dataset_name == "opus_books":
        from src.utils.dataset_loader import load_opus_books
        source_texts, target_texts = load_opus_books(language_pair, split="test")
    elif dataset_name == "custom":
        custom_data = EXAMPLE_CUSTOM_SAMPLES
        source_texts = custom_data["source"]
        target_texts = custom_data["target"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Limit samples
    source_texts = source_texts[:max_samples]
    target_texts = target_texts[:max_samples]
    
    print(f"Loaded {len(source_texts)} samples")
    
    # Load models and generate translations
    model_results = {}
    models = {}
    
    for model_key in model_names:
        print(f"\nLoading model: {model_key}")
        try:
            model = load_model(model_key)
            models[model_key] = model
            
            print(f"Translating with {model_key}...")
            predictions = []
            for i, text in enumerate(source_texts):
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(source_texts)}")
                pred = model.translate(text)
                predictions.append(pred)
            
            model_results[model_key] = predictions
            print(f"Completed translations for {model_key}")
        except Exception as e:
            print(f"Error loading model {model_key}: {e}")
            continue
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluator = TranslationEvaluator()
    comparison_df = evaluator.compare_models(
        model_results,
        target_texts,
        metrics=["bleu", "rouge1", "rouge2", "rougeL", "meteor", "chrf"]
    )
    
    # Display results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(format_comparison_table(comparison_df))
    
    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    comparison_df.to_csv("evaluation/results/model_comparison.csv", index=False)
    print(f"\nResults saved to evaluation/results/model_comparison.csv")
    
    # Create visualization
    create_comparison_plot(comparison_df, "evaluation/results/model_comparison.png")
    print("Visualization saved to evaluation/results/model_comparison.png")
    
    return comparison_df, model_results


def create_comparison_plot(df: pd.DataFrame, output_path: str):
    """Create a bar plot comparing models."""
    # Select numeric columns (exclude 'model')
    metric_cols = [col for col in df.columns if col != "model"]
    
    # Create subplots
    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metric_cols):
        ax = axes[idx]
        df.plot(x="model", y=metric, kind="bar", ax=ax, legend=False)
        ax.set_title(f"{metric.upper()} Score")
        ax.set_ylabel("Score")
        ax.set_xlabel("Model")
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare translation models")
    parser.add_argument("--models", nargs="+", default=["opus-mt-en-de", "marian-en-de"],
                       help="Model keys to compare")
    parser.add_argument("--dataset", default="custom", choices=["wmt19", "flores", "opus_ted", "opus_books", "custom"],
                       help="Dataset to use")
    parser.add_argument("--lang-pair", default="de-en",
                       help="Language pair")
    parser.add_argument("--max-samples", type=int, default=50,
                       help="Maximum number of samples")
    
    args = parser.parse_args()
    
    compare_models_on_dataset(
        model_names=args.models,
        dataset_name=args.dataset,
        language_pair=args.lang_pair,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

