"""
Visualization utilities for translation evaluation results.
Creates comparison figures and tables.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os


def plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    output_path: str = "evaluation/results/comparison.png"
):
    """
    Create a bar plot comparing models across metrics.
    
    Args:
        comparison_df: DataFrame with model comparison results
        metrics: List of metrics to plot (None = all numeric columns)
        output_path: Path to save the plot
    """
    if metrics is None:
        metrics = [col for col in comparison_df.columns if col != "model" and pd.api.types.is_numeric_dtype(comparison_df[col])]
    
    n_metrics = len(metrics)
    if n_metrics == 0:
        print("No metrics to plot")
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        comparison_df.plot(x="model", y=metric, kind="bar", ax=ax, legend=False, color='steelblue')
        ax.set_title(f"{metric.upper()} Score", fontsize=12, fontweight='bold')
        ax.set_ylabel("Score", fontsize=10)
        ax.set_xlabel("Model", fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_training_comparison(
    before_scores: Dict[str, float],
    after_scores: Dict[str, float],
    output_path: str = "evaluation/results/training_comparison.png"
):
    """
    Create a comparison plot showing before/after training performance.
    
    Args:
        before_scores: Dictionary of scores before training
        after_scores: Dictionary of scores after training
        output_path: Path to save the plot
    """
    metrics = list(before_scores.keys())
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = ['Before Training', 'After Training']
        y = [before_scores[metric], after_scores[metric]]
        bars = ax.bar(x, y, color=['lightcoral', 'lightgreen'])
        ax.set_title(f"{metric.upper()} Score", fontsize=12, fontweight='bold')
        ax.set_ylabel("Score", fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
        
        # Add improvement annotation
        improvement = after_scores[metric] - before_scores[metric]
        improvement_pct = (improvement / before_scores[metric] * 100) if before_scores[metric] > 0 else 0
        ax.text(0.5, max(y) * 1.1, f'Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)',
               ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training comparison plot saved to {output_path}")


def create_heatmap(
    comparison_df: pd.DataFrame,
    output_path: str = "evaluation/results/metric_heatmap.png"
):
    """
    Create a heatmap showing all metrics for all models.
    
    Args:
        comparison_df: DataFrame with model comparison results
        output_path: Path to save the heatmap
    """
    # Prepare data for heatmap
    metrics = [col for col in comparison_df.columns if col != "model" and pd.api.types.is_numeric_dtype(comparison_df[col])]
    heatmap_data = comparison_df.set_index("model")[metrics]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
    plt.title("Model Comparison Heatmap", fontsize=14, fontweight='bold')
    plt.xlabel("Metrics", fontsize=11)
    plt.ylabel("Models", fontsize=11)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")


