"""
Evaluation module for translation models.
Implements multiple evaluation metrics: BLEU, ROUGE, METEOR, chrF++, etc.
"""

from typing import List, Dict, Optional
import evaluate  # Now safe to import - file renamed to evaluator.py
from sacrebleu import BLEU, CHRF
import pandas as pd
import numpy as np


class TranslationEvaluator:
    """Evaluator for translation models using multiple metrics."""
    
    def __init__(self):
        """Initialize evaluation metrics."""
        # Load metrics from Hugging Face evaluate
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        self.meteor_metric = evaluate.load("meteor")
        self.chrf_metric = evaluate.load("chrf")
        
        # SacreBLEU for additional metrics
        self.sacrebleu = BLEU()
        self.sacrechrf = CHRF()
    
    def evaluate_single(
        self,
        predictions: List[str],
        references: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate translations using multiple metrics.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            metrics: List of metrics to compute (None = all)
        
        Returns:
            Dictionary of metric scores
        """
        # Validate inputs
        if not predictions or not references:
            return {"bleu": 0.0, "rougeL": 0.0, "meteor": 0.0, "chrf": 0.0}
        
        if len(predictions) != len(references):
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
        
        if metrics is None:
            metrics = ["bleu", "rouge", "meteor", "chrf", "sacrebleu", "sacrechrf"]
        
        results = {}
        
        # BLEU score
        if "bleu" in metrics:
            try:
                bleu_scores = self.bleu_metric.compute(
                    predictions=predictions,
                    references=[[ref] for ref in references]
                )
                results["bleu"] = bleu_scores.get("bleu", 0.0)
            except Exception as e:
                print(f"BLEU calculation error: {e}")
                results["bleu"] = 0.0
        
        # ROUGE scores
        if "rouge" in metrics:
            try:
                rouge_scores = self.rouge_metric.compute(
                    predictions=predictions,
                    references=references
                )
                results["rouge1"] = rouge_scores.get("rouge1", 0.0)
                results["rouge2"] = rouge_scores.get("rouge2", 0.0)
                results["rougeL"] = rouge_scores.get("rougeL", 0.0)
            except Exception as e:
                print(f"ROUGE calculation error: {e}")
                results["rouge1"] = 0.0
                results["rouge2"] = 0.0
                results["rougeL"] = 0.0
        
        # METEOR score
        if "meteor" in metrics:
            try:
                meteor_scores = self.meteor_metric.compute(
                    predictions=predictions,
                    references=references
                )
                results["meteor"] = meteor_scores.get("meteor", 0.0)
            except Exception as e:
                print(f"METEOR calculation error: {e}")
                results["meteor"] = 0.0
        
        # chrF++ score
        if "chrf" in metrics:
            try:
                chrf_scores = self.chrf_metric.compute(
                    predictions=predictions,
                    references=references
                )
                results["chrf"] = chrf_scores.get("score", 0.0)
            except Exception as e:
                print(f"chrF calculation error: {e}")
                results["chrf"] = 0.0
        
        # SacreBLEU
        if "sacrebleu" in metrics:
            sacrebleu_score = self.sacrebleu.corpus_score(predictions, [references])
            results["sacrebleu"] = sacrebleu_score.score
        
        # SacreCHRF
        if "sacrechrf" in metrics:
            sacrechrf_score = self.sacrechrf.corpus_score(predictions, [references])
            results["sacrechrf"] = sacrechrf_score.score
        
        return results
    
    def compare_models(
        self,
        model_results: Dict[str, List[str]],
        references: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same dataset.
        
        Args:
            model_results: Dictionary mapping model names to their predictions
            references: List of reference translations
            metrics: List of metrics to compute
        
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model_name, predictions in model_results.items():
            scores = self.evaluate_single(predictions, references, metrics)
            scores["model"] = model_name
            comparison_results.append(scores)
        
        df = pd.DataFrame(comparison_results)
        return df


def format_comparison_table(df: pd.DataFrame) -> str:
    """
    Format comparison results as a markdown table.
    
    Args:
        df: DataFrame with comparison results
    
    Returns:
        Formatted markdown table string
    """
    return df.to_markdown(index=False)


