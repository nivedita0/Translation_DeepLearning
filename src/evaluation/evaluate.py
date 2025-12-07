"""
Evaluation module for translation models.
Implements multiple evaluation metrics: BLEU, ROUGE, METEOR, chrF++, etc.
"""

from typing import List, Dict, Optional
from sacrebleu import BLEU, CHRF
import pandas as pd
import numpy as np
import sys
import os
import importlib.util

# Force import of evaluate library from site-packages, not local file
# The issue: local file src/evaluation/evaluate.py shadows the evaluate library
_evaluate_module = None

# Remove evaluate from sys.modules if it's the wrong one (local file)
if 'evaluate' in sys.modules:
    mod = sys.modules['evaluate']
    # Check if it's the local file (won't have 'load' attribute)
    if not hasattr(mod, 'load'):
        del sys.modules['evaluate']

# Try to import evaluate library
try:
    import evaluate
    # Verify it's the real library (has 'load' method)
    if hasattr(evaluate, 'load'):
        _evaluate_module = evaluate
    else:
        # It's the wrong module, try to find the real one
        raise AttributeError("Wrong evaluate module")
except (ImportError, AttributeError):
    # Use importlib to find the actual package
    import importlib
    import site
    
    # Find evaluate in site-packages
    for site_dir in site.getsitepackages():
        evaluate_init = os.path.join(site_dir, 'evaluate', '__init__.py')
        if os.path.exists(evaluate_init):
            spec = importlib.util.spec_from_file_location("hf_evaluate_lib", evaluate_init)
            if spec and spec.loader:
                _evaluate_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_evaluate_module)
                if hasattr(_evaluate_module, 'load'):
                    break
                else:
                    _evaluate_module = None

if _evaluate_module is None or not hasattr(_evaluate_module, 'load'):
    raise ImportError(
        "ERROR: Could not import 'evaluate' library.\n"
        "The file 'src/evaluation/evaluate.py' is shadowing the evaluate library.\n\n"
        "QUICK FIX: Rename this file to avoid conflict:\n"
        "  Option 1: Rename src/evaluation/evaluate.py to src/evaluation/evaluator.py\n"
        "  Option 2: Install evaluate library: pip install evaluate\n"
        "  Option 3: Use absolute import path"
    )


class TranslationEvaluator:
    """Evaluator for translation models using multiple metrics."""
    
    def __init__(self):
        """Initialize evaluation metrics."""
        # Load metrics from Hugging Face evaluate
        self.bleu_metric = _evaluate_module.load("bleu")
        self.rouge_metric = _evaluate_module.load("rouge")
        self.meteor_metric = _evaluate_module.load("meteor")
        self.chrf_metric = _evaluate_module.load("chrf")
        
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
        if metrics is None:
            metrics = ["bleu", "rouge", "meteor", "chrf", "sacrebleu", "sacrechrf"]
        
        results = {}
        
        # BLEU score
        if "bleu" in metrics:
            bleu_scores = self.bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            results["bleu"] = bleu_scores["bleu"]
        
        # ROUGE scores
        if "rouge" in metrics:
            rouge_scores = self.rouge_metric.compute(
                predictions=predictions,
                references=references
            )
            results["rouge1"] = rouge_scores["rouge1"]
            results["rouge2"] = rouge_scores["rouge2"]
            results["rougeL"] = rouge_scores["rougeL"]
        
        # METEOR score
        if "meteor" in metrics:
            meteor_scores = self.meteor_metric.compute(
                predictions=predictions,
                references=references
            )
            results["meteor"] = meteor_scores["meteor"]
        
        # chrF++ score
        if "chrf" in metrics:
            chrf_scores = self.chrf_metric.compute(
                predictions=predictions,
                references=references
            )
            results["chrf"] = chrf_scores["score"]
        
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


