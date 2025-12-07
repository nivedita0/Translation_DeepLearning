"""
Model loader for translation models.
Supports loading multiple translation models for comparison.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import Dict, List, Optional
import torch


class TranslationModel:
    """Wrapper class for translation models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize a translation model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name} on {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Create translation pipeline
        self.translator = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
    
    def translate(self, text: str, source_lang: str = None, target_lang: str = None) -> str:
        """
        Translate a single text.
        
        Args:
            text: Source text to translate
            source_lang: Source language code (optional)
            target_lang: Target language code (optional)
        
        Returns:
            Translated text
        """
        # Some models need language codes in the task name
        if source_lang and target_lang:
            task = f"translation_{source_lang}_to_{target_lang}"
        else:
            task = "translation"
        
        result = self.translator(text)
        return result[0]['translation_text']
    
    def translate_batch(self, texts: List[str], source_lang: str = None, target_lang: str = None) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of source texts
            source_lang: Source language code (optional)
            target_lang: Target language code (optional)
        
        Returns:
            List of translated texts
        """
        return [self.translate(text, source_lang, target_lang) for text in texts]


# Recommended models for comparison
RECOMMENDED_MODELS = {
    "opus-mt-en-de": "Helsinki-NLP/opus-mt-en-de",
    "opus-mt-en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "opus-mt-en-es": "Helsinki-NLP/opus-mt-en-es",
    "mBART50": "facebook/mbart-large-50-many-to-many-mmt",
    "mT5": "google/mt5-base",
    "marian-en-de": "Helsinki-NLP/opus-mt-en-de",  # Alternative to opus-mt
}


def load_model(model_key: str, device: Optional[str] = None) -> TranslationModel:
    """
    Load a model by key.
    
    Args:
        model_key: Key from RECOMMENDED_MODELS or full model name
        device: Device to run on
    
    Returns:
        TranslationModel instance
    """
    model_name = RECOMMENDED_MODELS.get(model_key, model_key)
    return TranslationModel(model_name, device)


def list_available_models() -> Dict[str, str]:
    """List all available recommended models."""
    return RECOMMENDED_MODELS.copy()


