"""
Dataset loader utilities for translation datasets.
Supports WMT-19, OPUS, FLORES-200, and custom datasets.
"""

from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import pandas as pd


def load_wmt19(language_pair: str = "de-en", split: str = "test") -> Tuple[List[str], List[str]]:
    """
    Load WMT-19 dataset.
    
    Args:
        language_pair: Language pair (e.g., "de-en", "fr-en")
        split: Dataset split ("test", "validation", etc.)
    
    Returns:
        Tuple of (source_texts, target_texts)
    """
    try:
        dataset = load_dataset("wmt19", language_pair, split=split)
        source_texts = [item["translation"][language_pair.split("-")[0]] for item in dataset]
        target_texts = [item["translation"][language_pair.split("-")[1]] for item in dataset]
        return source_texts, target_texts
    except Exception as e:
        print(f"Error loading WMT-19: {e}")
        print("Trying alternative loading method...")
        # Alternative: try loading from different source
        return [], []


def load_flores(language_pair: str = "eng_Latn-deu_Latn", split: str = "devtest") -> Tuple[List[str], List[str]]:
    """
    Load FLORES-200 dataset.
    
    Args:
        language_pair: Language pair (e.g., "eng_Latn-deu_Latn")
        split: Dataset split ("devtest", "dev", etc.)
    
    Returns:
        Tuple of (source_texts, target_texts)
    """
    try:
        lang1, lang2 = language_pair.split("-")
        dataset = load_dataset("facebook/flores", language_pair, split=split)
        source_texts = dataset[lang1]
        target_texts = dataset[lang2]
        return source_texts, target_texts
    except Exception as e:
        print(f"Error loading FLORES: {e}")
        return [], []


def load_opus_ted(language_pair: str = "en-fr", split: str = "test") -> Tuple[List[str], List[str]]:
    """
    Load OPUS TED talks dataset.
    
    Args:
        language_pair: Language pair (e.g., "en-fr")
        split: Dataset split
    
    Returns:
        Tuple of (source_texts, target_texts)
    """
    try:
        dataset = load_dataset("opus_ted", language_pair, split=split)
        source_texts = dataset["source"]
        target_texts = dataset["target"]
        return source_texts, target_texts
    except Exception as e:
        print(f"Error loading OPUS TED: {e}")
        return [], []


def load_opus_books(language_pair: str = "en-fr", split: str = "train", max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load OPUS Books dataset - parallel book translations.
    Great for training because it has large amounts of high-quality data.
    
    Args:
        language_pair: Language pair (e.g., "en-fr", "en-de", "en-es")
        split: Dataset split ("train", "test", "validation")
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        Tuple of (source_texts, target_texts)
    
    Example:
        >>> source, target = load_opus_books("en-fr", split="train", max_samples=1000)
        >>> print(f"Loaded {len(source)} sentence pairs")
    """
    try:
        print(f"Loading OPUS Books dataset: {language_pair}, split: {split}")
        print("Note: This may take a few minutes on first download...")
        
        # Load dataset from Hugging Face
        dataset = load_dataset("opus_books", language_pair, split=split)
        
        # Extract source and target texts
        # OPUS books uses "source" and "target" columns
        source_texts = dataset["source"]
        target_texts = dataset["target"]
        
        # Limit samples if requested
        if max_samples is not None:
            source_texts = source_texts[:max_samples]
            target_texts = target_texts[:max_samples]
        
        print(f"Successfully loaded {len(source_texts)} sentence pairs")
        return list(source_texts), list(target_texts)
        
    except Exception as e:
        print(f"Error loading OPUS Books: {e}")
        print(f"Available language pairs include: en-fr, en-de, en-es, en-it, etc.")
        print("Try checking: https://huggingface.co/datasets/opus_books")
        return [], []


def load_custom_dataset(file_path: str, source_col: str = "source", target_col: str = "target") -> Tuple[List[str], List[str]]:
    """
    Load custom dataset from CSV or JSON file.
    
    Args:
        file_path: Path to dataset file
        source_col: Name of source column
        target_col: Name of target column
    
    Returns:
        Tuple of (source_texts, target_texts)
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")
    
    source_texts = df[source_col].tolist()
    target_texts = df[target_col].tolist()
    return source_texts, target_texts


def create_custom_samples(source_texts: List[str], target_texts: List[str]) -> Dict[str, List[str]]:
    """
    Create a custom test samples dictionary.
    
    Args:
        source_texts: List of source texts
        target_texts: List of target texts
    
    Returns:
        Dictionary with 'source' and 'target' keys
    """
    return {
        "source": source_texts,
        "target": target_texts
    }


# Example custom test samples
EXAMPLE_CUSTOM_SAMPLES = {
    "source": [
        "Hello, how are you today?",
        "The weather is beautiful today.",
        "I love machine learning and artificial intelligence.",
        "This is a test sentence for translation evaluation.",
        "Natural language processing is fascinating."
    ],
    "target": [
        "Hola, ¿cómo estás hoy?",
        "El clima está hermoso hoy.",
        "Me encanta el aprendizaje automático y la inteligencia artificial.",
        "Esta es una oración de prueba para la evaluación de traducción.",
        "El procesamiento de lenguaje natural es fascinante."
    ]
}

