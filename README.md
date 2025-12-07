# Translation with Language Models

A comprehensive machine translation system that leverages language models for high-quality translations, with model comparison, evaluation metrics, and an interactive user interface.

## Project Features

1. **Multiple Model Comparison**: Compare at least 2 translation models using standard evaluation metrics
2. **Dataset Evaluation**: Evaluate models on standard datasets (WMT-19, OPUS, etc.)
3. **Custom Test Samples**: Test models with your own provided test samples
4. **Interactive UI**: Web interface built with Gradio for easy translation
5. **Training Pipeline**: Fine-tune models and compare performance before/after training

## Dataset

### 2. OPUS Dataset
- **Description**: Large collection of parallel corpora
- **Language Pairs**: 100+ language pairs
- **Download**: Available via OPUS website or Hugging Face
- **Use Case**: Training and evaluation across multiple domains


## Evaluation Metrics

- **BLEU**: N-gram precision-based metric
- **ROUGE**: Recall-oriented evaluation
- **METEOR**: Explicit ordering consideration
- **chrF++**: Character-level F-score
- **COMET**: Crosslingual optimized metric
- **TER**: Translation Error Rate

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Model Comparison
```bash
python src/evaluation/compare_models.py
```

### Running the UI
```bash
python ui/app.py
```

### Training a Model
```bash
python training/train.py
```

The .ipynb file has all the details including the UI which uses the model. The python runs the whole UI on a browser page.