# Translation with Language Models

A comprehensive machine translation system that leverages language models for high-quality translations, with model comparison, evaluation metrics, and an interactive user interface.

## Project Features

1. **Multiple Model Comparison**: Compare at least 2 translation models using standard evaluation metrics
2. **Dataset Evaluation**: Evaluate models on standard datasets (WMT-19, OPUS, etc.)
3. **Custom Test Samples**: Test models with your own provided test samples
4. **Interactive UI**: Web interface built with Gradio for easy translation
5. **Training Pipeline**: Fine-tune models and compare performance before/after training

## Project Structure

```
HWTranslation/
├── datasets/              # Translation datasets
├── models/                # Model files and checkpoints
├── src/                   # Source code
│   ├── models/           # Model loading and inference
│   ├── evaluation/       # Evaluation scripts
│   └── utils/            # Utility functions
├── evaluation/            # Evaluation results and scripts
├── ui/                    # User interface
├── training/              # Training scripts
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── CHANGELOG.md          # Project changelog

```

## Recommended Datasets

### 1. WMT-19 Dataset
- **Description**: Workshop on Machine Translation 2019 dataset
- **Language Pairs**: English ↔ German, French, Spanish, Russian, etc.
- **Download**: Available via Hugging Face Datasets or WMT website
- **Use Case**: Standard benchmark for translation evaluation

### 2. OPUS Dataset
- **Description**: Large collection of parallel corpora
- **Language Pairs**: 100+ language pairs
- **Download**: Available via OPUS website or Hugging Face
- **Use Case**: Training and evaluation across multiple domains

### 3. FLORES-200
- **Description**: Meta AI's multilingual evaluation dataset
- **Language Pairs**: 200 languages
- **Download**: Hugging Face Datasets
- **Use Case**: Multilingual evaluation

### 4. TED Talks Dataset
- **Description**: Subtitles from TED talks
- **Language Pairs**: Multiple
- **Download**: OPUS or Hugging Face
- **Use Case**: Domain-specific evaluation

## Recommended Models

1. **mBART50** - Multilingual BART supporting 50 languages
2. **OPUS-MT** - Pre-trained translation models (Helsinki NLP)
3. **MarianMT** - Fast neural machine translation
4. **NLLB** - Meta's No Language Left Behind (200+ languages)
5. **mT5** - Multilingual T5 model

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

## Documentation

- **TUTORIAL_OPUS_BOOKS.md**: Complete beginner-friendly tutorial for using OPUS Books dataset
- **QUICK_REFERENCE_OPUS_BOOKS.md**: Quick reference for OPUS Books workflow
- **DATASETS.md**: Detailed dataset information and usage
- **QUICKSTART.md**: Quick start guide
- **CHANGELOG.md**: Detailed tracking of all changes and modifications

## Getting Started with OPUS Books

If you're new to deep learning and want to use OPUS Books dataset:

1. **Read the Tutorial**: Start with `TUTORIAL_OPUS_BOOKS.md` - it's designed for beginners!
2. **Run Starter Script**: `python start_opus_books_tutorial.py`
3. **Follow Steps**: The tutorial guides you through everything step-by-step

The tutorial covers:
- Understanding datasets and machine translation
- Loading and exploring OPUS Books
- Comparing models before training
- Training your own model
- Evaluating and comparing results
- Key deep learning concepts explained simply

## License

[Add your license here]

