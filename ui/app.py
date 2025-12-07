"""
Gradio UI for translation application.
Provides an interactive interface for translation with multiple models.
"""

import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.model_loader import load_model, list_available_models
from src.evaluation.evaluator import TranslationEvaluator, format_comparison_table
from src.utils.dataset_loader import load_opus_books, EXAMPLE_CUSTOM_SAMPLES
from typing import Optional
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import sys
import os


# Global variables for loaded models
loaded_models = {}
evaluator = TranslationEvaluator()

# Pre-load models for faster translation (like in notebook)
def preload_models():
    """Pre-load models at startup to avoid delay on first translation."""
    print("Pre-loading models...")
    available_models = get_available_models()
    
    # Pre-load first 2 models (most commonly used)
    models_to_preload = available_models[:2] if len(available_models) >= 2 else available_models
    
    for model_key in models_to_preload:
        if model_key not in loaded_models:
            try:
                if "Fine-tuned" in model_key:
                    # Skip fine-tuned models in preload (they may not exist)
                    continue
                print(f"  Loading {model_key}...")
                loaded_models[model_key] = load_model(model_key)
                print(f"  ✓ {model_key} loaded")
            except Exception as e:
                print(f"  ✗ Failed to load {model_key}: {e}")
    
    print("Model pre-loading complete!\n")


def get_available_models():
    """Get list of available models, including fine-tuned if available."""
    models = list_available_models()
    model_list = list(models.keys())
    
    # Check for fine-tuned models
    fine_tuned_dirs = [
        "./models/fine-tuned",
        "./models/fine-tuned-opus-books",
        "./opus_en_fr_finetuned"
    ]
    
    for model_dir in fine_tuned_dirs:
        if os.path.exists(model_dir):
            # Extract model name from directory
            if "opus" in model_dir.lower():
                model_list.append("Fine-tuned (OPUS Books)")
            else:
                model_list.append("Fine-tuned")
            break
    
    return model_list


def translate_with_evaluation(
    text: str,
    model_key: str,
    reference: Optional[str] = None,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None
) -> tuple:
    """
    Translate text and optionally evaluate if reference is provided.
    
    Args:
        text: Source text
        model_key: Model to use
        reference: Reference translation (optional, for evaluation)
        source_lang: Source language (optional)
        target_lang: Target language (optional)
    
    Returns:
        Tuple of (translation, evaluation_results)
    """
    if not text:
        return "", ""
    
    # Load model if not already loaded
    if model_key not in loaded_models:
        try:
            # Check if it's a fine-tuned model
            if "Fine-tuned" in model_key:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import torch
                
                # Try to find fine-tuned model directory
                fine_tuned_dirs = [
                    "./models/fine-tuned",
                    "./models/fine-tuned-opus-books",
                    "./opus_en_fr_finetuned"
                ]
                
                model_path = None
                for model_dir in fine_tuned_dirs:
                    if os.path.exists(model_dir):
                        model_path = model_dir
                        break
                
                if model_path:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
                    model.eval()
                    
                    # Create a wrapper for fine-tuned model
                    class FineTunedModel:
                        def __init__(self, model, tokenizer, device):
                            self.model = model
                            self.tokenizer = tokenizer
                            self.device = device
                        
                        def translate(self, text, source_lang=None, target_lang=None):
                            encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                            with torch.no_grad():
                                outputs = self.model.generate(**encoded, max_length=128)
                            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    loaded_models[model_key] = FineTunedModel(model, tokenizer, device)
                else:
                    return "", "Fine-tuned model not found. Please train a model first."
            else:
                loaded_models[model_key] = load_model(model_key)
        except Exception as e:
            return "", f"Error loading model: {str(e)}"
    
    model = loaded_models[model_key]
    
    try:
        translation = model.translate(text, source_lang, target_lang)
    except Exception as e:
        return "", f"Error during translation: {str(e)}"
    
    # Evaluate if reference provided
    eval_result = ""
    if reference and reference.strip():
        try:
            scores = evaluator.evaluate_single([translation], [reference])
            eval_result = f"""
**BLEU Score:** {scores.get('bleu', 'N/A'):.4f}
**ROUGE-L:** {scores.get('rougeL', 'N/A'):.4f}
**METEOR:** {scores.get('meteor', 'N/A'):.4f}
**chrF:** {scores.get('chrf', 'N/A'):.4f}

**Translation:** {translation}
**Reference:** {reference}
"""
        except Exception as e:
            eval_result = f"Evaluation error: {str(e)}"
    else:
        eval_result = f"**Translation:** {translation}\n\n*Provide a reference translation to see evaluation metrics*"
    
    return translation, eval_result


def compare_translations(
    text: str,
    model1_key: str,
    model2_key: str,
    reference: Optional[str] = None
) -> tuple:
    """
    Compare translations from two models.
    
    Args:
        text: Source text
        model1_key: First model
        model2_key: Second model
        reference: Reference translation (optional, for evaluation)
    
    Returns:
        Tuple of (translation1, translation2, evaluation_results)
    """
    if not text:
        return "", "", ""
    
    # Load models if needed
    for model_key in [model1_key, model2_key]:
        if model_key not in loaded_models:
            try:
                # Check if it's a fine-tuned model
                if "Fine-tuned" in model_key:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    import torch
                    
                    fine_tuned_dirs = [
                        "./models/fine-tuned",
                        "./models/fine-tuned-opus-books",
                        "./opus_en_fr_finetuned"
                    ]
                    
                    model_path = None
                    for model_dir in fine_tuned_dirs:
                        if os.path.exists(model_dir):
                            model_path = model_dir
                            break
                    
                    if model_path:
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
                        model.eval()
                        
                        class FineTunedModel:
                            def __init__(self, model, tokenizer, device):
                                self.model = model
                                self.tokenizer = tokenizer
                                self.device = device
                            
                            def translate(self, text, source_lang=None, target_lang=None):
                                encoded = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                                with torch.no_grad():
                                    outputs = self.model.generate(**encoded, max_length=128)
                                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        loaded_models[model_key] = FineTunedModel(model, tokenizer, device)
                    else:
                        return f"Fine-tuned model not found", "", ""
                else:
                    loaded_models[model_key] = load_model(model_key)
            except Exception as e:
                return f"Error loading {model_key}: {str(e)}", "", ""
    
    # Get translations
    try:
        translation1 = loaded_models[model1_key].translate(text)
        translation2 = loaded_models[model2_key].translate(text)
    except Exception as e:
        return f"Error: {str(e)}", "", ""
    
    # Evaluate if reference provided
    eval_results = ""
    if reference and reference.strip():
        try:
            scores1 = evaluator.evaluate_single([translation1], [reference])
            scores2 = evaluator.evaluate_single([translation2], [reference])
            
            bleu1 = scores1.get('bleu', 0)
            bleu2 = scores2.get('bleu', 0)
            winner = model1_key if bleu1 > bleu2 else model2_key
            
            eval_results = f"""
## Model Comparison Results

### {model1_key}
- **Translation:** {translation1}
- **BLEU:** {bleu1:.4f}
- **ROUGE-L:** {scores1.get('rougeL', 'N/A'):.4f}
- **METEOR:** {scores1.get('meteor', 'N/A'):.4f}

### {model2_key}
- **Translation:** {translation2}
- **BLEU:** {bleu2:.4f}
- **ROUGE-L:** {scores2.get('rougeL', 'N/A'):.4f}
- **METEOR:** {scores2.get('meteor', 'N/A'):.4f}

### Winner
**{winner}** has higher BLEU score
"""
        except Exception as e:
            eval_results = f"Evaluation error: {str(e)}"
    else:
        eval_results = f"""
## Model Comparison

### {model1_key}
{translation1}

### {model2_key}
{translation2}

*Provide a reference translation to see evaluation metrics*
"""
    
    return translation1, translation2, eval_results


def show_training_comparison():
    """Display before/after training comparison."""
    try:
        before_file = "evaluation/results/before_training_scores.json"
        after_file = "evaluation/results/after_training_scores.json"
        
        if os.path.exists(before_file) and os.path.exists(after_file):
            with open(before_file, 'r') as f:
                before_scores = json.load(f)
            with open(after_file, 'r') as f:
                after_scores = json.load(f)
            
            result = "## Training Performance Comparison\n\n"
            result += "### Before Training (Baseline)\n"
            for metric, score in before_scores.items():
                if isinstance(score, (int, float)):
                    result += f"- **{metric.upper()}:** {score:.4f}\n"
            
            result += "\n### After Training (Fine-tuned)\n"
            for metric, score in after_scores.items():
                if isinstance(score, (int, float)):
                    result += f"- **{metric.upper()}:** {score:.4f}\n"
            
            result += "\n### Improvement\n"
            for metric in before_scores.keys():
                if metric in after_scores and isinstance(before_scores[metric], (int, float)):
                    before = before_scores[metric]
                    after = after_scores[metric]
                    improvement = after - before
                    improvement_pct = (improvement / before * 100) if before > 0 else 0
                    result += f"- **{metric.upper()}:** {improvement:+.4f} ({improvement_pct:+.2f}%)\n"
        else:
            result = """
## Training Performance Comparison

### Before Training (Baseline)
*Run training to see baseline results*

### After Training (Fine-tuned)
*Run training to see fine-tuned results*

### Expected Improvement
Fine-tuning typically improves BLEU scores by 5-15% on domain-specific data.

**To generate results:**
1. Run `python training/train.py` to train a model
2. Results will be saved to `evaluation/results/`
"""
    except Exception as e:
        result = f"Error loading comparison: {str(e)}"
    
    return result


def run_dataset_comparison(model1_key, model2_key, dataset_name, language_pair, max_samples, progress=gr.Progress()):
    """Run model comparison on dataset from UI."""
    try:
        progress(0, desc="Starting comparison...")
        
        model_names = [model1_key, model2_key]
        
        progress(0.1, desc="Loading dataset...")
        
        # Load dataset
        if dataset_name == "opus_books":
            source_texts, target_texts = load_opus_books(language_pair, split="test", max_samples=max_samples)
        elif dataset_name == "custom":
            custom_data = EXAMPLE_CUSTOM_SAMPLES
            source_texts = custom_data["source"]
            target_texts = custom_data["target"]
        else:
            return f"Dataset {dataset_name} not supported in UI. Use opus_books or custom.", ""
        
        progress(0.2, desc="Loading models and translating...")
        
        # Load models and translate
        model_results = {}
        for i, model_key in enumerate(model_names):
            progress(0.3 + i*0.3, desc=f"Processing {model_key}...")
            if model_key not in loaded_models:
                loaded_models[model_key] = load_model(model_key)
            
            model = loaded_models[model_key]
            predictions = []
            for j, text in enumerate(source_texts):
                if (j + 1) % 10 == 0:
                    progress(0.3 + i*0.3 + (j/len(source_texts))*0.25, desc=f"Translating {model_key}: {j+1}/{len(source_texts)}")
                pred = model.translate(text)
                predictions.append(pred)
            
            model_results[model_key] = predictions
        
        progress(0.85, desc="Evaluating models...")
        
        # Evaluate
        comparison_results = []
        for model_name, predictions in model_results.items():
            scores = evaluator.evaluate_single(predictions, target_texts)
            scores['model'] = model_name
            comparison_results.append(scores)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save results
        os.makedirs("evaluation/results", exist_ok=True)
        comparison_df.to_csv("evaluation/results/model_comparison.csv", index=False)
        
        progress(0.9, desc="Generating visualization...")
        
        # Format results
        results_text = f"""
## Dataset Comparison Results

**Dataset:** {dataset_name}
**Language Pair:** {language_pair}
**Samples Evaluated:** {len(source_texts)}

### Comparison Table
{comparison_df.to_markdown(index=False)}

### Results Saved
- Table: `evaluation/results/model_comparison.csv`
- Figure: `evaluation/results/model_comparison.png`
"""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_cols = [col for col in comparison_df.columns if col != "model" and pd.api.types.is_numeric_dtype(comparison_df[col])]
        
        if metric_cols:
            x = range(len(comparison_df))
            width = 0.35
            for i, metric in enumerate(metric_cols[:3]):  # Show first 3 metrics
                values = comparison_df[metric].values
                ax.bar([xi + i*width for xi in x], values, width, label=metric.upper())
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Comparison on Dataset')
            ax.set_xticks([xi + width for xi in x])
            ax.set_xticklabels(comparison_df['model'].values)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("evaluation/results/model_comparison.png", dpi=300, bbox_inches='tight')
        
        # Convert figure to base64 for display
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        figure_html = f'<img src="data:image/png;base64,{img_str}" style="max-width:100%">'
        
        progress(1.0, desc="Complete!")
        
        return results_text, figure_html
        
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", ""


def run_training_ui(base_model, dataset_name, language_pair, num_epochs, batch_size, max_train_samples, progress=gr.Progress()):
    """Run training from UI."""
    try:
        progress(0, desc="Initializing training...")
        
        # Import training functions
        from transformers import (
            AutoTokenizer, AutoModelForSeq2SeqLM,
            Seq2SeqTrainingArguments, Seq2SeqTrainer,
            DataCollatorForSeq2Seq
        )
        from datasets import Dataset
        import torch
        
        # Load dataset
        progress(0.1, desc="Loading dataset...")
        
        if dataset_name == "opus_books":
            train_source, train_target = load_opus_books(
                language_pair=language_pair,
                split="train",
                max_samples=max_train_samples
            )
            eval_source, eval_target = load_opus_books(
                language_pair=language_pair,
                split="validation",
                max_samples=min(100, max_train_samples // 5)
            )
        else:
            # Use custom samples
            custom_data = EXAMPLE_CUSTOM_SAMPLES
            train_source = custom_data["source"][:3]
            train_target = custom_data["target"][:3]
            eval_source = custom_data["source"][3:]
            eval_target = custom_data["target"][3:]
        
        progress(0.2, desc="Loading model...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        model.to(device)
        
        # Prepare datasets
        progress(0.3, desc="Preparing data...")
        
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples["source"],
                max_length=128,
                truncation=True,
                padding="max_length"
            )
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["target"],
                    max_length=128,
                    truncation=True,
                    padding="max_length"
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        train_dataset = Dataset.from_dict({"source": train_source, "target": train_target})
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        
        eval_dataset = Dataset.from_dict({"source": eval_source, "target": eval_target})
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)
        
        # Evaluate before training
        progress(0.4, desc="Evaluating before training...")
        before_predictions = []
        for text in eval_source[:50]:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            before_predictions.append(pred)
        
        before_scores = evaluator.evaluate_single(before_predictions, eval_target[:50])
        
        # Save before scores
        os.makedirs("evaluation/results", exist_ok=True)
        with open("evaluation/results/before_training_scores.json", "w") as f:
            json.dump(before_scores, f, indent=2)
        
        progress(0.5, desc="Starting training...")
        
        # Training setup
        output_dir = f"./models/fine-tuned-{dataset_name}"
        # Handle different transformers versions
        import inspect
        sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
        training_args_dict = {
            "output_dir": output_dir,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "learning_rate": 5e-5,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
        }
        
        if "eval_strategy" in sig.parameters:
            training_args_dict["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in sig.parameters:
            training_args_dict["evaluation_strategy"] = "epoch"
        
        training_args = Seq2SeqTrainingArguments(**training_args_dict)
        
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
        
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            labels = torch.where(torch.tensor(labels) != -100, torch.tensor(labels), tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            scores = evaluator.evaluate_single(decoded_preds, decoded_labels)
            return {"bleu": scores.get("bleu", 0)}
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset.select(range(min(50, len(eval_dataset)))),
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        
        # Train
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        progress(0.9, desc="Evaluating after training...")
        
        # Evaluate after training
        after_predictions = []
        for text in eval_source[:50]:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            after_predictions.append(pred)
        
        after_scores = evaluator.evaluate_single(after_predictions, eval_target[:50])
        
        # Save after scores
        with open("evaluation/results/after_training_scores.json", "w") as f:
            json.dump(after_scores, f, indent=2)
        
        # Format results
        results_text = "## Training Complete! ✅\n\n"
        results_text += f"**Model:** {base_model}\n"
        results_text += f"**Dataset:** {dataset_name}\n"
        results_text += f"**Epochs:** {num_epochs}\n"
        results_text += f"**Training Samples:** {len(train_source)}\n\n"
        
        results_text += "### Performance Comparison\n\n"
        for metric in ['bleu', 'rougeL', 'meteor']:
            if metric in before_scores and metric in after_scores:
                before = before_scores[metric]
                after = after_scores[metric]
                improvement = after - before
                improvement_pct = (improvement / before * 100) if before > 0 else 0
                
                results_text += f"**{metric.upper()}:**\n"
                results_text += f"- Before: {before:.4f}\n"
                results_text += f"- After: {after:.4f}\n"
                results_text += f"- Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)\n\n"
        
        results_text += f"\n**Model saved to:** `{output_dir}`\n"
        results_text += "**Results saved to:** `evaluation/results/`\n"
        results_text += "\nYou can now use the fine-tuned model in other tabs!"
        
        progress(1.0, desc="Complete!")
        
        return results_text
        
    except Exception as e:
        import traceback
        return f"Error during training: {str(e)}\n\n**Troubleshooting:**\n- Check model name is correct\n- Ensure dataset is available\n- Check sufficient memory/disk space\n- Reduce batch_size or max_train_samples if out of memory\n\n**Error details:**\n```\n{traceback.format_exc()}\n```"


# Create Gradio interface
def create_interface():
    """Create and launch Gradio interface."""
    
    available_models = get_available_models()
    
    with gr.Blocks(title="Translation with Language Models") as app:
        gr.Markdown("# 🌐 Translation with Language Models")
        gr.Markdown("Translate text using multiple language models. Compare different models and evaluate translations.")
        
        with gr.Tabs():
            # Tab 1: Single Translation with Evaluation
            with gr.Tab("🔤 Translate & Evaluate"):
                gr.Markdown("### Translate text and evaluate with reference translation")
                
                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            choices=available_models,
                            label="Select Model",
                            value=available_models[0] if available_models else None
                        )
                        input_text = gr.Textbox(
                            label="Input Text (English)",
                            placeholder="Enter text to translate...",
                            lines=4
                        )
                        reference_text = gr.Textbox(
                            label="Reference Translation (Optional, for evaluation)",
                            placeholder="Enter reference French translation to see BLEU score...",
                            lines=2
                        )
                        translate_btn = gr.Button("Translate & Evaluate", variant="primary")
                    
                    with gr.Column():
                        translation_output = gr.Textbox(
                            label="Translation (French)",
                            lines=4
                        )
                        evaluation_output = gr.Markdown(
                            label="Evaluation Results"
                        )
                
                translate_btn.click(
                    fn=translate_with_evaluation,
                    inputs=[input_text, model_dropdown, reference_text],
                    outputs=[translation_output, evaluation_output]
                )
            
            # Tab 2: Model Comparison
            with gr.Tab("⚖️ Compare Models"):
                gr.Markdown("### Compare two models side-by-side with evaluation metrics")
                
                with gr.Row():
                    with gr.Column():
                        input_text_compare = gr.Textbox(
                            label="Source Text",
                            placeholder="Enter text to translate...",
                            lines=3
                        )
                        model1_dropdown = gr.Dropdown(
                            choices=available_models,
                            label="Model 1",
                            value=available_models[0] if available_models else None
                        )
                        model2_dropdown = gr.Dropdown(
                            choices=available_models,
                            label="Model 2",
                            value=available_models[1] if len(available_models) > 1 else available_models[0]
                        )
                        reference_text = gr.Textbox(
                            label="Reference Translation (Optional, for evaluation)",
                            placeholder="Enter reference translation for evaluation...",
                            lines=2
                        )
                        compare_btn = gr.Button("Compare", variant="primary")
                    
                    with gr.Column():
                        translation1_output = gr.Textbox(
                            label="Translation from Model 1",
                            lines=3
                        )
                        translation2_output = gr.Textbox(
                            label="Translation from Model 2",
                            lines=3
                        )
                        eval_output = gr.Markdown(
                            label="Evaluation Results"
                        )
                
                compare_btn.click(
                    fn=compare_translations,
                    inputs=[input_text_compare, model1_dropdown, model2_dropdown, reference_text],
                    outputs=[translation1_output, translation2_output, eval_output]
                )
            
            # Tab 3: Dataset Comparison
            with gr.Tab("📊 Compare on Dataset"):
                gr.Markdown("### Compare multiple models on a dataset")
                gr.Markdown("This will evaluate models on a full dataset and generate comparison tables and figures.")
                
                with gr.Row():
                    with gr.Column():
                        dataset_model1 = gr.Dropdown(
                            choices=available_models,
                            label="Model 1",
                            value=available_models[0] if available_models else None
                        )
                        dataset_model2 = gr.Dropdown(
                            choices=available_models,
                            label="Model 2",
                            value=available_models[1] if len(available_models) > 1 else available_models[0]
                        )
                        dataset_name = gr.Dropdown(
                            choices=["opus_books", "custom"],
                            value="opus_books",
                            label="Dataset"
                        )
                        lang_pair = gr.Textbox(
                            label="Language Pair",
                            value="en-fr",
                            placeholder="e.g., en-fr, en-de"
                        )
                        max_samples_input = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=50,
                            step=10,
                            label="Number of Samples"
                        )
                        compare_dataset_btn = gr.Button("Run Dataset Comparison", variant="primary")
                    
                    with gr.Column():
                        dataset_results = gr.Markdown(label="Comparison Results")
                        dataset_figure = gr.HTML(label="Comparison Figure")
                
                compare_dataset_btn.click(
                    fn=run_dataset_comparison,
                    inputs=[dataset_model1, dataset_model2, dataset_name, lang_pair, max_samples_input],
                    outputs=[dataset_results, dataset_figure]
                )
            
            # Tab 4: Train Model
            with gr.Tab("🚀 Train Model"):
                gr.Markdown("### Fine-tune a translation model")
                gr.Markdown("⚠️ **Warning:** Training takes time! (CPU: hours, GPU: minutes)")
                
                with gr.Row():
                    with gr.Column():
                        base_model_input = gr.Textbox(
                            label="Base Model",
                            value="Helsinki-NLP/opus-mt-en-fr",
                            placeholder="Helsinki-NLP/opus-mt-en-fr"
                        )
                        train_dataset = gr.Dropdown(
                            choices=["opus_books", "custom"],
                            value="opus_books",
                            label="Training Dataset"
                        )
                        train_lang_pair = gr.Textbox(
                            label="Language Pair",
                            value="en-fr",
                            placeholder="e.g., en-fr"
                        )
                        epochs_input = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Number of Epochs"
                        )
                        batch_size_input = gr.Slider(
                            minimum=2,
                            maximum=32,
                            value=8,
                            step=2,
                            label="Batch Size"
                        )
                        train_samples = gr.Slider(
                            minimum=50,
                            maximum=5000,
                            value=500,
                            step=50,
                            label="Training Samples"
                        )
                        train_btn = gr.Button("Start Training", variant="primary")
                    
                    with gr.Column():
                        training_results = gr.Markdown(label="Training Results")
                
                train_btn.click(
                    fn=run_training_ui,
                    inputs=[base_model_input, train_dataset, train_lang_pair, epochs_input, batch_size_input, train_samples],
                    outputs=training_results
                )
            
            # Tab 5: Training Comparison (View Results)
            with gr.Tab("📈 View Training Results"):
                gr.Markdown("### View training performance comparison")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ### Training Performance Comparison
                        
                        This section shows the improvement in model performance after fine-tuning.
                        
                        **Metrics Shown:**
                        - BLEU Score (primary metric)
                        - Other evaluation metrics if available
                        
                        **Expected Results:**
                        - Baseline (before): ~0.20 BLEU
                        - Fine-tuned (after): ~0.22-0.25 BLEU
                        - Improvement: 10-25% increase
                        """)
                        show_comparison_btn = gr.Button("Show Training Comparison", variant="primary")
                    
                    with gr.Column():
                        training_output = gr.Markdown(
                            label="Training Comparison Results"
                        )
                
                show_comparison_btn.click(
                    fn=show_training_comparison,
                    outputs=training_output
                )
            
            # Tab 6: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## Neural Machine Translation with OPUS Books
                
                ### Features:
                - **Translation**: Translate English to French using multiple models
                - **Model Selection**: Choose from available translation models
                - **Evaluation**: Get BLEU, ROUGE, METEOR scores when reference translations are provided
                - **Comparison**: Compare multiple models side-by-side
                - **Training Results**: View before/after training performance
                
                ### Available Models:
                - **OPUS-MT**: Pre-trained OPUS-MT models (en-de, en-fr, en-es)
                - **mBART50**: Multilingual BART model
                - **mT5**: Multilingual T5 model
                
                ### Evaluation Metrics:
                - **BLEU**: Bilingual Evaluation Understudy - measures n-gram precision
                - **ROUGE-L**: Longest common subsequence based F-score
                - **METEOR**: Explicit ordering and synonym matching
                - **chrF++**: Character-level F-score
                
                ### Usage:
                1. **Translate & Evaluate**: Enter text, select model, optionally provide reference
                2. **Compare Models**: Compare two models on individual sentences
                3. **Compare on Dataset**: Run full dataset comparison (generates tables/figures)
                4. **Train Model**: Fine-tune a model on a dataset
                5. **View Training Results**: See before/after training performance
                
                ### All Features Available in UI:
                - ✅ Translation with language models
                - ✅ Model comparison on datasets (with tables & figures)
                - ✅ Custom test samples comparison
                - ✅ Model training (fine-tuning)
                - ✅ Training performance comparison
                """)
    
    return app


if __name__ == "__main__":
    print("="*80)
    print("Launching Enhanced Translation UI...")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Model selection (OPUS-MT, mBART50, mT5)")
    print("  ✓ Translation with evaluation metrics (BLEU, ROUGE, METEOR)")
    print("  ✓ Side-by-side model comparison")
    print("  ✓ Before/after training performance comparison")
    print("\n" + "="*80)
    
    # Pre-load models for faster first translation
    preload_models()
    
    print("Starting server...")
    print("\n" + "="*80)
    print("🌐 UI is now running!")
    print("="*80)
    print("\nOpen your browser and go to:")
    print("  → http://localhost:7860")
    print("  → or http://127.0.0.1:7860")
    print("\nPress Ctrl+C to stop the server\n")
    print("="*80 + "\n")
    
    app = create_interface()
    # Use localhost for browser access, 0.0.0.0 for network access
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)


