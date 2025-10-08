from transformers import AutoModelForTokenClassification, AutoTokenizer
from textSummarizer.logging import logger
from datasets import load_from_disk
import evaluate
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from textSummarizer.entity import ModelEvaluationConfig
import os

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, 
                                    batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Calculate NER metrics (precision, recall, f1) on test dataset
        """
        all_predictions = []
        all_labels = []
        
        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i + batch_size]
            
            # Tokenize inputs
            inputs = tokenizer(
                batch["tokens"],
                truncation=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            
            # Align predictions with original tokens
            for j, (prediction, label) in enumerate(zip(predictions, batch["labels"])):
                word_ids = inputs.word_ids(batch_index=j)
                previous_word_idx = None
                aligned_predictions = []
                aligned_labels = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        # Skip special tokens
                        continue
                    elif word_idx != previous_word_idx:
                        # Only take the first token of each word
                        aligned_predictions.append(prediction[word_idx])
                        aligned_labels.append(label[word_idx])
                    previous_word_idx = word_idx
                
                all_predictions.append(aligned_predictions)
                all_labels.append(aligned_labels)
        
        # Compute metrics
        results = metric.compute(predictions=all_predictions, references=all_labels)
        return results

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForTokenClassification.from_pretrained(self.config.model_path).to(device)
        
        # Load dataset
        dataset = load_from_disk(self.config.data_path)
        
        # Load seqeval metric for NER
        ner_metric = evaluate.load("seqeval")
        
        # Use a smaller sample for faster evaluation
        test_dataset = dataset['test'].select(range(200))  # Evaluate on 200 examples
        
        logger.info(f"Evaluating on {len(test_dataset)} examples...")
        
        # Compute NER metrics
        results = self.calculate_metric_on_test_ds(
            test_dataset, ner_metric, model, tokenizer,
            batch_size=16, device=device
        )
        
        # Extract main metrics
        metrics_dict = {
            'overall_accuracy': results['overall_accuracy'],
            'overall_precision': results['overall_precision'],
            'overall_recall': results['overall_recall'],
            'overall_f1': results['overall_f1']
        }
        
        # Add per-entity metrics
        for entity_type in ['PER', 'ORG', 'LOC', 'MISC']:
            if entity_type in results:
                metrics_dict[f'{entity_type}_precision'] = results[entity_type]['precision']
                metrics_dict[f'{entity_type}_recall'] = results[entity_type]['recall']
                metrics_dict[f'{entity_type}_f1'] = results[entity_type]['f1']
        
        # Save to CSV
        df = pd.DataFrame([metrics_dict])
        df.to_csv(self.config.metric_file_name, index=False)
        
        # Print results
        logger.info("NER Evaluation Results:")
        logger.info(f"Overall F1: {metrics_dict['overall_f1']:.4f}")
        logger.info(f"Overall Precision: {metrics_dict['overall_precision']:.4f}")
        logger.info(f"Overall Recall: {metrics_dict['overall_recall']:.4f}")
        logger.info(f"Overall Accuracy: {metrics_dict['overall_accuracy']:.4f}")
        
        logger.info(f"Evaluation complete. Metrics saved to: {self.config.metric_file_name}")
        
        return metrics_dict