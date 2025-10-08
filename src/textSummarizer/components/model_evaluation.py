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
        # Define label mappings for CoNLL-2003
        self.label_list = [
            "O",        # 0: Outside
            "B-PER",    # 1: Beginning of Person
            "I-PER",    # 2: Inside of Person  
            "B-ORG",    # 3: Beginning of Organization
            "I-ORG",    # 4: Inside of Organization
            "B-LOC",    # 5: Beginning of Location
            "I-LOC",    # 6: Inside of Location
            "B-MISC",   # 7: Beginning of Miscellaneous
            "I-MISC"    # 8: Inside of Miscellaneous
        ]

    def convert_to_string_labels(self, predictions, references):
        """
        Convert integer predictions and references to string labels
        """
        str_predictions = []
        str_references = []
        
        for pred_seq, ref_seq in zip(predictions, references):
            str_pred_seq = []
            str_ref_seq = []
            
            for pred, ref in zip(pred_seq, ref_seq):
                # Skip padding tokens (-100)
                if ref != -100:
                    str_pred_seq.append(self.label_list[pred])
                    str_ref_seq.append(self.label_list[ref])
            
            str_predictions.append(str_pred_seq)
            str_references.append(str_ref_seq)
        
        return str_predictions, str_references

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
            batch_tokens = batch["tokens"]
            batch_labels = batch["ner_tags"]  # Use ner_tags instead of labels
            
            # Tokenize inputs and get word_ids for alignment
            tokenized_inputs = tokenizer(
                batch_tokens,
                truncation=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            
            # Align predictions with original tokens using word_ids
            for j in range(len(batch_tokens)):
                # Get word_ids for this specific example
                word_ids = tokenized_inputs.word_ids(batch_index=j)
                previous_word_idx = None
                aligned_predictions = []
                aligned_labels = []
                
                for k, word_idx in enumerate(word_ids):
                    if word_idx is None:
                        # Skip special tokens ([CLS], [SEP], [PAD])
                        continue
                    elif word_idx != previous_word_idx:
                        # Only take the first token of each word
                        aligned_predictions.append(predictions[j][k])
                        # Get the corresponding label
                        aligned_labels.append(batch_labels[j][word_idx])
                    previous_word_idx = word_idx
                
                all_predictions.append(aligned_predictions)
                all_labels.append(aligned_labels)
        
        # Convert to string labels for seqeval
        str_predictions, str_references = self.convert_to_string_labels(all_predictions, all_labels)
        
        # Compute metrics
        results = metric.compute(predictions=str_predictions, references=str_references)
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
            'overall_accuracy': results.get('overall_accuracy', 0),
            'overall_precision': results.get('overall_precision', 0),
            'overall_recall': results.get('overall_recall', 0),
            'overall_f1': results.get('overall_f1', 0)
        }
        
        # Add per-entity metrics
        entity_types = ['PER', 'ORG', 'LOC', 'MISC']
        for entity_type in entity_types:
            if entity_type in results:
                metrics_dict[f'{entity_type}_precision'] = results[entity_type]['precision']
                metrics_dict[f'{entity_type}_recall'] = results[entity_type]['recall']
                metrics_dict[f'{entity_type}_f1'] = results[entity_type]['f1']
            else:
                # If entity type not found, set default values
                metrics_dict[f'{entity_type}_precision'] = 0
                metrics_dict[f'{entity_type}_recall'] = 0
                metrics_dict[f'{entity_type}_f1'] = 0
        
        # Save to CSV
        df = pd.DataFrame([metrics_dict])
        df.to_csv(self.config.metric_file_name, index=False)
        
        # Print results
        logger.info("NER Evaluation Results:")
        logger.info(f"Overall F1: {metrics_dict['overall_f1']:.4f}")
        logger.info(f"Overall Precision: {metrics_dict['overall_precision']:.4f}")
        logger.info(f"Overall Recall: {metrics_dict['overall_recall']:.4f}")
        logger.info(f"Overall Accuracy: {metrics_dict['overall_accuracy']:.4f}")
        
        # Log per-entity results
        for entity_type in entity_types:
            logger.info(f"{entity_type} - F1: {metrics_dict[f'{entity_type}_f1']:.4f}, "
                       f"Precision: {metrics_dict[f'{entity_type}_precision']:.4f}, "
                       f"Recall: {metrics_dict[f'{entity_type}_recall']:.4f}")
        
        logger.info(f"Evaluation complete. Metrics saved to: {self.config.metric_file_name}")
        
        return metrics_dict