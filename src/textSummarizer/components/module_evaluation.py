from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import evaluate
import torch
import pandas as pd
from tqdm import tqdm
from textSummarizer.entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, 
                                    batch_size=8, device="cuda" if torch.cuda.is_available() else "cpu", 
                                    column_text="dialogue", column_summary="summary",
                                    max_input_length=1024, max_output_length=256):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            # Tokenize inputs
            inputs = tokenizer(article_batch, max_length=max_input_length, truncation=True, 
                               padding="max_length", return_tensors="pt")
            
            # Generate summaries
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device), 
                length_penalty=0.8,
                num_beams=8,
                max_length=max_output_length
            )
            
            # Decode predictions
            decoded_summaries = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                for s in summaries
            ]
            
            # Add batch to metric
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       
        # Load dataset
        dataset = load_from_disk(self.config.data_path)

        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        # Use evaluate instead of load_metric
        rouge_metric = evaluate.load("rouge")

        # Use a larger sample for realistic evaluation
        test_dataset = dataset['test'][:500]  # adjust number based on your GPU/memory

        # Compute ROUGE scores
        score = self.calculate_metric_on_test_ds(
            test_dataset, rouge_metric, model, tokenizer,
            batch_size=8, column_text='dialogue', column_summary='summary',
            max_input_length=1024, max_output_length=256
        )

        # Create dictionary directly from scores (no .mid)
        rouge_dict = {rn: score[rn] for rn in rouge_names}

        # Save to CSV
        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        df.to_csv(self.config.metric_file_name, index=False)

        print("Evaluation complete. ROUGE scores saved to:", self.config.metric_file_name)