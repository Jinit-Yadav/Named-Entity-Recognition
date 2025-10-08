from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_from_disk
import torch
import os
from textSummarizer.entity import ModelTrainerConfig
from textSummarizer.logging import logger

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load tokenizer and model - use distilbert for speed
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        
        # Define label mappings for CoNLL-2003
        label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        num_labels = len(label_list)
        
        model = AutoModelForTokenClassification.from_pretrained(
            "distilbert-base-cased",
            num_labels=num_labels
        ).to(device)

        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Load dataset
        dataset_ner = load_from_disk(self.config.data_path)
        
        # Use very small subset for ultra-fast training
        train_dataset = dataset_ner["train"].select(range(300))  # Only 300 examples

        # ULTRA-SIMPLE Training arguments - ALL HARDCODED
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            max_steps=30,  # Only 30 steps!
            logging_steps=5,
            save_steps=1000,
            report_to=None,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Start training
        logger.info("🚀 Starting ULTRA-FAST NER training (30 steps only)...")
        logger.info(f"📊 Training on {len(train_dataset)} examples")
        logger.info("⏰ Expected time: 30-60 seconds...")
        
        trainer.train()
        
        # Save model
        output_path = os.path.join(self.config.root_dir, "ner-model")
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        
        logger.info(f"✅ Training completed! Model saved to: {output_path}")
        
        return trainer