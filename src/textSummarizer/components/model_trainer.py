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

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        
        # Define label mappings for CoNLL-2003
        label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        num_labels = len(label_list)
        
        model = AutoModelForTokenClassification.from_pretrained(
            "distilbert-base-cased",
            num_labels=num_labels,
            id2label={i: label for i, label in enumerate(label_list)},
            label2id={label: i for i, label in enumerate(label_list)}
        ).to(device)

        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        # Load dataset
        dataset_ner = load_from_disk(self.config.data_path)
        
        # Use reasonable amount of data
        train_dataset = dataset_ner["train"].select(range(2000))  # 2000 examples is enough
        
        # FIXED TRAINING ARGUMENTS - removed evaluation_strategy
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=3,  # Train for 3 full epochs
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            logging_steps=50,
            save_steps=500,
            # Remove evaluation_strategy since we're not using eval dataset
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
        logger.info("Starting NER training...")
        logger.info(f"Training on {len(train_dataset)} examples for 3 epochs")
        logger.info("Expected time: 10-15 minutes...")
        
        trainer.train()
        
        # Save model
        output_path = os.path.join(self.config.root_dir, "ner-model")
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        
        logger.info(f"Training completed! Model saved to: {output_path}")
        
        return trainer