import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def parse_conll2003_file(self, file_path):
        """Parse CoNLL-2003 format text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        examples = []
        tokens = []
        ner_tags = []
        example_id = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('-DOCSTART-') or line == '':
                if tokens:  # Save current sentence
                    examples.append({
                        'id': str(example_id),
                        'tokens': tokens,
                        'ner_tags': ner_tags
                    })
                    example_id += 1
                    tokens = []
                    ner_tags = []
                continue
            
            parts = line.split()
            if len(parts) >= 4:  # -DOCSTART- lines have fewer parts
                token = parts[0]
                ner_tag = parts[3]
                
                # Convert NER tag to numerical label
                tag_mapping = {
                    'O': 0, 'B-PER': 1, 'I-PER': 2, 
                    'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 
                    'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8
                }
                
                tokens.append(token)
                ner_tags.append(tag_mapping.get(ner_tag, 0))
        
        # Don't forget the last sentence
        if tokens:
            examples.append({
                'id': str(example_id),
                'tokens': tokens,
                'ner_tags': ner_tags
            })
        
        return examples

    def load_conll2003_manual(self):
        """Load CoNLL-2003 dataset by parsing text files manually"""
        data_path = self.config.data_path
        
        splits = {}
        for split_name in ['train', 'test', 'validation']:
            # CoNLL-2003 uses 'valid.txt' for validation
            file_name = 'valid.txt' if split_name == 'validation' else f'{split_name}.txt'
            file_path = os.path.join(data_path, file_name)
            
            if os.path.exists(file_path):
                logger.info(f"Parsing {file_path}...")
                examples = self.parse_conll2003_file(file_path)
                splits[split_name] = examples
                logger.info(f"Loaded {len(examples)} examples from {file_name}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        # Convert to Hugging Face dataset format
        dataset_dict = {}
        for split_name, examples in splits.items():
            dataset_dict[split_name] = Dataset.from_list(examples)
        
        return DatasetDict(dataset_dict)

    def tokenize_and_align_labels(self, examples):
        """
        NER-specific tokenization and label alignment
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            is_split_into_words=True  # CRITICAL for NER
        )
        
        labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                # Special tokens (CLS, SEP, PAD) get label -100
                if word_idx is None:
                    label_ids.append(-100)
                # Only label the first token of a given word
                elif word_idx != previous_word_idx:
                    label_ids.append(ner_tags[word_idx])
                # For subsequent subword tokens, use -100
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def convert(self):
        """Convert CoNLL-2003 dataset for NER training using manual parser"""
        try:
            logger.info("Loading CoNLL-2003 dataset from text files using manual parser...")
            dataset = self.load_conll2003_manual()
            logger.info(f"Loaded dataset structure: {dataset}")
            
            # Show sample data
            if 'train' in dataset:
                logger.info("Sample from training set:")
                sample = dataset['train'][0]
                logger.info(f"Tokens: {sample['tokens']}")
                logger.info(f"NER tags: {sample['ner_tags']}")
            
            # Apply transformation
            dataset_processed = dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                batch_size=1000
            )
            
            # Save processed dataset
            output_path = os.path.join(self.config.root_dir, "ner_processed_dataset")
            dataset_processed.save_to_disk(output_path)
            logger.info(f"Processed NER dataset saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise e