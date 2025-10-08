import sys
import os

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self):
        try:
            self.config = ConfigurationManager().get_model_evaluation_config()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_path, 
                local_files_only=True
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.config.model_path,
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            
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
            
            # Map to readable entity types
            self.entity_type_map = {
                "PER": "PERSON",
                "ORG": "ORGANIZATION", 
                "LOC": "LOCATION",
                "MISC": "MISCELLANEOUS"
            }
            
            logger.info("NER model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NER pipeline: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean input text for NER processing"""
        if not text or len(text.strip()) == 0:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def predict(self, text: str, **gen_kwargs) -> List[Dict]:
        """Perform Named Entity Recognition on input text"""
        
        # Default parameters for NER
        default_kwargs = {
            "confidence_threshold": 0.7,
            "max_entities": 50,
            "return_confidence": True,
            "merge_adjacent": True
        }
        
        # Update defaults with provided kwargs
        for key, value in gen_kwargs.items():
            if key in default_kwargs and value is not None:
                default_kwargs[key] = value
        
        logger.info(f"Input text length: {len(text)}")
        
        try:
            # Clean input text
            processed_text = self.clean_text(text)
            
            if not processed_text:
                return []
            
            if len(processed_text) < 10:
                return []
            
            # Tokenize text for NER - FIXED: Remove is_split_into_words for regular strings
            inputs = self.tokenizer(
                processed_text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
                return_offsets_mapping=False  # Remove this to avoid issues
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask")
                )
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
                # Get confidence scores (softmax probabilities)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                confidence_scores = np.max(probabilities, axis=-1)
            
            # Get word IDs for alignment
            word_ids = inputs.word_ids(batch_index=0)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            # Align predictions with words
            aligned_entities = []
            current_entity = None
            
            for idx, (word_idx, prediction, confidence) in enumerate(zip(word_ids, predictions, confidence_scores)):
                if word_idx is None:
                    # Skip special tokens
                    continue
                
                entity_label = self.label_list[prediction]
                
                # If it's a new entity or different from current
                if entity_label.startswith('B-') or current_entity is None or current_entity["entity"] != entity_label:
                    # Save current entity if it exists
                    if current_entity is not None:
                        aligned_entities.append(current_entity)
                    
                    # Start new entity
                    if entity_label != 'O':
                        current_entity = {
                            "entity": entity_label,
                            "word": tokens[idx].replace("##", ""),
                            "start": idx,
                            "end": idx,
                            "score": confidence,
                            "type": self.entity_type_map.get(entity_label.split("-")[-1], "OTHER")
                        }
                    else:
                        current_entity = None
                elif current_entity is not None and entity_label.startswith('I-') and current_entity["entity"].endswith(entity_label[2:]):
                    # Continue current entity
                    current_entity["word"] += tokens[idx].replace("##", "")
                    current_entity["end"] = idx
                    current_entity["score"] = max(current_entity["score"], confidence)
                else:
                    # Entity changed or ended
                    if current_entity is not None:
                        aligned_entities.append(current_entity)
                        current_entity = None
            
            # Add the last entity if it exists
            if current_entity is not None:
                aligned_entities.append(current_entity)
            
            # Post-process entities
            processed_entities = self.postprocess_entities(
                aligned_entities, 
                default_kwargs["confidence_threshold"]
            )
            
            # Merge adjacent entities if requested
            if default_kwargs["merge_adjacent"]:
                processed_entities = self.merge_adjacent_entities(processed_entities)
            
            # Limit number of entities
            if default_kwargs["max_entities"] > 0:
                processed_entities = processed_entities[:default_kwargs["max_entities"]]
            
            logger.info(f"Found {len(processed_entities)} entities")
            return processed_entities
            
        except torch.cuda.OutOfMemoryError:
            error_msg = "Error: GPU out of memory. Try with shorter text."
            logger.error(error_msg)
            return []
            
        except Exception as e:
            error_msg = f"Error during NER processing: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())  # Add detailed traceback
            return []
    
    def postprocess_entities(self, entities: List[Dict], confidence_threshold: float = 0.7) -> List[Dict]:
        """Post-process entities and filter by confidence"""
        processed_entities = []
        
        for entity in entities:
            # Filter out "O" (outside) entities
            if entity["entity"] == "O":
                continue
            
            # Filter by confidence threshold
            if entity["score"] < confidence_threshold:
                continue
            
            # Clean entity word
            entity_word = entity["word"].replace(" ##", "").replace("##", "")
            
            # Skip very short entities (likely noise)
            if len(entity_word.strip()) < 2:
                continue
            
            processed_entity = {
                "text": entity_word,
                "type": entity["type"],
                "entity_label": entity["entity"],
                "confidence": round(entity["score"], 3),
                "start_pos": entity["start"],
                "end_pos": entity["end"]
            }
            
            processed_entities.append(processed_entity)
        
        return processed_entities
    
    def merge_adjacent_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge adjacent entities of the same type"""
        if not entities:
            return []
        
        merged_entities = []
        current_entity = entities[0].copy()
        
        for i in range(1, len(entities)):
            current = entities[i]
            prev = entities[i-1]
            
            # Check if adjacent and same type
            if (current["type"] == prev["type"] and 
                current["start_pos"] == prev["end_pos"] + 1):
                # Merge entities
                current_entity["text"] += " " + current["text"]
                current_entity["end_pos"] = current["end_pos"]
                current_entity["confidence"] = (current_entity["confidence"] + current["confidence"]) / 2
            else:
                # Save current entity and start new one
                merged_entities.append(current_entity)
                current_entity = current.copy()
        
        # Add the last entity
        merged_entities.append(current_entity)
        
        return merged_entities
    
    def get_entity_types(self) -> List[str]:
        """Get available entity types"""
        return list(self.entity_type_map.values())
    
    def get_recommended_parameters(self, text_length: int) -> Dict[str, Any]:
        """Get recommended parameters based on text length"""
        if text_length < 100:
            return {
                "confidence_threshold": 0.8,
                "max_entities": 10
            }
        elif text_length < 500:
            return {
                "confidence_threshold": 0.7,
                "max_entities": 25
            }
        else:
            return {
                "confidence_threshold": 0.6,
                "max_entities": 50
            }
    
    def analyze_entity_statistics(self, entities: List[Dict]) -> Dict[str, Any]:
        """Analyze entity statistics for the given text"""
        stats = {
            "total_entities": len(entities),
            "entities_by_type": defaultdict(int),
            "average_confidence": 0.0,
            "confidence_distribution": defaultdict(int)
        }
        
        if not entities:
            return stats
        
        total_confidence = 0.0
        
        for entity in entities:
            entity_type = entity["type"]
            confidence = entity["confidence"]
            
            stats["entities_by_type"][entity_type] += 1
            total_confidence += confidence
            
            # Categorize confidence
            if confidence >= 0.9:
                stats["confidence_distribution"]["high"] += 1
            elif confidence >= 0.7:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1
        
        stats["average_confidence"] = round(total_confidence / len(entities), 3)
        stats["entities_by_type"] = dict(stats["entities_by_type"])
        stats["confidence_distribution"] = dict(stats["confidence_distribution"])
        
        return stats