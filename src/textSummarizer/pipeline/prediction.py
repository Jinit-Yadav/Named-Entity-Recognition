import sys
import os

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import logging
from typing import Dict, Any, Optional
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_path,
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Thoroughly clean input text"""
        if not text or len(text.strip()) == 0:
            return ""
        
        # Remove Wikipedia citations like [1], [a], etc.
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[[a-z]\]', '', text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common formatting issues
        text = re.sub(r'\s*\.\s*', '. ', text)
        text = re.sub(r'\s*,\s*', ', ', text)
        
        return text
    
    def extract_key_sentences(self, text: str, num_sentences: int = 5) -> str:
        """Extract key sentences as a fallback when model fails"""
        try:
            sentences = sent_tokenize(text)
            if len(sentences) <= num_sentences:
                return text
            
            # Simple heuristic: take first, middle, and last sentences
            key_indices = [0, len(sentences)//3, 2*len(sentences)//3, -1]
            key_sentences = [sentences[i] for i in key_indices if i < len(sentences)]
            
            return ' '.join(key_sentences)
        except:
            return text[:500]  # Fallback to first 500 chars
    
    def postprocess_summary(self, summary: str) -> str:
        """Aggressive post-processing to fix model output"""
        if not summary or len(summary.strip()) == 0:
            return "Unable to generate summary. Please try with different text or parameters."
        
        # Remove extra whitespace
        summary = re.sub(r'\s+', ' ', summary.strip())
        
        # Fix the specific issues seen in your output
        summary = re.sub(r'\.\.+', '.', summary)  # Fix multiple dots
        summary = re.sub(r'\s*\.\s*', '. ', summary)  # Ensure space after dots
        summary = re.sub(r',\s*,', ',', summary)  # Fix multiple commas
        summary = re.sub(r'\s*,\s*', ', ', summary)  # Ensure space after commas
        
        # Remove isolated words and fragments (like "I.A. at. a.. (.")
        summary = re.sub(r'\b[A-Z]\.\s*[A-Z]?\.?\s*[a-z]?\.?', '', summary)
        summary = re.sub(r'\s*\.\s*\.\s*', '. ', summary)
        
        # Fix sentence casing and structure
        sentences = []
        for sentence in re.split(r'(?<=[.!?])\s+', summary):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Remove very short fragments (likely artifacts)
            if len(sentence.split()) < 3:
                continue
                
            # Capitalize first letter
            if sentence and sentence[0].isalpha():
                sentence = sentence[0].upper() + sentence[1:]
            sentences.append(sentence)
        
        if not sentences:
            return "Summary generation failed. The output contained mostly artifacts."
        
        summary = '. '.join(sentences)
        
        # Ensure it ends with proper punctuation
        if not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        # Final cleanup
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        return summary
    
    def is_quality_summary(self, summary: str, original_text: str) -> bool:
        """Check if the generated summary is of acceptable quality"""
        if not summary or len(summary) < 20:
            return False
        
        # Check for excessive fragments
        sentences = re.split(r'[.!?]', summary)
        valid_sentences = [s for s in sentences if len(s.strip().split()) >= 3]
        
        if len(valid_sentences) < 2:
            return False
        
        # Check for common artifact patterns
        artifact_patterns = [
            r'\b[A-Z]\.[A-Z]\.',  # I.A. patterns
            r'\s*\.\s*\.\s*',     # Multiple dots
            r'^[^a-zA-Z]*$',      # No alphabetic characters
        ]
        
        for pattern in artifact_patterns:
            if re.search(pattern, summary):
                return False
        
        return True
    
    def predict(self, text: str, **gen_kwargs) -> str:
        """Generate summary with multiple fallback strategies"""
        
        # Enhanced default parameters for better quality
        default_kwargs = {
            "length_penalty": 2.0,  # Increased for longer, more coherent summaries
            "num_beams": 8,         # More beams for better quality
            "max_length": 300,
            "min_length": 100,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "do_sample": False,
            "temperature": 0.8,     # Lower temperature for less randomness
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
                return "Error: Input text is empty or invalid."
            
            if len(processed_text) < 100:
                return "Error: Text is too short for summarization. Please provide at least 100 characters."
            
            # Tokenize with appropriate settings
            inputs = self.tokenizer(
                processed_text, 
                max_length=1024, 
                truncation=True, 
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary with multiple attempts if needed
            max_attempts = 2
            for attempt in range(max_attempts):
                try:
                    with torch.no_grad():
                        summary_ids = self.model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            **default_kwargs
                        )
                    
                    # Decode summary
                    summary = self.tokenizer.decode(
                        summary_ids[0], 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Post-process
                    summary = self.postprocess_summary(summary)
                    
                    # Check quality
                    if self.is_quality_summary(summary, processed_text):
                        logger.info(f"Quality summary generated (attempt {attempt + 1})")
                        return summary
                    else:
                        logger.warning(f"Poor quality summary on attempt {attempt + 1}")
                        # Adjust parameters for retry
                        default_kwargs["temperature"] = max(0.5, default_kwargs["temperature"] - 0.1)
                        default_kwargs["num_beams"] = min(10, default_kwargs["num_beams"] + 1)
                
                except Exception as e:
                    logger.error(f"Generation attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_attempts - 1:
                        raise
            
            # If all attempts failed, use extractive fallback
            logger.info("Using extractive fallback summary")
            fallback_summary = self.extract_key_sentences(processed_text)
            return self.postprocess_summary(fallback_summary)
            
        except torch.cuda.OutOfMemoryError:
            error_msg = "Error: GPU out of memory. Try with shorter text or reduce beam size."
            logger.error(error_msg)
            return error_msg
            
        except Exception as e:
            error_msg = f"Error during summarization: {str(e)}"
            logger.error(error_msg)
            # Final fallback
            fallback_summary = self.extract_key_sentences(self.clean_text(text))
            return f"Note: Using basic summary due to generation issues.\n\n{fallback_summary}"
    
    def get_recommended_parameters(self, text_length: int) -> Dict[str, Any]:
        """Get recommended parameters based on text characteristics"""
        if text_length < 500:
            return {
                "max_length": 80,
                "min_length": 40,
                "num_beams": 4,
                "length_penalty": 1.5
            }
        elif text_length < 2000:
            return {
                "max_length": 150,
                "min_length": 80,
                "num_beams": 6,
                "length_penalty": 2.0
            }
        else:
            return {
                "max_length": 200,
                "min_length": 100,
                "num_beams": 8,
                "length_penalty": 2.5
            }