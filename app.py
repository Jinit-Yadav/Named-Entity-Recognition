from flask import Flask, render_template, request, jsonify
from textSummarizer.pipeline.prediction import PredictionPipeline
import logging
import time
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize pipeline
try:
    pipeline = PredictionPipeline()
    logger.info("Prediction pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize prediction pipeline: {e}")
    pipeline = None

def validate_text(text: str) -> tuple[bool, str]:
    """Validate input text"""
    if not text or len(text.strip()) == 0:
        return False, "Please enter some text to summarize."
    
    if len(text.strip()) < 100:
        return False, "Text is too short. Please provide at least 100 characters."
    
    # Check if text has meaningful content (not just special characters)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if len(words) < 10:
        return False, "Text doesn't contain enough meaningful content."
    
    return True, ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    summary = None
    original_text = ""
    processing_time = 0
    word_count = 0
    char_count = 0
    error_message = ""
    
    if request.method == "POST":
        start_time = time.time()
        
        text = request.form.get("text", "").strip()
        original_text = text
        
        # Validate input
        is_valid, validation_msg = validate_text(text)
        if not is_valid:
            error_message = validation_msg
        else:
            word_count = len(text.split())
            char_count = len(text)
            
            # Get and validate parameters
            try:
                length_penalty = float(request.form.get("length_penalty", 2.0))
                num_beams = int(request.form.get("num_beams", 6))
                max_length = int(request.form.get("max_length", 200))
                min_length = int(request.form.get("min_length", 80))
                
                # Validate ranges
                length_penalty = max(0.5, min(3.0, length_penalty))
                num_beams = max(2, min(12, num_beams))
                max_length = max(50, min(500, max_length))
                min_length = max(20, min(300, min_length))
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid parameter values: {e}")
                # Use recommended parameters
                recommended = pipeline.get_recommended_parameters(len(text))
                length_penalty = recommended["length_penalty"]
                num_beams = recommended["num_beams"]
                max_length = recommended["max_length"]
                min_length = recommended["min_length"]
            
            if pipeline is None:
                error_message = "Summarization service is currently unavailable. Please try again later."
            else:
                # Prepare generation parameters
                gen_kwargs = {
                    "length_penalty": length_penalty,
                    "num_beams": num_beams,
                    "max_length": max_length,
                    "min_length": min_length,
                }
                
                # Process text
                summary = pipeline.predict(text, **gen_kwargs)
                processing_time = round(time.time() - start_time, 2)
                
                logger.info(f"Summary generated in {processing_time} seconds")
    
    return render_template(
        "home.html", 
        summary=summary, 
        original_text=original_text,
        processing_time=processing_time,
        word_count=word_count,
        char_count=char_count,
        error_message=error_message
    )

# ... rest of your Flask app routes remain the same ...
@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    """API endpoint for summarization"""
    if pipeline is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Get parameters with defaults
        params = data.get("parameters", {})
        gen_kwargs = {
            "length_penalty": float(params.get("length_penalty", 1.0)),
            "num_beams": int(params.get("num_beams", 4)),
            "max_length": int(params.get("max_length", 256)),
            "min_length": int(params.get("min_length", 100)),
            "temperature": float(params.get("temperature", 1.0)),
            "top_p": float(params.get("top_p", 1.0))
        }
        
        # Truncate long texts
        if len(text) > 10000:
            text = text[:10000]
        
        summary = pipeline.predict(text, **gen_kwargs)
        
        return jsonify({
            "summary": summary,
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": round(len(text) / len(summary), 2) if summary else 0
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/recommend_params", methods=["POST"])
def api_recommend_params():
    """API endpoint to get recommended parameters based on text length"""
    if pipeline is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        data = request.get_json()
        text_length = data.get("text_length", 0)
        
        recommended = pipeline.get_recommended_parameters(text_length)
        
        return jsonify({
            "recommended_parameters": recommended
        })
        
    except Exception as e:
        logger.error(f"Parameter recommendation error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)