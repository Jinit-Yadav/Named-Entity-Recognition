import sys
import os

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
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
    logger.info("NER prediction pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize NER prediction pipeline: {e}")
    pipeline = None

def validate_text(text: str) -> tuple[bool, str]:
    """Validate input text for NER"""
    if not text or len(text.strip()) == 0:
        return False, "Please enter some text to analyze."
    
    if len(text.strip()) < 10:
        return False, "Text is too short. Please provide at least 10 characters."
    
    # Check if text has meaningful content (not just special characters)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if len(words) < 3:
        return False, "Text doesn't contain enough meaningful content."
    
    return True, ""

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    entities = None
    original_text = ""
    processing_time = 0
    word_count = 0
    char_count = 0
    error_message = ""
    entity_statistics = None
    
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
                confidence_threshold = float(request.form.get("confidence_threshold", 0.7))
                max_entities = int(request.form.get("max_entities", 50))
                
                # Validate ranges
                confidence_threshold = max(0.1, min(1.0, confidence_threshold))
                max_entities = max(1, min(200, max_entities))
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid parameter values: {e}")
                # Use recommended parameters
                recommended = pipeline.get_recommended_parameters(len(text))
                confidence_threshold = recommended["confidence_threshold"]
                max_entities = recommended["max_entities"]
            
            if pipeline is None:
                error_message = "NER service is currently unavailable. Please try again later."
            else:
                # Prepare prediction parameters
                gen_kwargs = {
                    "confidence_threshold": confidence_threshold,
                    "max_entities": max_entities,
                }
                
                # Process text for NER
                entities = pipeline.predict(text, **gen_kwargs)
                processing_time = round(time.time() - start_time, 2)
                
                # Get entity statistics
                if entities:
                    entity_statistics = pipeline.analyze_entity_statistics(entities)
                
                logger.info(f"NER completed in {processing_time} seconds, found {len(entities) if entities else 0} entities")
    
    return render_template(
        "home.html", 
        entities=entities, 
        original_text=original_text,
        processing_time=processing_time,
        word_count=word_count,
        char_count=char_count,
        error_message=error_message,
        entity_statistics=entity_statistics
    )

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """API endpoint for NER analysis"""
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
            "confidence_threshold": float(params.get("confidence_threshold", 0.7)),
            "max_entities": int(params.get("max_entities", 50)),
            "return_confidence": bool(params.get("return_confidence", True))
        }
        
        # Truncate very long texts
        if len(text) > 5000:
            text = text[:5000]
        
        entities = pipeline.predict(text, **gen_kwargs)
        
        # Count entities by type
        entity_counts = {}
        if entities:
            for entity in entities:
                entity_type = entity.get('type', 'UNKNOWN')
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        return jsonify({
            "entities": entities,
            "entity_counts": entity_counts,
            "total_entities": len(entities) if entities else 0,
            "text_length": len(text)
        })
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/entity_types", methods=["GET"])
def api_entity_types():
    """API endpoint to get available entity types"""
    if pipeline is None:
        return jsonify({"error": "Service unavailable"}), 503
    
    try:
        entity_types = pipeline.get_entity_types()
        
        return jsonify({
            "entity_types": entity_types
        })
        
    except Exception as e:
        logger.error(f"Entity types error: {str(e)}")
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
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>404 - Page Not Found</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container mt-5">
            <div class="row justify-content-center">
                <div class="col-md-6 text-center">
                    <h1 class="display-1 text-muted">404</h1>
                    <h2 class="mb-4">Page Not Found</h2>
                    <p class="lead mb-4">The page you're looking for doesn't exist.</p>
                    <a href="{{ url_for('index') }}" class="btn btn-primary">Go Home</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """, 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>500 - Internal Server Error</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body class="bg-light">
        <div class="container mt-5">
            <div class="row justify-content-center">
                <div class="col-md-6 text-center">
                    <h1 class="display-1 text-muted">500</h1>
                    <h2 class="mb-4">Internal Server Error</h2>
                    <p class="lead mb-4">Something went wrong on our end. Please try again later.</p>
                    <a href="{{ url_for('index') }}" class="btn btn-primary">Go Home</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """, 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)