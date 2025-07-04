from flask import Flask, request, jsonify
import os
import numpy as np
import torch
import torch.nn.functional as F
import traceback
import sys
import socket
import onnxruntime as ort
import logging
from transformers import AutoFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
MODELS = {
    "alpha": {
        "path": "my_model_alpha.onnx",
        "requires_feature_extractor": True,
        "feature_extractor_name": "mahmoudmamdouh13/ast-finetuned-en-alphabets",
        "id2label": {
            0: "A", 1: "B", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q",
            17: "R", 18: "S", 19: "T", 2: "C", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
            25: "Z", 26: "_silence_", 27: "_unknown_", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
            8: "I", 9: "J"
        }
    },
    "word": {
        "path": "my_model_word.onnx",
        "requires_feature_extractor": True,
        "feature_extractor_name": "mahmoudmamdouh13/ast-mlcommons-speech-commands",
        "id2label": {
            0: "_silence_", 1: "_unknown_", 10: "four", 11: "fox", 12: "green",
            13: "horse", 14: "nine", 15: "one", 16: "red", 17: "seven", 18: "six",
            19: "ten", 2: "black", 20: "three", 21: "two", 22: "white", 3: "blue",
            4: "brown", 5: "cat", 6: "dog", 7: "eight", 8: "fish", 9: "five"
        }
    }
}

app = Flask(__name__)

# Global variables to store the models and feature extractor
LOADED_MODELS = {}
FEATURE_EXTRACTORS = {}

def load_feature_extractor(model_type):
    """
    Load the feature extractor for the specified model
    """
    if model_type not in MODELS:
        logger.error(f"Unknown model type: {model_type}")
        return None
        
    feature_extractor_name = MODELS[model_type]["feature_extractor_name"]
    logger.info(f"Loading feature extractor for {model_type}: {feature_extractor_name}")
    
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)
        logger.info(f"Feature extractor for {model_type} loaded successfully")
        return feature_extractor
    except Exception as e:
        logger.error(f"Error loading feature extractor for {model_type}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def load_model(model_type):
    """
    Load and return the specified model
    """
    if model_type not in MODELS:
        logger.error(f"Unknown model type: {model_type}")
        return None
        
    model_path = MODELS[model_type]["path"]
    logger.info(f"Loading model: {model_path}")
    
    try:
        model = ort.InferenceSession(model_path)
        logger.info(f"Model {model_type} loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_type}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def load_all_models():
    """
    Load all models and their respective feature extractors
    """
    models = {}
    feature_extractors = {}
    
    for model_type in MODELS:
        models[model_type] = load_model(model_type)
        
        if MODELS[model_type]["requires_feature_extractor"]:
            feature_extractors[model_type] = load_feature_extractor(model_type)
    
    return models, feature_extractors

# Update the initialization
LOADED_MODELS, FEATURE_EXTRACTORS = load_all_models()

def process_audio_for_model(audio_data, model_type):
    """
    Process audio data based on the model requirements
    """
    if model_type not in MODELS:
        raise ValueError(f"Unknown model type: {model_type}")
        
    if FEATURE_EXTRACTORS.get(model_type) is not None:
        try:
            sample_rate = 16000  # Assuming 16kHz sample rate
            inputs = FEATURE_EXTRACTORS[model_type](
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="np"
            )
            processed_audio = inputs.input_values[0]
            return processed_audio
        except Exception as e:
            logger.error(f"Error processing audio with feature extractor for {model_type}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    else:
        # If feature extractor failed to load
        return audio_data

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to process audio and make predictions with the selected voice model.
    """
    try:
        if not request.json or 'audio_data' not in request.json:
            return jsonify({'error': 'Missing audio_data in JSON request'}), 400
            
        # Get the model type from request, default to "alpha" if not specified
        model_type = request.json.get('model_type', 'alpha')
        
        if model_type not in MODELS:
            return jsonify({'error': f'Unknown model type: {model_type}. Available models: {list(MODELS.keys())}'}), 400
            
        if LOADED_MODELS.get(model_type) is None:
            return jsonify({'error': f'Model {model_type} failed to load. Check server logs.'}), 500
            
        # Process raw numpy array from JSON
        audio_data = np.array(request.json['audio_data'], dtype=np.float32)
        
        logger.info(f"Received audio data array of length {len(audio_data)} for model {model_type}")
        
        # Update this line to pass the model_type
        processed_audio = process_audio_for_model(audio_data, model_type)
        
        # Make prediction with the selected model
        logger.info(f"Running inference on audio data with model {model_type}")
        model = LOADED_MODELS[model_type]
        
        model_output = model.run(None, {"input": [processed_audio]})[0][0]
        model_output = F.softmax(torch.tensor(model_output, dtype=torch.float32))
        
        predicted_idx = model_output.argmax().item()
        confidence = model_output.max().item()
        
        # Get label using the appropriate id2label mapping
        predicted_label = MODELS[model_type]["id2label"].get(predicted_idx, "Unknown")
        
        result = {
            "predictions": {predicted_label: confidence},
            "audio_length": len(audio_data),
            "model_type": model_type
        }
        
        logger.info(f"Prediction successful: {predicted_label} with confidence {confidence:.4f}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/models', methods=['GET'])
def list_models():
    """Endpoint to list available models and their status"""
    model_status = {}
    for model_type in MODELS:
        status = "loaded" if LOADED_MODELS.get(model_type) is not None else "failed to load"
        
        # Check feature extractor status for this specific model
        feature_extractor_status = "not required"
        if MODELS[model_type]["requires_feature_extractor"]:
            feature_extractor_status = "loaded" if FEATURE_EXTRACTORS.get(model_type) is not None else "failed to load"
        
        model_status[model_type] = {
            "status": status,
            "path": MODELS[model_type]["path"],
            "labels_count": len(MODELS[model_type]["id2label"]),
            "feature_extractor": {
                "name": MODELS[model_type].get("feature_extractor_name"),
                "status": feature_extractor_status
            }
        }
    
    return jsonify({
        'models': model_status
    })


@app.route('/health', methods=['GET'])
def health_check():
    models_status = {
        model_type: "loaded" if model is not None else "failed to load"
        for model_type, model in LOADED_MODELS.items()
    }
    
    feature_extractors_status = {
        model_type: "loaded" if FEATURE_EXTRACTORS.get(model_type) is not None else "failed to load"
        for model_type in MODELS if MODELS[model_type]["requires_feature_extractor"]
    }
    
    return jsonify({
        'status': 'healthy', 
        'models': models_status,
        'feature_extractors': feature_extractors_status
    })

def check_port_available(port, host='127.0.0.1'):
    """Check if the port is available on the host"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to the Voice Prediction API!',
        'endpoints': {
            '/health': 'Health check endpoint',
            '/models': 'List available models and their status',
            '/predict': 'POST endpoint for predictions (requires model_type and audio_data)'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    host = '0.0.0.0'
    
    # Check if the port is available
    if not check_port_available(port, '127.0.0.1'):
        logger.warning(f"Port {port} is already in use!")
        
        # Try to find an available port
        for test_port in range(5001, 5010):
            if check_port_available(test_port, '127.0.0.1'):
                logger.info(f"Found available port: {test_port}")
                port = test_port
                break
        else:
            logger.error("Could not find an available port. Please close the application using port 5000.")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Starting Flask server on http://{host}:{port}")
    print(f"To test the API health: http://localhost:{port}/health")
    print(f"To list available models: http://localhost:{port}/models")
    print(f"To make predictions, send POST requests to: http://localhost:{port}/predict")
    print("Include 'model_type': 'alpha' or 'word' in your request to select the model")
    print(f"{'='*60}\n")
    
    # Start the Flask server
    app.run(host=host, port=port, debug=True, use_reloader=False)