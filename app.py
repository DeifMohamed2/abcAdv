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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
model_path = "my_model.onnx"
idtolabel = {
    0: "A",
    1: "B",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "Other",
    16: "P",
    17: "Q",
    18: "R",
    19: "S",
    2: "C",
    20: "T",
    21: "U",
    22: "V",
    23: "W",
    24: "X",
    25: "Y",
    26: "Z",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J"
  }



app = Flask(__name__)

# Load your model
def load_model():
    """
    Load and return the voice model from Hugging Face.
    """
    logger.info(f"Loading model: {model_path}")
    try:
        model = ort.InferenceSession(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        # Return None so the app can still start, and we'll handle the model error during prediction
        return None

# Global variable to store the model
MODEL = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to process audio and make predictions with the voice model.
    Optimized for numpy array input from JSON.
    """
    try:
        if MODEL is None:
            return jsonify({'error': 'Model failed to load. Check server logs.'}), 500
            
        if not request.json or 'audio_data' not in request.json:
            return jsonify({'error': 'Missing audio_data in JSON request'}), 400
            
        # Process raw numpy array from JSON
        audio_data = np.array(request.json['audio_data'], dtype=np.float32)
        
        logger.info(f"Received audio data array of length {len(audio_data)}")
        
        # Make prediction with the model
        logger.info("Running inference on audio data")
        model_output = MODEL.run(None, {"input": [audio_data]})[0][0]
        model_output = F.softmax(torch.tensor(model_output, dtype=torch.float32))
        predicted_class = {idtolabel[model_output.argmax().item()]: model_output.max().item()}
        
        result = {
            "predictions": predicted_class,
            "audio_length": len(audio_data),
        }
        
        logger.info(f"Prediction successful: {predicted_class}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    model_status = "loaded" if MODEL is not None else "failed to load"
    return jsonify({
        'status': 'healthy', 
        'model_status': model_status,
        'model_name': model_path
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
            '/predict': 'POST endpoint for predictions'
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
    print(f"To make predictions, send POST requests to: http://localhost:{port}/predict")
    print("Server configured for numpy array input only")
    print(f"{'='*60}\n")
    
    # Start the Flask server
    app.run(host=host, port=port, debug=True, use_reloader=False)