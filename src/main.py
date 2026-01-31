from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
import os
import structlog
from predict import ModelService

# Load env vars
load_dotenv()

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

app = Flask(__name__)

# Initialize Model Service
# We do this globally so it persists (and can lazy load on first request)
model_path = os.getenv("MODEL_PATH", "model/model.pkl")
model_service = ModelService(model_path)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.route('/predict', methods=['POST'])
def predict():
    log = logger.bind(job="api_request")
    
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        
        # Validation
        input_data = IrisInput(**data)
        
        # Prediction
        prediction = model_service.predict([
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ])
        
        return jsonify({'prediction': prediction})

    except ValidationError as e:
        log.warn("validation_failed", error=str(e))
        return jsonify({'error': 'Validation Error', 'details': e.errors()}), 422
        
    except FileNotFoundError as e:
        log.error("model_missing", error=str(e))
        return jsonify({'error': 'Model not initialized', 'details': 'Please run training first'}), 503

    except Exception as e:
        log.error("internal_error", error=str(e))
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("ENV", "development") == "development"
    app.run(debug=debug, host="0.0.0.0", port=port)