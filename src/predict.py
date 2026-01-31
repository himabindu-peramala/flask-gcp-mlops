import os
import joblib
import numpy as np
import structlog

logger = structlog.get_logger()

class ModelService:
    def __init__(self, model_path="model/model.pkl"):
        self.model_path = model_path
        self.model = None
        self.target_names = None
        self.metadata = None

    def load_model(self):
        if self.model is not None:
            return

        log = logger.bind(job="model_loading", path=self.model_path)
        
        if not os.path.exists(self.model_path):
            log.error("model_not_found")
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Run training first.")

        try:
            bundle = joblib.load(self.model_path)
            # Handle both old format (raw model) and new format (dict bundle) for backward compatibility if needed
            # But here we assume new format since we upgraded train.py
            if isinstance(bundle, dict) and "model" in bundle:
                self.model = bundle["model"]
                self.target_names = bundle.get("target_names", [])
                self.metadata = bundle.get("metadata", {})
                log.info("model_loaded", metadata=self.metadata)
            else:
                # Fallback for old simple save
                self.model = bundle
                self.target_names = ['setosa', 'versicolor', 'virginica'] # Fallback default
                log.warn("legacy_model_format_detected")
                
        except Exception as e:
            log.error("model_load_failed", error=str(e))
            raise e

    def predict(self, input_features: list):
        """
        input_features: list of floats [sepal_length, sepal_width, petal_length, petal_width]
        """
        self.load_model()
        
        log = logger.bind(job="prediction")
        try:
            payload = np.array([input_features])
            prediction_idx = self.model.predict(payload)[0]
            
            # Map index to name if we have target names
            label = str(prediction_idx)
            if self.target_names and 0 <= int(prediction_idx) < len(self.target_names):
                label = self.target_names[int(prediction_idx)]
                
            log.info("prediction_made", input=input_features, result=label)
            return label
        except Exception as e:
            log.error("prediction_failed", error=str(e))
            raise e

# Singleton instance (optional, or better to instantiate in main)
# We won't instantiate global here to allow main.py to handle lifecycle