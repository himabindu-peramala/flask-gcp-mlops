import pytest
import os
import sys

# Ensure src path is available (sometimes conftest fixture isn't enough for raw imports in test body if not running via module)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from predict import ModelService

def test_model_service_load():
    # Ensure model exists (we assume 'make train' ran before 'make test')
    if not os.path.exists("model/model.pkl"):
        pytest.skip("Model not found, run 'make train' first")
        
    service = ModelService("model/model.pkl")
    service.load_model()
    assert service.model is not None
    assert isinstance(service.target_names, list)
    assert len(service.target_names) == 3

def test_model_service_prediction():
    if not os.path.exists("model/model.pkl"):
        pytest.skip("Model not found")
        
    service = ModelService("model/model.pkl")
    # Setosa typically small petal (length ~1.4, width ~0.2)
    prediction = service.predict([5.1, 3.5, 1.4, 0.2])
    assert prediction in ['setosa', 'versicolor', 'virginica']
