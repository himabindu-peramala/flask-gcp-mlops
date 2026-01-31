import pytest

def test_predict_endpoint_valid(client):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert data["prediction"] in ['setosa', 'versicolor', 'virginica']

def test_predict_endpoint_validation_error_missing_field(client):
    # Missing petal_width
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 422
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "Validation Error"

def test_predict_endpoint_validation_error_invalid_type(client):
    # String instead of float
    payload = {
        "sepal_length": "invalid",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 422

def test_predict_method_not_allowed(client):
    response = client.get('/predict')
    assert response.status_code == 405
