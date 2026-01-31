# Flask GCP MLOps Lab (Enhanced)

A production-ready Flask API for Iris species prediction, enhanced with MLOps best practices, automated testing, and CI/CD pipelines.

## 🚀 Features

### Core Enhancements
- **Robust Input Validation**: Integrated `pydantic` to validate API requests, ensuring strict type checking and helpful error messages.
- **Production-Grade Logging**: Replaced standard prints with `structlog` for structured JSON logging, making the app observability-ready.
- **Model Bundling**: Enhanced training pipeline to bundle metadata (class names, accuracy) with the model artifact, eliminating hardcoded labels.
- **Resilient Architecture**: Implemented a `ModelService` with lazy loading and error handling for robust inference.

### DevSecOps & Automation
- **Automated Testing**: Comprehensive `pytest` suite covering both unit logic and API integration points.
- **CI/CD Pipeline**: GitHub Actions workflow (`.github/workflows/ci.yml`) to automatically run linting and tests on every push.
- **Developer Experience**: A `Makefile` simplifying common lifecycle commands (`install`, `train`, `test`, `run`).
- **Environment Management**: Configuration via `.env` files for secure and flexible deployment settings.

## 🛠️ Quick Start

### Prerequisites
- Python 3.9+
- `virtualenv` (recommended)

### Installation
1.  **Clone the repository**
2.  **Setup Environment**:
    ```bash
    make venv
    make install
    ```

### Training the Model
Train the Random Forest model and generate the artifact bundle (`model/model.pkl`):
```bash
make train
```

### Running the API
Start the Flask development server (default port: 8080):
```bash
make run
```
Test the endpoint:
```bash
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## 🧪 Testing
Run the full regression test suite:
```bash
make test
```

## 📂 Project Structure
```text
.
├── .github/workflows/   # CI/CD definitions
├── model/               # Trained artifacts
├── src/
│   ├── main.py          # Flask API Entrypoint
│   ├── train.py         # Training Pipeline
│   └── predict.py       # Inference Logic
├── tests/               # Pytest Suite
├── Makefile             # Command Shortcuts
└── requirements.txt     # Pinned Dependencies
```
