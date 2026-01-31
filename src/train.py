import os
import joblib
import structlog
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logger = structlog.get_logger()

def run_training():
    log = logger.bind(job="training")
    log.info("starting_training")

    # 1. Load dataset (Iris)
    iris = load_iris()
    X = iris.data   # shape (150, 4)
    y = iris.target # 0,1,2 classes
    target_names = iris.target_names.tolist() # ['setosa', 'versicolor', 'virginica']

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info("data_split", train_size=len(X_train), test_size=len(X_test))

    # 3. Train a model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    log.info("model_trained", accuracy=acc)

    # 5. Save Model Bundle (Model + Metadata)
    os.makedirs("model", exist_ok=True)
    
    bundle = {
        "model": model,
        "target_names": target_names,
        "metadata": {
            "accuracy": acc,
            "model_type": "RandomForestClassifier"
        }
    }
    
    model_path = "model/model.pkl"
    joblib.dump(bundle, model_path)
    log.info("model_saved", path=model_path)

if __name__ == "__main__":
    run_training()