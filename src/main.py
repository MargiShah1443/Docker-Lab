# src/main.py
"""
Docker ML Lab - Breast Cancer Classifier

This script:
1. Loads the Breast Cancer dataset from scikit-learn
2. Splits the data into train and test sets
3. Builds a Pipeline with StandardScaler + LogisticRegression
4. Trains the model
5. Evaluates it on the test set
6. Saves the trained model and metrics to disk
"""

from pathlib import Path
import json

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


def train_and_evaluate(random_state: int = 42, test_size: float = 0.2):
    """Train a Logistic Regression classifier on the Breast Cancer dataset."""

    # 1. Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    target_names = list(data.target_names)

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 3. Build model pipeline
    # StandardScaler helps LogisticRegression converge and improves performance
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )

    # 4. Train model
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on test set
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )

    return pipeline, acc, report


def save_artifacts(model, accuracy: float, report: dict, artifacts_dir: str = "artifacts"):
    """
    Save the trained model and evaluation metrics to disk.

    - model -> artifacts/breast_cancer_model.pkl
    - metrics -> artifacts/metrics.json
    """
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_path / "breast_cancer_model.pkl"
    metrics_path = artifacts_path / "metrics.json"

    # Save model with joblib
    joblib.dump(model, model_path)

    # Prepare a summary of metrics
    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    print("Starting model training for Breast Cancer classification…")

    model, accuracy, report = train_and_evaluate()

    print(f"Training complete. Test Accuracy: {accuracy:.4f}")
    print("Classification report (summary):")
    # Print only accuracy + per-class F1 scores as a compact view
    for label, stats in report.items():
        if label in ("benign", "malignant"):
            print(f"  - {label}: F1 = {stats['f1-score']:.4f}")
    print()

    save_artifacts(model, accuracy, report)

    print("Docker ML lab run finished successfully.")
