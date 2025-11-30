# Docker ML Lab – Breast Cancer Classifier

This lab demonstrates how to **containerize a simple machine learning training script with Docker**.

Instead of using the classic Iris dataset and a Random Forest model, this lab uses:

- **Dataset:** Breast Cancer Wisconsin dataset (`sklearn.datasets.load_breast_cancer`)
- **Model:** Logistic Regression inside a `scikit-learn` `Pipeline` with `StandardScaler`
- **Outputs:**
  - Trained model saved as `artifacts/breast_cancer_model.pkl`
  - Evaluation metrics saved as `artifacts/metrics.json`

The goal is to show how model training can be **reproducibly executed inside a Docker container**, independent of your local Python environment.

---

## Repository Structure

```text
DOCKER-LAB/
├── dockerfile         # Docker definition for the training container
├── ReadMe.md          # This file
└── src/
    ├── main.py        # Training + evaluation script
    └── requirements.txt
````

---

## What the Lab Does

1. **Loads a dataset**

   The script uses the Breast Cancer dataset from `scikit-learn`, which contains 30 numeric features describing cell nuclei and a binary target: `malignant` vs `benign`.

2. **Splits the data**

   * 80% training data
   * 20% test data
   * Stratified split to preserve class balance

3. **Builds and trains a model**

   A `Pipeline` is created with:

   * `StandardScaler` – normalizes input features
   * `LogisticRegression` – a linear classifier, trained to distinguish between benign and malignant tumors

4. **Evaluates the model**

   * Computes **accuracy** on the test set
   * Generates a full **classification report** (precision, recall, F1-score per class)

5. **Saves artifacts**

   In an `artifacts/` folder created by the script:

   * `breast_cancer_model.pkl` – the trained Pipeline (scaler + classifier)
   * `metrics.json` – a JSON file with accuracy and the full classification report

   These artifacts can be used later for inference or analysis outside the container.

---

## Prerequisites

* [Git](https://git-scm.com/)
* [Docker](https://www.docker.com/) installed and running

You **do not** need a local Python environment for the lab – all dependencies are installed inside the container.

---

## Setup and Usage

### 1. Clone the Repository

```bash
git clone <your-github-repo-url>.git
cd DOCKER-LAB
```

> Replace `<your-github-repo-url>` with the URL of your fork on GitHub.

---

### 2. Build the Docker Image

From the project root (where the `dockerfile` is):

```bash
docker build -t docker-ml-lab .
```

* `docker-ml-lab` is the name/tag of the image (you can change it if you like).

---

### 3. Run the Container

```bash
docker run --rm docker-ml-lab
```

What happens when you run this:

1. Docker starts a container based on the `docker-ml-lab` image.

2. Inside the container, `python main.py` is executed.

3. The script trains the model, prints the accuracy and F1-scores, and saves:

   * `artifacts/breast_cancer_model.pkl`
   * `artifacts/metrics.json`

4. When finished, the container exits (and is removed because of `--rm`).

---

### 4. Persisting Artifacts to Your Host (Optional but Recommended)

By default, artifacts are saved **inside** the container’s filesystem.
To save them directly to your local machine, you can mount a volume:

```bash
# Linux / macOS
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" docker-ml-lab

# Windows PowerShell
docker run --rm -v "${PWD}/artifacts:/app/artifacts" docker-ml-lab
```

After the run, you will find on your host:

* `artifacts/breast_cancer_model.pkl`
* `artifacts/metrics.json`

---

## Verifying the Results Locally (Optional)

If you also have Python installed locally, you can quickly inspect the saved metrics:

```python
import json
from pathlib import Path

metrics_path = Path("artifacts/metrics.json")
with open(metrics_path, "r") as f:
    metrics = json.load(f)

print("Test Accuracy:", metrics["accuracy"])
print("Classes in report:", list(metrics["classification_report"].keys()))
```

## Running Everything in One Go

Typical workflow:

```bash
# 1. Clone your repo
git clone <your-github-repo-url>.git
cd DOCKER-LAB

# 2. Build image
docker build -t docker-ml-lab .

# 3. Run container and save artifacts to host
docker run --rm -v "${PWD}/artifacts:/app/artifacts" docker-ml-lab
```
