# Titanic Survival Prediction — MLOps Pipeline

An end-to-end MLOps project that preprocesses the Titanic dataset, trains and compares multiple classifiers, tracks experiments with MLflow, serves predictions via a FastAPI REST API, and monitors data drift and model performance using Evidently AI.

> **Tested end-to-end on Windows using Git Bash.**

## Project Structure

```
MLOpsUnitEnd/
├── README.md                        # Instructions to fllow.
└── titanic/
    ├── data/
    │   ├── titanic.csv              # Raw Titanic dataset
    │   └── processed_titanic.csv    # Cleaned & feature-engineered dataset
    ├── src/
    │   ├── data_preprocessing.py    # Data cleaning, outlier handling, feature engineering
    │   ├── model_training.py        # Model training & evaluation (standalone)
    │   ├── model_tracking_mlflow.py # MLflow experiment tracking & model registry
    │   └── generate_report.py       # Drift, quality & performance monitoring reports
    ├── app/
    │   ├── main.py                  # FastAPI REST API for predictions
    │   ├── utils.py                 # Model loading & feature engineering helpers
    │   └── test_api.py              # Automated API tests (pytest)
    ├── reports/                     # Generated HTML monitoring reports
    ├── clean_slate.py               # Reset project to clean state
    ├── pipeline.py                  # End-to-end pipeline orchestrator
    ├── trigger.py                   # Watchdog-based file watcher for auto-triggering
    ├── requirements.txt             # Python dependencies
    ├── .gitignore                   # Git ignore rules
    └── venv/                        # Virtual environment (created during setup)
```

## Project Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
cd titanic
```

> **Note:** All commands in this README are run from inside the `titanic/` directory. Make sure you `cd titanic` before proceeding.

```bash
python -m venv venv
```

Activate the virtual environment:

- **Windows (PowerShell)**: `venv\Scripts\activate`
- **Windows (Git Bash)**: `source venv/Scripts/activate`
- **macOS / Linux**: `source venv/bin/activate`

Then install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset

The raw Titanic dataset is already included at `data/titanic.csv` — no download required. It contains 891 rows with columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.

## Quick Start — Full Pipeline

The fastest way to run everything end-to-end is with the pipeline orchestrator. It chains preprocessing, training, MLflow tracking, monitoring, and API tests in one command:

```bash
python clean_slate.py --yes    # optional: reset to a clean state first
python pipeline.py             # run all stages end-to-end
```

Pipeline options:

```bash
python pipeline.py --skip-api  # skip API tests
python pipeline.py --check     # CI mode: exit code 1 if monitoring alerts fire
```

The pipeline runs these stages in order:

1. **Data Preprocessing** → `src/data_preprocessing.py`
2. **Model Training & MLflow Tracking** → `src/model_tracking_mlflow.py`
3. **Monitoring Reports** → `src/generate_report.py`
4. **API Tests** → `app/test_api.py` (optional)

If any stage fails, the pipeline stops immediately with a non-zero exit code.

If you prefer to run each step individually, or want to understand what each stage does, continue reading below.

---

## Step-by-Step Guide

### Step 0: Clean Slate (Optional but Recommended)

If you want a fresh start — or if you're picking up after a partial run — reset everything first:

```bash
python clean_slate.py       # interactive: shows what will be removed, asks for confirmation
python clean_slate.py --yes # non-interactive: skips confirmation (for automation)
```

This removes all generated artifacts (processed data, MLflow database & runs, monitoring reports, Python cache) while preserving the raw dataset and all source code.

### Step 1: Data Preprocessing

```bash
python src/data_preprocessing.py
```

This script:
- Loads and explores the raw Titanic dataset
- Handles missing values (Age → median, Embarked → mode)
- Engineers features: Title (from Name), FamilySize, IsAlone
- Handles outliers via IQR capping on Age and Fare
- One-hot encodes categorical variables (Sex, Embarked, Title)
- Saves processed data to `data/processed_titanic.csv`

### Step 2: Model Training & Evaluation

```bash
python src/model_training.py                    # Train all 3 models
python src/model_training.py --model XGBoost    # Train a specific model
```

Trains and compares three classifiers:

| Model | Description |
|---|---|
| **Random Forest** | Ensemble of decision trees (default winner) |
| **XGBoost** | Gradient-boosted trees with regularization |
| **Logistic Regression** | Linear baseline model |

Each model is evaluated with: Accuracy, Precision, Recall, F1-score, and ROC-AUC.

### Step 3: MLflow Experiment Tracking

```bash
python src/model_tracking_mlflow.py                    # Track all models
python src/model_tracking_mlflow.py --model XGBoost    # Track a specific model
```

For each model run, MLflow logs:
- **Parameters**: model type, all hyperparameters
- **Metrics**: accuracy, precision, recall, f1_score, roc_auc
- **Artifacts**: serialized model pipeline, processed dataset snapshot
- **Registry**: best model (by F1-score) is registered as `titanic_survival_model` with auto-incrementing version numbers

**Why MLflow over DVC**: MLflow provides integrated tracking for parameters, metrics, model artifacts, and a model registry with versioning — all in one tool. DVC is primarily a data/pipeline versioning layer and does not natively offer experiment comparison, a model registry, or a tracking UI.

To browse experiments locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

This starts a web UI at `http://127.0.0.1:5000`. The terminal will remain occupied — open a **new terminal** (with the venv activated) to continue running other commands.

### Step 4: Deploy the REST API

```bash
python app/main.py                  # Auto-loads latest registered model
python app/main.py --select         # Interactively pick a model from MLflow
```

The API runs at `http://127.0.0.1:8000`. Open `http://127.0.0.1:8000/docs` in your browser to access the interactive Swagger UI where you can test the API directly.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check, shows loaded model run ID |
| `/predict` | POST | Predict survival for a passenger |
| `/docs` | GET | Interactive Swagger UI with full schema documentation |

**Example request:**

```json
POST /predict
{
    "Pclass": 1,
    "Sex": "female",
    "Age": 29,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 211.34,
    "Embarked": "S",
    "Name": "Smith, Mrs. John"
}
```

**Example response:**

```json
{
    "prediction": 1,
    "label": "survived"
}
```

**Input Validation**: All fields are validated via Pydantic with constraints (e.g., Pclass 1-3, Age 0-120, Sex must be "male"/"female", Embarked must be "C"/"Q"/"S"). Invalid input returns a 422 with descriptive error messages.

### Step 5: Run API Tests

> The API server (`main.py`) occupies the terminal. Open a **new terminal** and run `cd titanic` followed by the venv activation command (e.g., `source venv/Scripts/activate` on Git Bash) before running tests or other commands.

```bash
python -m pytest app/test_api.py -v
```

11 automated tests covering:
- Happy path predictions (survived, did not survive, optional Name)
- Input validation (invalid Pclass, Sex, Age, Embarked, Fare, missing fields, empty body)

## CI/CD Pipeline Configuration

The full pipeline (`pipeline.py`) is documented in the [Quick Start](#quick-start--full-pipeline) section above.

### Auto-Trigger on File Changes

```bash
python trigger.py                  # Watch data/ and src/ for changes
python trigger.py --cooldown 60   # 60 seconds between pipeline runs
python trigger.py --once           # Run pipeline on next change, then exit
```

Uses the `watchdog` library to monitor `data/` and `src/` directories. When a `.csv` or `.py` file is created or modified, the full pipeline runs automatically. A configurable cooldown prevents rapid re-runs.

To test the trigger, open a **separate terminal** and run:

```bash
cd titanic
touch data/titanic.csv
```

The trigger terminal should detect the change and start the pipeline automatically.

## Monitoring Strategy

### Report Generation

```bash
python src/generate_report.py           # Generate HTML reports and open in browser
python src/generate_report.py --check   # Headless: print alerts, exit 1 on issues
```

Three HTML reports are generated in the `reports/` directory:

| Report | Purpose |
|---|---|
| `quality_report.html` | Dataset overview: column types, missing values, statistics |
| `monitoring_report.html` | Feature-level drift analysis (reference vs. current data) |
| `performance_report.html` | Model metrics vs. thresholds on current data |

### Drift Detection

Uses **Evidently AI** with `DataDriftPreset`:
- Numerical features: Wasserstein distance
- Categorical features: Jensen-Shannon distance
- Features are flagged when their score exceeds the per-feature threshold

### Performance Monitoring

The latest registered model is evaluated on the current data split:

| Metric | Threshold |
|---|---|
| Accuracy | >= 0.75 |
| Precision | >= 0.70 |
| Recall | >= 0.65 |
| F1-score | >= 0.70 |
| ROC-AUC | >= 0.75 |

### Alerting

Console alerts fire when:
- Any feature's drift score exceeds **0.3** (high drift threshold)
- Any model metric falls below its threshold

In `--check` mode, the script exits with code 1 if any alerts fire, making it suitable for CI/scheduled automation.

## Tools and Configurations

| Tool | Purpose | Version |
|---|---|---|
| **pandas** | Data manipulation and preprocessing | < 3.0 |
| **scikit-learn** | Model training (Random Forest, Logistic Regression), metrics, pipelines | latest |
| **XGBoost** | Gradient-boosted tree classifier | latest |
| **MLflow** | Experiment tracking, model registry, artifact storage | latest |
| **FastAPI** | REST API framework for model serving | latest |
| **Pydantic** | Request/response validation with type constraints | latest |
| **uvicorn** | ASGI server for FastAPI | latest |
| **Evidently AI** | Data drift detection and monitoring reports | latest |
| **watchdog** | Filesystem monitoring for auto-triggering pipeline | latest |
| **pytest** | Automated API testing | latest |

**MLflow storage**: SQLite backend (`mlflow.db`) — no external database needed. All artifacts stored locally in `mlruns/`.

## Best Practices for Reproducibility and Scalability

### Reproducibility

- **Virtual environment**: A dedicated `venv/` isolates project dependencies from the system Python, ensuring consistent behavior across machines
- **Fixed random seeds**: All models use `random_state=42`; train/test split is deterministic
- **Data versioning**: Each MLflow run logs the processed dataset as an artifact, so every model version can be traced to the exact data it was trained on
- **Full pipeline serialization**: The entire sklearn Pipeline (scaler + classifier) is saved as one artifact, ensuring preprocessing is always applied consistently at inference time
- **Pinned dependencies**: `requirements.txt` pins critical versions (e.g., `pandas<3.0`) to avoid compatibility issues
- **Deterministic preprocessing**: Feature engineering steps (title extraction, encoding maps) use explicit mappings rather than data-dependent operations

### Scalability

- **Modular architecture**: Training (`model_training.py`) is separated from tracking (`model_tracking_mlflow.py`), so either can be replaced or extended independently
- **Model registry**: MLflow's model registry supports version management — roll back to any previous model version without retraining
- **API-first deployment**: FastAPI with Pydantic schemas provides a production-ready serving layer with auto-generated documentation and input validation
- **Configurable thresholds**: Drift and performance thresholds are defined as constants at the top of `src/generate_report.py`, easily adjustable without code changes
- **Pipeline extensibility**: `pipeline.py` stages are a simple list — add new stages (e.g., data validation, A/B testing) by appending to the `STEPS` list
- **Trigger-based automation**: `trigger.py` watches for file changes locally; for production, replace with CI/CD webhooks (GitHub Actions, Jenkins) using the same `pipeline.py` entry point
