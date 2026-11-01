# Why MLflow over DVC for experiment tracking:
# MLflow provides integrated tracking for parameters, metrics, model artifacts, and a
# model registry with versioning — all in one tool. DVC is primarily a data/pipeline
# versioning layer on top of Git and does not natively offer experiment comparison,
# a model registry, or a tracking UI. For this project, MLflow covers both experiment
# tracking and model version control in a single, self-contained workflow.

# %%
import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn
from model_training import ALL_MODELS, PROCESSED_FILE, train_and_evaluate

# %%
PROJECT_DIR = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_DIR / "mlflow.db"
MLRUNS_DIR = PROJECT_DIR / "mlruns"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("titanic_experiment")
if experiment is None:
    client.create_experiment("titanic_experiment", artifact_location=str(MLRUNS_DIR))
mlflow.set_experiment("titanic_experiment")

# %%
parser = argparse.ArgumentParser(description="Track Titanic model experiments with MLflow")
parser.add_argument(
    "--model",
    choices=list(ALL_MODELS.keys()),
    default=None,
    help="Train only this model (default: train all three and pick the best)",
)
args, _ = parser.parse_known_args()

# %%
SKOPS_TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]

selected = [args.model] if args.model else None
results = train_and_evaluate(selected)

# %%
# Log each model's parameters, metrics, artifacts, and processed dataset to MLflow
tracked = {}

for name, result in results.items():
    classifier = result["classifier"]
    pipeline = result["pipeline"]
    metrics = result["metrics"]

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_metrics(metrics)

        if hasattr(classifier, "get_params"):
            mlflow.log_params(classifier.get_params())

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=f"{name}_classifier",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        # Track data changes: log the processed dataset as an artifact
        mlflow.log_artifact(str(PROCESSED_FILE), artifact_path="data")

        tracked[name] = {"run_id": run.info.run_id, "f1": metrics["f1_score"]}
        print(f"[{name}] Logged MLflow run {run.info.run_id}")

# %%
# Register the best model by F1-score in the MLflow Model Registry
best_name = max(tracked, key=lambda k: tracked[k]["f1"])
best_run_id = tracked[best_name]["run_id"]
best_f1 = tracked[best_name]["f1"]

print(f"\nBest model: {best_name} (F1: {best_f1:.4f})")
print(f"Registering {best_name} in the MLflow Model Registry...")

model_uri = f"runs:/{best_run_id}/{best_name}_classifier"
registered = mlflow.register_model(model_uri, "titanic_survival_model")
print(f"Registered model '{registered.name}' version {registered.version}")
