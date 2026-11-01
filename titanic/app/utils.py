import sys
import mlflow
import mlflow.sklearn
import pandas as pd


def select_model(experiment_name: str = "titanic_experiment", interactive: bool = False):
    """Loads the latest registered model, or falls back to interactive selection."""
    client = mlflow.tracking.MlflowClient()

    if not interactive:
        model_name = "titanic_survival_model"
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                run_id = latest.run_id
                print(f"\nLoading registered model '{model_name}' version {latest.version} (run {run_id[:8]})...")
                model_uri = f"models:/{model_name}/{latest.version}"
                loaded_model = mlflow.sklearn.load_model(model_uri)
                return loaded_model, run_id
        except Exception:
            pass

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found in mlflow.db.")
        sys.exit(1)

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs:
        print("No logged model runs found in experiment.")
        sys.exit(1)

    print(f"\n--- Available Models in MLflow ({experiment_name}) ---")
    for idx, run in enumerate(runs):
        run_id = run.info.run_id
        model_type = run.data.params.get("model_type", "Unknown")
        accuracy = run.data.metrics.get("accuracy", "N/A")
        f1 = run.data.metrics.get("f1_score", "N/A")
        roc_auc = run.data.metrics.get("roc_auc", "N/A")
        print(f"[{idx}] Run ID: {run_id} | Type: {model_type} | "
              f"Accuracy: {accuracy} | F1: {f1} | ROC-AUC: {roc_auc}")

    choice = int(input("\nSelect model index to load: "))
    selected_run = runs[choice]
    run_id = selected_run.info.run_id
    model_type = selected_run.data.params.get("model_type", "model")

    print(f"\nLoading model from Run ID: {run_id}...")
    model_uri = f"runs:/{run_id}/{model_type}_classifier"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    return loaded_model, run_id


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive engineered features from raw passenger data to match the training pipeline."""
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")
        title_map = {
            "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
            "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Sir": "Rare", "Lady": "Rare", "Capt": "Rare", "Don": "Rare",
            "Dona": "Rare", "Countess": "Rare", "Jonkheer": "Rare",
        }
        df["Title"] = df["Title"].map(title_map).fillna("Rare")

    df["FamilySize"] = df.get("SibSp", 0) + df.get("Parch", 0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Explicit one-hot encoding (get_dummies with drop_first is unreliable on single rows)
    encoding_map = {
        "Sex": {"male": {"Sex_male": 1}},
        "Embarked": {"Q": {"Embarked_Q": 1}, "S": {"Embarked_S": 1}},
        "Title": {
            "Miss": {"Title_Miss": 1}, "Mr": {"Title_Mr": 1},
            "Mrs": {"Title_Mrs": 1}, "Rare": {"Title_Rare": 1},
        },
    }
    for col, mapping in encoding_map.items():
        if col in df.columns:
            val = df[col].iloc[0]
            for encoded_col in mapping.get(val, {}):
                df[encoded_col] = mapping[val][encoded_col]

    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin", "Sex", "Embarked", "Title"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def align_features(df: pd.DataFrame, loaded_model) -> pd.DataFrame:
    """Ensure exact feature presence and order matching the model schema."""
    if hasattr(loaded_model, "feature_names_in_"):
        expected_cols = list(loaded_model.feature_names_in_)
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]
    return df
