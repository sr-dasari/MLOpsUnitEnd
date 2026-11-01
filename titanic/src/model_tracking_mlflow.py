# %%
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# %%
# Use SQLite backend in project root
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("titanic_experiment")

# %%
# Locate the input and output CSV files in the project's data directory.
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
df = pd.read_csv(DATA_DIR / 'processed_titanic.csv')

# %%
# Separate features (X) and target (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Identify numerical columns that need scaling
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

# %%
# Check the shape of y
print(f"Training survived shape: {y_train.shape}, Testing survived shape: {y_test.shape}")

# %%
# Define models to train and evaluate
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

# Define trusted types for skops tree serialization
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

# Track results across models to select the best one
results = {}

# Train, evaluate, and log each model in its own MLflow run
for name, classifier in models.items():
    # Bundle the scaler with the classifier so it is fitted on the training data
    # only and is saved and loaded as part of the model.
    scaling = ColumnTransformer(
        transformers=[("numerical", StandardScaler(), numerical_cols)],
        remainder="passthrough"
    )
    model = Pipeline(steps=[("scaling", scaling), ("classifier", classifier)])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate with Accuracy, Precision, Recall, and F1-score
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"\n[{name}] Confusion matrix (rows=actual, columns=predicted):")
    print("                 Predicted 0   Predicted 1")
    print(f"Actual 0         {matrix[0, 0]:<14} {matrix[0, 1]}")
    print(f"Actual 1         {matrix[1, 0]:<14} {matrix[1, 1]}")
    print(f"Accuracy: {acc:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        if hasattr(classifier, "get_params"):
            mlflow.log_params(classifier.get_params())

        # Pass skops_trusted_types and use name instead of deprecated artifact_path
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES
        )

        results[name] = {"run_id": run.info.run_id, "f1": f1}
        print(f"[{name}] Logged run {run.info.run_id}")

# %%
# Select the best-performing model based on F1-score and register it
best_name = max(results, key=lambda k: results[k]["f1"])
best_run_id = results[best_name]["run_id"]
best_f1 = results[best_name]["f1"]

print(f"\nBest model: {best_name} (F1: {best_f1:.4f})")
print(f"Registering {best_name} in the MLflow Model Registry...")

model_uri = f"runs:/{best_run_id}/model"
registered = mlflow.register_model(model_uri, "titanic_survival_model")
print(f"Registered model '{registered.name}' version {registered.version}")
