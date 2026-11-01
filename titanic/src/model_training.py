# %%
import argparse
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# %%
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
PROCESSED_FILE = DATA_DIR / 'processed_titanic.csv'
NUMERICAL_COLS = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

ALL_MODELS = {
    "RandomForest": lambda: RandomForestClassifier(random_state=42),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=42,
    ),
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
}


def load_data(path=PROCESSED_FILE):
    df = pd.read_csv(path)
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_pipeline(classifier):
    scaling = ColumnTransformer(
        transformers=[("numerical", StandardScaler(), NUMERICAL_COLS)],
        remainder="passthrough",
    )
    return Pipeline(steps=[("scaling", scaling), ("classifier", classifier)])


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return metrics, matrix


def train_and_evaluate(model_names=None):
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")

    names = model_names or list(ALL_MODELS.keys())
    results = {}

    for name in names:
        classifier = ALL_MODELS[name]()
        pipeline = build_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        metrics, matrix = evaluate(pipeline, X_test, y_test)

        print(f"\n[{name}] Confusion matrix (rows=actual, columns=predicted):")
        print("                 Predicted 0   Predicted 1")
        print(f"Actual 0         {matrix[0, 0]:<14} {matrix[0, 1]}")
        print(f"Actual 1         {matrix[1, 0]:<14} {matrix[1, 1]}")
        print(f"Accuracy: {metrics['accuracy']:.4f}  Precision: {metrics['precision']:.4f}  "
              f"Recall: {metrics['recall']:.4f}  F1: {metrics['f1_score']:.4f}  ROC-AUC: {metrics['roc_auc']:.4f}")

        results[name] = {
            "classifier": classifier,
            "pipeline": pipeline,
            "metrics": metrics,
        }

    return results


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Titanic survival models")
    parser.add_argument(
        "--model",
        choices=list(ALL_MODELS.keys()),
        default=None,
        help="Train only this model (default: train all three and pick the best)",
    )
    args, _ = parser.parse_known_args()

    selected = [args.model] if args.model else None
    results = train_and_evaluate(selected)

    best = max(results, key=lambda k: results[k]["metrics"]["f1_score"])
    print(f"\nBest model: {best} (F1: {results[best]['metrics']['f1_score']:.4f})")
