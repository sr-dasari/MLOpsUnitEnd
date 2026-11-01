import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


VALID_PASSENGER = {
    "Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0,
    "Parch": 0, "Fare": 211.34, "Embarked": "S",
    "Name": "Smith, Mrs. John",
}


@pytest.fixture(autouse=True)
def _mock_mlflow(monkeypatch):
    """Patch MLflow model loading so tests run without mlflow.db."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.feature_names_in_ = np.array([
        "Pclass", "Age", "SibSp", "Parch", "Fare",
        "FamilySize", "IsAlone", "Sex_male",
        "Embarked_Q", "Embarked_S",
        "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare",
    ])

    with patch("utils.select_model", return_value=(mock_model, "test_run_id_1234")):
        # Re-import main so it picks up the mock
        if "main" in sys.modules:
            del sys.modules["main"]
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import main
        yield TestClient(main.app), mock_model


# ── Happy path ──────────────────────────────────────────────────────────────

def test_root(_mock_mlflow):
    client, _ = _mock_mlflow
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "active"
    assert "loaded_run_id" in body


def test_predict_survived(_mock_mlflow):
    client, mock_model = _mock_mlflow
    mock_model.predict.return_value = np.array([1])
    resp = client.post("/predict", json=VALID_PASSENGER)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == 1
    assert body["label"] == "survived"


def test_predict_did_not_survive(_mock_mlflow):
    client, mock_model = _mock_mlflow
    mock_model.predict.return_value = np.array([0])
    passenger = {**VALID_PASSENGER, "Pclass": 3, "Sex": "male", "Fare": 8.05}
    resp = client.post("/predict", json=passenger)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == 0
    assert body["label"] == "did not survive"


def test_predict_without_name(_mock_mlflow):
    client, _ = _mock_mlflow
    passenger = {k: v for k, v in VALID_PASSENGER.items() if k != "Name"}
    resp = client.post("/predict", json=passenger)
    assert resp.status_code == 200


# ── Validation (security / input constraints) ──────────────────────────────

def test_invalid_pclass(_mock_mlflow):
    client, _ = _mock_mlflow
    resp = client.post("/predict", json={**VALID_PASSENGER, "Pclass": 5})
    assert resp.status_code == 422


def test_invalid_sex(_mock_mlflow):
    client, _ = _mock_mlflow
    resp = client.post("/predict", json={**VALID_PASSENGER, "Sex": "unknown"})
    assert resp.status_code == 422


def test_negative_age(_mock_mlflow):
    client, _ = _mock_mlflow
    resp = client.post("/predict", json={**VALID_PASSENGER, "Age": -5})
    assert resp.status_code == 422


def test_invalid_embarked(_mock_mlflow):
    client, _ = _mock_mlflow
    resp = client.post("/predict", json={**VALID_PASSENGER, "Embarked": "X"})
    assert resp.status_code == 422


def test_missing_required_field(_mock_mlflow):
    client, _ = _mock_mlflow
    incomplete = {k: v for k, v in VALID_PASSENGER.items() if k != "Age"}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_negative_fare(_mock_mlflow):
    client, _ = _mock_mlflow
    resp = client.post("/predict", json={**VALID_PASSENGER, "Fare": -10})
    assert resp.status_code == 422


def test_empty_body(_mock_mlflow):
    client, _ = _mock_mlflow
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
