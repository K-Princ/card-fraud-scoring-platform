import numpy as np
import pandas as pd
import pytest
from src.monitoring import alerts, prediction_monitor
from src.monitoring.alerts import check_drift_status
from src.monitoring.performance_monitor import (
    check_performance_alerts,
    evaluate_model,
)
from src.monitoring.prediction_monitor import compare_predictions


def test_compare_predictions():
    reference = np.array([0.1, 0.2, 0.3])
    current = np.array([0.2, 0.3, 0.4])

    result = compare_predictions(reference, current)

    assert isinstance(result, pd.DataFrame)
    assert "Reference" in result.columns
    assert "Current" in result.columns
    assert np.isclose(result.loc["mean", "Reference"], 0.2)
    assert np.isclose(result.loc["mean", "Current"], 0.3)


def test_drift_alert_detected(capsys):
    check_drift_status(True)

    captured = capsys.readouterr()

    assert "WARNING: Data drift detected!" in captured.out


def test_no_drift_alert(capsys):
    check_drift_status(False)

    captured = capsys.readouterr()

    assert "No drift detected." in captured.out


def test_evaluate_model():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.8, 0.2])

    metrics = evaluate_model(
        y_true,
        probabilities,
        threshold=0.5
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5

    assert np.isclose(
        metrics["f1_score"],
        2 / 3
    )

    assert metrics["confusion_matrix"] == [
        [2, 0],
        [1, 1]
    ]


def test_performance_alerts_healthy(capsys):
    metrics = {
        "recall": 0.50,
        "precision": 0.20
    }

    check_performance_alerts(metrics)

    captured = capsys.readouterr()

    assert "Recall status: Healthy" in captured.out
    assert "Precision status: Healthy" in captured.out


def test_performance_alerts_warning(capsys):
    metrics = {
        "recall": 0.20,
        "precision": 0.10
    }

    check_performance_alerts(metrics)

    captured = capsys.readouterr()

    assert "WARNING: Model recall is below threshold!" in captured.out
    assert "WARNING: Model precision is below threshold!" in captured.out


@pytest.mark.parametrize(
    "drift_score, expected_alert",
    [
        (0.15, True),
        (0.05, False)
    ]
)
def test_generate_prediction_drift_report(
    monkeypatch,
    drift_score,
    expected_alert
):
    class FakeDataset:
        @staticmethod
        def from_pandas(data):
            return data

    class FakeResult:
        def dict(self):
            return {
                "metrics": [
                    {
                        "metric_name": "ValueDrift",
                        "value": drift_score,
                        "config": {
                            "threshold": 0.10
                        }
                    }
                ]
            }

        def save_html(self, path):
            self.saved_path = path

    class FakeReport:
        def __init__(self, presets):
            pass

        def run(self, reference, current):
            return FakeResult()

    alert_calls = []

    def fake_alert(drift_detected):
        alert_calls.append(drift_detected)

    monkeypatch.setattr(
        prediction_monitor,
        "Dataset",
        FakeDataset
    )

    monkeypatch.setattr(
        prediction_monitor,
        "Report",
        FakeReport
    )

    monkeypatch.setattr(
    prediction_monitor,
    "check_drift_status",
    fake_alert
)

    result = prediction_monitor.generate_prediction_drift_report(
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4]
    )

    assert isinstance(result, FakeResult)
    assert alert_calls == [expected_alert]
