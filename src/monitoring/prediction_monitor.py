
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"

# Load saved model and preprocessing artifacts
model = joblib.load(MODEL_DIR / "fraud_calibrated.pkl")
encoder = joblib.load(MODEL_DIR / "encoder.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
medians = joblib.load(MODEL_DIR / "train_medians.pkl")


def generate_predictions(data):
    data = data.copy()

    numerical_columns = [
        "TransactionAmt", "dist2", "D7", "D13"
    ]

    categorical_columns = [
        "ProductCD", "card4", "card6"
    ]

    # Handle missing numerical values
    for col in numerical_columns:
        data[col] = data[col].fillna(medians[col])

    # Scale numerical features
    scaled_numeric = scaler.transform(data[numerical_columns])

    # Handle missing categorical values
    data[categorical_columns] = (
        data[categorical_columns].fillna("Unknown")
    )

    # Encode categorical features
    encoded_categorical = encoder.transform(
        data[categorical_columns]
    )

    # Combine features in the same order as the API
    features = pd.DataFrame(
        scaled_numeric,
        columns=numerical_columns,
        index=data.index
    )

    encoded_features = pd.DataFrame(
        encoded_categorical,
        columns=encoder.get_feature_names_out(
            categorical_columns
        ),
        index=data.index
    )

    model_features = pd.concat(
        [features, encoded_features],
        axis=1
    )

    # Generate fraud probabilities
    probabilities = model.predict_proba(model_features)[:, 1]

    return probabilities


def compare_predictions(reference_predictions, current_predictions):
    reference_stats = pd.Series(reference_predictions).describe()
    current_stats = pd.Series(current_predictions).describe()

    comparison = pd.DataFrame({
        "Reference": reference_stats,
        "Current": current_stats
    })

    print("\nPrediction Distribution Comparison:")
    print(comparison)

    return comparison


from evidently import Dataset, Report
from evidently.presets import DataDriftPreset


def generate_prediction_drift_report(
    reference_predictions,
    current_predictions
):
    reference_df = pd.DataFrame({
        "fraud_probability": reference_predictions
    })

    current_df = pd.DataFrame({
        "fraud_probability": current_predictions
    })

    reference_dataset = Dataset.from_pandas(reference_df)
    current_dataset = Dataset.from_pandas(current_df)

    report = Report([
        DataDriftPreset()
    ])

    result = report.run(
        reference_dataset,
        current_dataset
    )

    # Extract actual drift result
    report_data = result.dict()

    drift_metric = next(
        metric
        for metric in report_data["metrics"]
        if metric["metric_name"].startswith("ValueDrift")
    )

    drift_score = float(drift_metric["value"])
    drift_threshold = drift_metric["config"]["threshold"]

    drift_detected = drift_score >= drift_threshold

    print("Prediction drift score:", drift_score)
    print("Drift threshold:", drift_threshold)

    # Trigger monitoring alert
    from src.monitoring.alerts import check_drift_status

    check_drift_status(drift_detected)

    output_path = (
        BASE_DIR / "reports" / "prediction_drift_report.html"
    )

    result.save_html(str(output_path))

    print("Prediction drift report generated!")
    print("Report location:", output_path)

    return result
