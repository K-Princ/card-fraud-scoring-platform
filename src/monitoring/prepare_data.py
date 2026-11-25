
from pathlib import Path

import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

from src.monitoring.alerts import check_drift_status
from src.monitoring.performance_monitor import (
    check_performance_alerts,
    evaluate_model,
)
from src.monitoring.prediction_monitor import (
    compare_predictions,
    generate_prediction_drift_report,
    generate_predictions,
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "raw" / "train_transaction.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load transaction data
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "dist2",
        "D7",
        "D13",
        "ProductCD",
        "card4",
        "card6",
        "isFraud",
    ],
)

print("Dataset shape:", df.shape)
print(df.head())


# --------------------------------------------------
# 3. Sort transactions chronologically
# --------------------------------------------------

df = df.sort_values("TransactionDT").reset_index(drop=True)


# --------------------------------------------------
# 4. Create reference and current datasets
# --------------------------------------------------

split_index = int(len(df) * 0.8)

reference_raw = df.iloc[:split_index].copy()
current_raw = df.iloc[split_index:].copy()

print("\nReference shape:", reference_raw.shape)
print("Current shape:", current_raw.shape)


# --------------------------------------------------
# 5. Compare actual fraud rates
# --------------------------------------------------

print(
    "\nReference fraud rate:",
    reference_raw["isFraud"].mean()
)

print(
    "Current fraud rate:",
    current_raw["isFraud"].mean()
)


# --------------------------------------------------
# 6. Prepare feature monitoring datasets
# --------------------------------------------------

monitoring_columns = [
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
]

reference_data = reference_raw[monitoring_columns].copy()
current_data = current_raw[monitoring_columns].copy()

print("\nMonitoring columns:", monitoring_columns)
print("Reference monitoring shape:", reference_data.shape)
print("Current monitoring shape:", current_data.shape)


# --------------------------------------------------
# 7. Generate feature data drift report
# --------------------------------------------------

reference_dataset = Dataset.from_pandas(reference_data)
current_dataset = Dataset.from_pandas(current_data)

feature_report = Report([
    DataDriftPreset()
])

feature_result = feature_report.run(
    reference_dataset,
    current_dataset
)

feature_report_path = REPORT_DIR / "data_drift_report.html"

feature_result.save_html(str(feature_report_path))

# Extract feature drift results
feature_report_data = feature_result.dict()

feature_drift_metric = next(
    metric
    for metric in feature_report_data["metrics"]
    if metric["metric_name"].startswith("DriftedColumnsCount")
)

drifted_count = int(feature_drift_metric["value"]["count"])

print("\nFeature drifted columns:", drifted_count)

check_drift_status(drifted_count > 0)

print("Feature drift report saved:", feature_report_path)


# --------------------------------------------------
# 8. Generate reference predictions
# --------------------------------------------------

print("\nGenerating reference predictions...")

reference_predictions = generate_predictions(reference_raw)

print(
    "Reference predictions generated:",
    len(reference_predictions)
)


# --------------------------------------------------
# 9. Generate current predictions
# --------------------------------------------------

print("\nGenerating current predictions...")

current_predictions = generate_predictions(current_raw)

print(
    "Current predictions generated:",
    len(current_predictions)
)


# --------------------------------------------------
# 10. Compare prediction distributions
# --------------------------------------------------

comparison = compare_predictions(
    reference_predictions,
    current_predictions,
)

print("\nPrediction distribution comparison:")
print(comparison)


# --------------------------------------------------
# 11. Save prediction comparison CSV
# --------------------------------------------------

prediction_summary_path = (
    REPORT_DIR / "prediction_drift_summary.csv"
)

comparison.to_csv(prediction_summary_path)

print("\nPrediction comparison saved:", prediction_summary_path)


# --------------------------------------------------
# 12. Generate prediction drift report
# --------------------------------------------------

generate_prediction_drift_report(
    reference_predictions,
    current_predictions,
)


# --------------------------------------------------
# 13. Evaluate model performance on current data
# --------------------------------------------------

print("\nEvaluating model performance...")

performance_results = evaluate_model(
    y_true=current_raw["isFraud"].values,
    probabilities=current_predictions,
    threshold=0.10,
)

# Check model performance alerts
check_performance_alerts(performance_results)


# --------------------------------------------------
# 14. Save model performance metrics
# --------------------------------------------------

performance_df = pd.DataFrame([
    {
        key: value
        for key, value in performance_results.items()
        if key != "confusion_matrix"
    }
])

performance_path = REPORT_DIR / "model_performance.csv"

performance_df.to_csv(
    performance_path,
    index=False
)

print("\nModel performance report saved:", performance_path)


# --------------------------------------------------
# 15. Completion
# --------------------------------------------------

print("\n======================================")
print("MONITORING PIPELINE COMPLETED")
print("======================================")
