

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(y_true, probabilities, threshold=0.10):

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    average_precision = average_precision_score(
        y_true, probabilities
    )

    matrix = confusion_matrix(
        y_true, predictions
    )

    print("\nModel Performance Results")
    print("-------------------------")
    print("Threshold:", threshold)
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("Average Precision:", round(average_precision, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "average_precision": average_precision,
        "confusion_matrix": matrix.tolist(),
    }


def check_performance_alerts(metrics):
    recall_threshold = 0.30
    precision_threshold = 0.15

    print("\nPerformance Monitoring Alerts")
    print("-----------------------------")

    if metrics["recall"] < recall_threshold:
        print("WARNING: Model recall is below threshold!")
    else:
        print("Recall status: Healthy")

    if metrics["precision"] < precision_threshold:
        print("WARNING: Model precision is below threshold!")
    else:
        print("Precision status: Healthy")
