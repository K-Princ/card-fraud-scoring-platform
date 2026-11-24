
import numpy as np
import pandas as pd
from src.monitoring.prediction_monitor import generate_predictions


def sample_transactions():
    return pd.DataFrame({
        "TransactionAmt": [150.0, 250.0],
        "dist2": [38.0, None],
        "D7": [0.0, None],
        "D13": [0.0, 10.0],
        "ProductCD": ["W", "C"],
        "card4": ["visa", "mastercard"],
        "card6": ["debit", "credit"]
    })


def test_generate_predictions_output_shape():
    df = sample_transactions()

    predictions = generate_predictions(df)

    assert len(predictions) == len(df)


def test_predictions_are_valid_probabilities():
    df = sample_transactions()

    predictions = generate_predictions(df)

    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)


def test_predictions_are_reproducible():
    df = sample_transactions()

    first = generate_predictions(df)
    second = generate_predictions(df)

    np.testing.assert_allclose(first, second)


def test_generate_predictions_does_not_modify_input():
    df = sample_transactions()
    original_df = df.copy(deep=True)

    generate_predictions(df)

    pd.testing.assert_frame_equal(df, original_df)
