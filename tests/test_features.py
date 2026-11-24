
import pandas as pd
from src.features.build_features import (
    add_missing_indicators,
    apply_encoder,
    apply_medians,
    build_features,
    create_encoder,
    encode_categorical,
    fill_numeric_missing,
    get_medians,
)


def test_add_missing_indicators():
    df = pd.DataFrame({
        "dist2": [10, None],
        "D7": [None, 20],
        "D13": [30, None]
    })

    result = add_missing_indicators(df)

    assert result["dist2_missing"].tolist() == [0, 1]
    assert result["D7_missing"].tolist() == [1, 0]
    assert result["D13_missing"].tolist() == [0, 1]


def test_get_medians():
    df = pd.DataFrame({
        "dist2": [10, 20, None],
        "D7": [5, None, 15]
    })

    medians = get_medians(df, ["dist2", "D7"])

    assert medians == {
        "dist2": 15.0,
        "D7": 10.0
    }


def test_apply_medians():
    df = pd.DataFrame({
        "dist2": [10, None],
        "D7": [None, 20]
    })

    medians = {
        "dist2": 15.0,
        "D7": 10.0
    }

    result = apply_medians(df, medians)

    assert result["dist2"].tolist() == [10.0, 15.0]
    assert result["D7"].tolist() == [10.0, 20.0]


def test_encode_categorical():
    df = pd.DataFrame({
        "ProductCD": ["W", "C", None],
        "card4": ["visa", "mastercard", "visa"]
    })

    result = encode_categorical(
        df,
        ["ProductCD", "card4"]
    )

    assert "ProductCD_W" in result.columns
    assert "ProductCD_C" in result.columns
    assert "ProductCD_Unknown" in result.columns

    assert result.shape[0] == 3
    assert df["ProductCD"].isnull().sum() == 1


def test_apply_encoder_handles_unknown_categories():
    training_data = pd.DataFrame({
        "ProductCD": ["W", "C"],
        "card4": ["visa", "mastercard"],
        "card6": ["debit", "credit"]
    })

    encoder = create_encoder()
    encoder.fit(training_data)

    new_data = pd.DataFrame({
        "ProductCD": ["W", "R"],
        "card4": ["visa", "discover"],
        "card6": ["debit", "credit"]
    })

    result = apply_encoder(
        new_data,
        encoder,
        ["ProductCD", "card4", "card6"]
    )

    assert result.shape[0] == 2
    assert result.shape[1] == len(
        encoder.get_feature_names_out(
            ["ProductCD", "card4", "card6"]
        )
    )
    assert result.isnull().sum().sum() == 0


def test_build_features():
    df = pd.DataFrame({
        "dist2": [10, None],
        "D7": [5, 15],
        "D13": [None, 30],
        "ProductCD": ["W", "C"],
        "card4": ["visa", "mastercard"],
        "card6": ["debit", "credit"]
    })

    medians = {
        "dist2": 10.0,
        "D7": 10.0,
        "D13": 20.0
    }

    encoder = create_encoder()
    encoder.fit(df[["ProductCD", "card4", "card6"]])

    result = build_features(df, medians, encoder)

    assert result.shape[0] == 2
    assert result.isnull().sum().sum() == 0

    assert "dist2_missing" in result.columns
    assert "D7_missing" in result.columns
    assert "D13_missing" in result.columns

def test_fill_numeric_missing():
    df = pd.DataFrame({
        "dist2": [10.0, None, 30.0],
        "D7": [None, 20.0, 40.0]
    })

    result = fill_numeric_missing(
        df,
        ["dist2", "D7"]
    )

    assert result["dist2"].tolist() == [10.0, 20.0, 30.0]
    assert result["D7"].tolist() == [30.0, 20.0, 40.0]

    # Original DataFrame should remain unchanged
    assert df["dist2"].isnull().sum() == 1
