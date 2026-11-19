import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def create_encoder():
    return OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )


def add_missing_indicators(df):
    df = df.copy()

    for col in ["dist2", "D7", "D13"]:
        df[col + "_missing"] = df[col].isnull().astype(int)

    return df

def fill_numeric_missing(df, columns):
    df = df.copy()

    for col in columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    return df

def get_medians(df, columns):
    return df[columns].median().to_dict()

def apply_medians(df, medians):
    df = df.copy()

    for col, value in medians.items():
        df[col] = df[col].fillna(value)

    return df

def preprocess_features(df, medians):
    df = add_missing_indicators(df)
    df = apply_medians(df, medians)
    return df

def encode_categorical(df, columns):
    df = df.copy()

    df[columns] = df[columns].fillna("Unknown")

    df = pd.get_dummies(
        df,
        columns=columns,
        dtype=int
    )

    return df

def apply_encoder(df, encoder, columns):
    df = df.copy()

    df[columns] = df[columns].fillna("Unknown")

    encoded_array = encoder.transform(df[columns])

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(columns),
        index=df.index
    )

    df = df.drop(columns=columns)

    return pd.concat([df, encoded_df], axis=1)

def build_features(df, medians, encoder):
    df = preprocess_features(df, medians)

    df = apply_encoder(
        df,
        encoder,
        ["ProductCD", "card4", "card6"]
    )

    return df