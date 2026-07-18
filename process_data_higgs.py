import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data():

    data = pd.read_csv("training/training.csv")
    print(data.shape)
    print(data.columns.tolist())
    print(data["Label"].value_counts(normalize=True))

    # Convert label to binary
    Y = (data["Label"] == "s").astype(int).values

    # Drop non-feature columns
    drop_cols = ["EventId", "Label", "Weight"]
    feature_cols = [c for c in data.columns if c not in drop_cols]

    X = data[feature_cols].values

    X_clean = X.copy()
    for col_idx in range(X.shape[1]):
        col = X_clean[:, col_idx]
        missing_mask = col == -999.0
        if missing_mask.sum() > 0:
            valid_median = np.median(col[~missing_mask])
            col[missing_mask] = valid_median

    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
        X_clean, Y, test_size=0.2, random_state=42, stratify=Y
    )

    X_train = X_train_raw.T
    X_test = X_test_raw.T
    Y_train = Y_train_raw.reshape(1, -1)
    Y_test = Y_test_raw.reshape(1, -1)

    mean = X_train.mean(axis=1, keepdims=True)
    std = X_train.std(axis=1, keepdims=True)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, Y_train, X_test, Y_test
