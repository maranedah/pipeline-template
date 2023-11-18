import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from .metrics import get_metrics


def split_data(df):
    X = df.drop(columns=["y"])
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def run_model(project_id, preprocessed_dataset_uri):
    df = pd.read_parquet(preprocessed_dataset_uri)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df)
    model = LGBMClassifier(
        valid_sets=(X_val, y_val),
        callbacks=[lgb.early_stopping(stopping_rounds=5)],
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
    )
    baseline_model = None
    metrics = get_metrics(X_train, y_train, X_test, y_test, model, baseline_model)

    X_test["y"] = model.predict(X_test)

    return X_test, model, metrics
