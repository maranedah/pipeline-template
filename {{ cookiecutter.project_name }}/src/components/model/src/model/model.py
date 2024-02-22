import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .constants import EVALUATION_METRICS, MODELS, RANDOM_STATE
from .metrics import get_metrics


def split_data(
    df: pd.DataFrame, split_ratio: str, stratify: str | None = None
) -> tuple[
    (pd.DataFrame, pd.DataFrame),
    (pd.DataFrame, pd.DataFrame),
    (pd.DataFrame, pd.DataFrame),
]:
    """
    Split the input DataFrame into training, validation, and test sets.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the dataset.
    split_ratio : str
        A string representing the split ratio in the format "train:val:test".
    stratify : str or None, optional
        If not None, data is split in a stratified fashion.

    Returns:
    --------
    tuple
        A tuple containing three pairs of DataFrames representing
        (X_train, y_train), (X_val, y_val), and (X_test, y_test).
    """
    _, val_size, test_size = [int(value) / 10 for value in split_ratio.split(":")]
    X = df.drop(columns=["y"])
    y = df["y"]
    X_train, X_, y_train, y_ = train_test_split(
        X,
        y,
        test_size=val_size + test_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_,
        y_,
        test_size=test_size / (val_size + test_size),
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def init_model(model: str) -> lgb.LGBMModel:
    model = MODELS[model](random_state=RANDOM_STATE)
    return model


def run_model(
    project_id: str, preprocessed_dataset_uri: str, split_ratio: str, model_str: str
) -> tuple[LGBMClassifier, dict]:
    """
    Train a LightGBM classifier on a preprocessed dataset and evaluate its performance.

    Parameters
    ----------
    project_id : str
        Identifier for the project in GCP.
    preprocessed_dataset_uri : str
        URI of the preprocessed dataset in parquet format.
    split_ratio : str
        Ratio specifying the split of the dataset into
        training, validation, and test sets
        in the format "train:val:test".

    Returns
    -------
    tuple
        A tuple containing the trained LightGBM model
        and a dictionary of evaluation metrics.
    """
    # Read data
    df = pd.read_parquet(preprocessed_dataset_uri)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df, split_ratio)

    # Init model from string
    model = init_model(model_str)
    model.n_jobs = -1

    # Fit and performance testing
    model.valid_sets = (X_val, y_val)
    model.fit(X_train, y_train)

    # Fit baseline model
    baseline_model = LogisticRegression(multi_class="ovr", random_state=RANDOM_STATE)
    baseline_model.fit(X_train, y_train)

    metrics = get_metrics(
        X_train, y_train, X_test, y_test, model, baseline_model, EVALUATION_METRICS
    )

    return model, metrics
