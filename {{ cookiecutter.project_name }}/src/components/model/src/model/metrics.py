import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def f1_score_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def precision_score_macro(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro")


def recall_score_macro(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro")


metrics_functions = {
    "accuracy": accuracy_score,
    "f1": f1_score_macro,
    "precision": precision_score_macro,
    "recall": recall_score_macro,
}


def get_metrics(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model: lgb.LGBMModel,
    baseline_model: any,
    evaluation_metrics: list[str],
) -> pd.DataFrame:
    """
    Evaluate the performance metrics of a model on training and test sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features of the training set.
    y_train : pd.DataFrame
        Labels of the training set.
    X_test : pd.DataFrame
        Features of the test set.
    y_test : pd.DataFrame
        Labels of the test set.
    model : Any
        The trained model to evaluate.
    baseline_model : Any
        The baseline model for comparison (if applicable).
    evaluation_metrics : List[str]
        List of evaluation metrics to compute.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing performance metrics for different sets and metrics.
    """
    sets_to_compare = {
        "train": (model.predict(X_train), y_train),
        "test": (model.predict(X_test), y_test),
        # "baseline": (baseline_model.predict(X_test), y_test),
    }
    metrics = {}
    for set_name, (y_hat, y_true) in sets_to_compare.items():
        for metric in evaluation_metrics:
            metrics.update(
                {f"{set_name}_{metric}": metrics_functions[metric](y_true, y_hat)}
            )

    return metrics
