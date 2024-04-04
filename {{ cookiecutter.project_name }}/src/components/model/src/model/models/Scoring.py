import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# To turn off all warnings
# warnings.filterwarnings("ignore")


def stability_score(y_true, y_pred):
    base_valid = pd.read_parquet("base_valid.parquet")
    base_valid["score"] = y_pred
    return gini_stability(base_valid)


def gini_score(y_true, y_pred):
    gini = 2 * roc_auc_score(y_true, y_pred) - 1
    return gini


def gini_stability(base, w_fallingrate=88.0, w_resstd=-0.5):
    gini_in_time = (
        base.loc[:, ["WEEK_NUM", "target", "score"]]
        .sort_values("WEEK_NUM")
        .groupby("WEEK_NUM")[["target", "score"]]
        .apply(lambda x: 2 * roc_auc_score(x["target"], x["score"]) - 1)
        .tolist()
    )

    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std


def stability(scores):
    w_fallingrate = 0.88
    w_resstd = 0.5
    x = np.arange(len(scores))
    y = scores
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(scores)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std


def temporal_validation_score(
    estimator, X, y, fit_params, score_funcs, n_splits, random_state
):
    # read base_train to get indexes
    base_train = pd.read_parquet("base_train.parquet")
    folds = base_train.groupby("WEEK_NUM").apply(lambda group: group.index.tolist())
    scores = []
    for i in range(1, len(folds)):
        train_index = [item for sublist in folds[:i] for item in sublist]
        test_index = folds[i]
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_hat = estimator.fit(X_train, y_train, **fit_params).predict(X_test)
        score = gini_score(y_test, y_hat)
        scores.append(score)
    result = stability(scores)
    return result


def cross_validation_score(
    estimator, X, y, fit_params, score_funcs, n_splits, random_state
) -> dict:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = []
    for train_index, test_index in kf.split(X, y):
        print("training fold")
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_hat = estimator.fit(X_train, y_train, **fit_params).predict_proba(X_test)[
            :, 1
        ]
        scores = {f.__name__: f(y_test, y_hat) for f in score_funcs}
        result.append(scores)
    result = pd.DataFrame(data=result).mean(axis=0)
    return result.to_dict()
