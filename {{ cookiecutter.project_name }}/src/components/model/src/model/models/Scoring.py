import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# To turn off all warnings
warnings.filterwarnings("ignore")


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


def load_index(df, index):
    return df.values[index]


def parallel_data_load(X, y, train_index, test_index):
    from multiprocessing import Pool

    arguments_list = [
        (X, train_index),
        (y, train_index),
        (X, test_index),
        (y, test_index),
    ]
    num_processes = 4
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(load_index, arguments_list)
    return results


class PersistentFolds:
    def __init__(self, X, y, n_splits):
        self.n_splits = n_splits
        self.kfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.persist_kfolds(X, y)

    def persist_kfolds(self, X, y):
        for i, (train_index, test_index) in enumerate(self.kfolds.split(X, y)):
            X_train, y_train, X_test, y_test = parallel_data_load(
                X, y, train_index, test_index
            )
            np.savez_compressed(
                f"fold_{i}.npz",
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

    def load_fold(self, i):
        loaded = np.load(f"fold_{i}.npz")
        X_train = loaded["X_train"]
        y_train = loaded["y_train"]
        X_test = loaded["X_test"]
        y_test = loaded["y_test"]
        return X_train, y_train, X_test, y_test


def cross_validation_score(
    estimator, kfolds, fit_params, score_funcs, n_splits, random_state
) -> dict:
    # kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = []
    import time

    for i in range(kfolds.n_splits):
        print("training fold")
        start = time.time()
        X_train, y_train, X_test, y_test = kfolds.load_fold(i)
        end = time.time()
        print(end - start)
        y_hat = estimator.fit(X_train, y_train, **fit_params).predict_proba(X_test)[
            :, 1
        ]
        scores = {f.__name__: f(y_test, y_hat) for f in score_funcs}
        result.append(scores)
    result = pd.DataFrame(data=result).mean(axis=0)
    return result.to_dict()
