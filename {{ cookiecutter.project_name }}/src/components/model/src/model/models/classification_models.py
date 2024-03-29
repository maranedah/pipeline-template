import logging

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from model.models.ModelSelector import ModelSelector
from model.models.Scoring import (
    gini_score,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ClassificationModels(ModelSelector):
    def __init__(self):
        self.models = [
            LogisticRegression,
            DecisionTreeClassifier,
            RandomForestClassifier,
            GradientBoostingClassifier,
            SVC(probability=True),
            KNeighborsClassifier,
            GaussianNB,
            MLPClassifier,
            XGBClassifier,
            LGBMClassifier,
            CatBoostClassifier,
        ]
        self.early_stopping_rounds = 10
        self.metrics = [roc_auc_score, gini_score]
        self.tuning_metric = gini_score
        self.ensemble_scoring_function = gini_score
        self.cv_splits = 5
        self.suggestion_trials = None
        self.timeout_in_hours = 7
        self.eval_metric = "auc"
        self.random_state = 42

    def get_fit_params(self, model, eval_set):
        model_name = type(model()).__name__
        fit_params = {
            "LGBMClassifier": {
                "eval_set": eval_set,
                "eval_metric": self.eval_metric.lower(),
                "callbacks": [lgb.early_stopping(stopping_rounds=10, verbose=False)],
            },
            "CatBoostClassifier": {
                "eval_set": eval_set,
                "verbose": 0,
                "early_stopping_rounds": self.early_stopping_rounds,
                "callbacks": [],
            },
            "XGBClassifier": {
                "eval_set": [eval_set],
                "verbose": 0,
                "callbacks": [
                    xgb.callback.EarlyStopping(rounds=self.early_stopping_rounds)
                ],
            },
        }
        if model_name in fit_params:
            return fit_params[model_name]
        else:
            return {}

    def get_init_params(self, model):
        model_name = type(model()).__name__
        init_params = {
            "LGBMClassifier": {
                "verbosity": -1,
                "random_state": 42,
                "verbose_eval": -1,
                # "is_unbalance": True,
            },
            "CatBoostClassifier": {
                "random_seed": 42,
                "eval_metric": self.eval_metric.upper(),
            },
            "XGBRegressor": {
                "verbosity": 0,
                "random_seed": 42,
                "verbose_eval": False,
                "eval_metric": self.eval_metric.lower(),
            },
        }
        if model_name in init_params:
            return init_params[model_name]
        else:
            return {}


if __name__ == "__main__":
    X_train = pd.read_parquet("processed_X_train.parquet")
    y_train = pd.read_parquet("y_train.parquet")
    X_valid = pd.read_parquet("processed_X_valid.parquet")
    y_valid = pd.read_parquet("y_valid.parquet")
    X_test = pd.read_parquet("processed_X_test.parquet")
    y_test = pd.read_parquet("y_test.parquet")

    for df in X_train, X_valid, X_test:
        for column in df.columns:
            min_value = df[column].min()
            df[column].fillna(min_value, inplace=True)

    model = ClassificationModels()
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

    base_train = pd.read_parquet("base_train.parquet")
    base_valid = pd.read_parquet("base_valid.parquet")
    base_test = pd.read_parquet("base_test.parquet")

    for base, X in [(base_train, X_train), (base_valid, X_valid), (base_test, X_test)]:
        y_pred = model.predict_ensemble(X)
        base["score"] = y_pred

    print(
        f'The AUC score on the train set is: {roc_auc_score(base_train["target"], base_train["score"])}'  # noqa: E501
    )
    print(
        f'The AUC score on the valid set is: {roc_auc_score(base_valid["target"], base_valid["score"])}'  # noqa: E501
    )
    print(
        f'The AUC score on the test set is: {roc_auc_score(base_test["target"], base_test["score"])}'  # noqa: E501
    )

    from model.models.Scoring import gini_stability

    stability_score_train = gini_stability(base_train)
    stability_score_valid = gini_stability(base_valid)
    stability_score_test = gini_stability(base_test)

    print(f"The stability score on the train set is: {stability_score_train}")
    print(f"The stability score on the valid set is: {stability_score_valid}")
    print(f"The stability score on the test set is: {stability_score_test}")
