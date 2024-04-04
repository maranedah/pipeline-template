import logging

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier
from model.models.ModelSelector import ModelSelector
from model.models.Scoring import (
    gini_score,
)
from sklearn.metrics import roc_auc_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ClassificationModels(ModelSelector):
    def __init__(self):
        self.models = [
            # LogisticRegression,
            # DecisionTreeClassifier,
            # RandomForestClassifier,
            # GradientBoostingClassifier,
            # SVC(probability=True),
            # KNeighborsClassifier,
            # GaussianNB,
            # MLPClassifier,
            # XGBClassifier,
            LGBMClassifier,
            # CatBoostClassifier,
        ]
        self.early_stopping_rounds = 10
        self.metrics = [roc_auc_score, gini_score]
        self.tuning_metric = gini_score
        self.ensemble_scoring_function = gini_score
        self.cv_splits = 5
        self.suggestion_trials = None
        self.timeout_in_hours = 100
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
                # "device": "gpu"
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
    import numpy as np

    logging.info("Started program")
    X_train = np.load("X_train.npy")
    y_train = np.ravel(pd.read_parquet("y_train.parquet").values)
    X_valid = np.load("X_valid.npy")
    y_valid = np.ravel(pd.read_parquet("y_valid.parquet").values)

    model = ClassificationModels()
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
