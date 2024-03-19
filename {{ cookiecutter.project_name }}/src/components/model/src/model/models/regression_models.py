import logging

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from model.hyperparameter_tuning import HyperparameterTuning
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RegressionModels:
    def __init__(self):
        self.models = [
            LinearRegression,
            Ridge,
            Lasso,
            ElasticNet,
            DecisionTreeRegressor,
            RandomForestRegressor,
            GradientBoostingRegressor,
            SVR,
            KNeighborsRegressor,
            MLPRegressor,
            XGBRegressor,
            LGBMRegressor,
            CatBoostRegressor,
        ]
        self.metrics = [
            "neg_mean_squared_error",
            "neg_root_mean_squared_error",
            "neg_mean_absolute_error",
            "r2",
            "neg_mean_absolute_percentage_error",
        ]
        self.tuning_metric = "neg_mean_absolute_percentage_error"
        self.cv_splits = 5

    def cross_validation(self, models, X, y, eval_set):
        logging.info("Cross Validation...")
        cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=42)
        cv_info = []
        for model in models:
            logging.info(f"Training model {type(model).__name__}")
            result = cross_validate(
                estimator=model,
                X=X,
                y=y,
                scoring=self.metrics,
                cv=cv,
                params=self.get_fit_params(
                    type(model), eval_set, early_stopping_rounds=5
                ),
                n_jobs=-1,
            )
            cv_info.append(result)

        logging.info("Finished Cross Validation...")
        return cv_info

    def fit(self, X, y, eval_set):
        # Init all models with default params
        logging.info("Instantiating models with default parameters")
        models = [model(**self.get_init_params(model)) for model in self.models]

        # Cross Validation for each default model
        cv_models_info = self.cross_validation(models, X, y, eval_set)

        models_performance = [
            {
                "model_name": type(model).__name__,
                **{k: v.mean() for k, v in cv_info.items()},
            }
            for model, cv_info in zip(models, cv_models_info)
        ]
        mlflow.set_experiment("Model Selection - No Tuning")

        for model_performance in models_performance:
            with mlflow.start_run():
                for k, v in model_performance.items():
                    if k == "model_name":
                        mlflow.log_param("model_name", v)
                    else:
                        mlflow.log_metric(k, v)

        df = pd.DataFrame(models_performance)
        # Get top 3 models
        top_3_indexes = df.nlargest(3, f"test_{self.tuning_metric}").index.tolist()
        top_3_models = [self.models[i] for i in top_3_indexes]

        # Hyperparameter tuning of best 3 models
        suggestion_trials = None
        timeout_in_hours = 0.5
        early_stopping_rounds = 5
        model_studies = [
            HyperparameterTuning(
                model,
                metrics=self.tuning_metric,
                model_init_params=self.get_init_params(model),
            ).run_tuning(
                X,
                y,
                self.get_fit_params(model, eval_set, early_stopping_rounds),
                self.cv_splits,
                suggestion_trials,
                timeout_in_hours,
            )
            for model in top_3_models
        ]

        # Assign optimized top 3 best models to object
        self.top_3_models = [
            model(**study.best_params, **self.get_init_params(model)).fit(
                X, y, **self.get_fit_params(model, eval_set, early_stopping_rounds)
            )
            for study, model in zip(model_studies, top_3_models)
        ]

        # Save top 3 models results
        top_3_performances = [
            {"model_name": type(model()).__name__, self.tuning_metric: study.best_value}
            for study, model in zip(model_studies, top_3_models)
        ]
        top_3_df = pd.DataFrame(top_3_performances)
        top_3_df.to_csv("Top 3 Tuned Model Performances.csv", index=False)

        # Get best model
        self.best_model = self.top_3_models[
            top_3_df.nlargest(1, self.tuning_metric).index.tolist()[0]
        ]

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_ensemble(self, X):
        return np.array([model.predict(X) for model in self.top_3_models]).mean(axis=0)

    def get_fit_params(self, model, eval_set, early_stopping_rounds):
        model_name = type(model()).__name__
        fit_params = {
            "LGBMRegressor": {
                "eval_set": eval_set,
                "eval_metric": "mape",
                "callbacks": [
                    lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds, verbose=False
                    )
                ],
            },
            "CatBoostRegressor": {
                "eval_set": eval_set,
                "verbose": 0,
                # TODO: find how to add mape metric to catboost
                "early_stopping_rounds": early_stopping_rounds,
                "callbacks": [],
            },
            "XGBRegressor": {
                "eval_set": [eval_set],
                "verbose": 0,
            },
        }
        if model_name in fit_params:
            return fit_params[model_name]
        else:
            return {}

    def get_init_params(self, model):
        model_name = type(model()).__name__
        init_params = {
            "LGBMRegressor": {"verbosity": -1, "random_state": 42, "verbose_eval": -1},
            "CatBoostRegressor": {
                # "silent": True,
                "random_seed": 42,
                "eval_metric": "RMSE",
            },
            "XGBRegressor": {
                "verbosity": 0,
                "random_seed": 42,
                "verbose_eval": False,
                "eval_metric": "mape",
                "callbacks": [xgb.callback.EarlyStopping(rounds=5)],
            },
        }
        if model_name in init_params:
            return init_params[model_name]
        else:
            return {}


if __name__ == "__main__":
    # You can add some code here to test or use the models
    from sklearn.datasets import fetch_california_housing
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.model_selection import train_test_split

    model = RegressionModels()
    data = fetch_california_housing()
    X = data.data
    y = data.target

    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_, y_, test_size=0.5, random_state=42
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    print(mean_absolute_percentage_error(y_test, model.predict(X_test)))
    print(mean_absolute_percentage_error(y_test, model.predict_ensemble(X_test)))
