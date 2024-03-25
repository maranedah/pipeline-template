import logging
from datetime import datetime, timezone

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from model.callbacks import StopIfStudyDoesNotImproveCallback
from model.hyperparameter_tuning import HyperparameterTuning
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RegressionModels:
    def __init__(self):
        self.models = [
            # LinearRegression,
            # Ridge,
            # Lasso,
            # ElasticNet,
            # DecisionTreeRegressor,
            # RandomForestRegressor,
            # GradientBoostingRegressor,
            # SVR,
            # KNeighborsRegressor,
            # MLPRegressor,
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
        self.suggestion_trials = 1
        self.timeout_in_hours = 10

    def init_models(self):
        logging.info("Instantiating models with default parameters")
        return [model(**self.get_init_params(model)) for model in self.models]

    def get_scores_from_mlflow(self, experiment_timestamp):
        runs_df = mlflow.search_runs()
        time_filter = runs_df.start_time.dt.tz_convert(None) > experiment_timestamp
        runs_df = runs_df[time_filter]
        metrics_cols = [col for col in runs_df.columns if col.startswith("metrics.")]
        renaming = [
            col.split(".")[-1] for col in runs_df.columns if col.startswith("metrics.")
        ]
        models_score = runs_df[metrics_cols].rename(
            columns=dict(zip(metrics_cols, renaming))
        )
        return models_score

    def score_models(self, models, X, y, eval_set):
        mlflow.set_experiment("All Models")

        experiment_timestamp = np.datetime64(datetime.now(timezone.utc), "ns")

        cv = KFold(n_splits=self.cv_splits, shuffle=True, random_state=42)
        for model in models:
            with mlflow.start_run():
                logging.info(f"Training model {type(model).__name__}")
                result = cross_validate(
                    estimator=model,
                    X=X,
                    y=y,
                    scoring=self.metrics,
                    cv=cv,
                    params=self.get_fit_params(type(model), eval_set),
                    n_jobs=-1,
                )
                mlflow.log_param("model", type(model).__name__)
                for k, v in result.items():
                    mlflow.log_metric(k, v.mean())

        logging.info("Finished Training...")
        models_score = self.get_scores_from_mlflow(experiment_timestamp)

        return models_score

    def get_top_k_models(self, df_models_score, top_k):
        top_k_models = (
            df_models_score.nlargest(n=top_k, columns=f"test_{self.tuning_metric}")
            .index.map(lambda x: (len(self.models) - 1) - x)
            .map(self.models.__getitem__)
            .tolist()
        )
        return top_k_models

    def best_models_hparams_tuning(self, best_models, eval_set):
        mlflow.set_experiment("All Models")
        studies = []
        scores = []
        for model in best_models:
            study = HyperparameterTuning(
                model,
                metrics=self.tuning_metric,
                model_init_params=self.get_init_params(model),
            ).run_tuning(
                X_train,
                y_train,
                self.get_fit_params(model, eval_set),
                self.suggestion_trials,
                self.timeout_in_hours,
            )
            scores.append(study.best_value)
            studies.append(study)
        models_score = pd.DataFrame(scores, columns=["score"])
        return models_score, studies

    def init_tuned_models(self, best_models, studies, eval_set):
        best_models = [
            model(**study.best_params, **self.get_init_params(model)).fit(
                X, y, **self.get_fit_params(model, eval_set)
            )
            for study, model in zip(studies, best_models)
        ]
        return best_models

    def ensemble_tuning(self, eval_set):
        X_val, Y_val = eval_set
        logging.info("Ensemble tuning")

        def objective(trial, predictions):
            # Define the search space for parameters
            a = trial.suggest_float("a", -1.0, 1.0)
            b = trial.suggest_float("b", -1.0, 1.0)
            c = trial.suggest_float("c", -1.0, 1.0)

            # Example objective function (can be replaced with any other function)
            weighted_predictions = [
                prediction * coefficient
                for prediction, coefficient in zip(predictions, [a, b, c])
            ]
            predictions = np.array(weighted_predictions).sum(axis=0)
            score = mean_absolute_percentage_error(Y_val, predictions)
            return score  # Perform optimization

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=100),
        )
        predictions = [model.predict(X_val) for model in self.best_models]
        study.optimize(
            lambda trial: objective(trial, predictions),
            n_trials=None,
            callbacks=[StopIfStudyDoesNotImproveCallback(threshold=1000)],
            n_jobs=1,
        )
        best_params = study.best_params
        self.mix_coefficients = [
            best_params["a"],
            best_params["b"],
            best_params["c"],
        ]

    def fit(self, X, y, eval_set):
        models = self.init_models()
        df_models_score = self.score_models(models, X, y, eval_set)
        best_models = self.get_top_k_models(df_models_score, top_k=3)
        top_models_score, studies = self.best_models_hparams_tuning(
            best_models, eval_set
        )
        print(top_models_score)
        self.best_models = self.init_tuned_models(best_models, studies, eval_set)

        self.ensemble_tuning(eval_set)
        # Get best model
        self.best_model = self.best_models[
            top_models_score.nsmallest(1, "score").index.tolist()[0]
        ]

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_ensemble(self, X):
        return np.array(
            [
                model.predict(X) * coef
                for model, coef in zip(self.best_models, self.mix_coefficients)
            ]
        ).sum(axis=0)

    def predict_ensemble_mean(self, X):
        return np.array([model.predict(X) for model in self.best_models]).mean(axis=0)

    def get_fit_params(self, model, eval_set):
        model_name = type(model()).__name__
        fit_params = {
            "LGBMRegressor": {
                "eval_set": eval_set,
                "eval_metric": "mape",
                "callbacks": [lgb.early_stopping(stopping_rounds=10, verbose=False)],
            },
            "CatBoostRegressor": {
                "eval_set": eval_set,
                "verbose": 0,
                "early_stopping_rounds": 10,
                "callbacks": [],
            },
            "XGBRegressor": {
                "eval_set": [eval_set],
                "verbose": 0,
                "callbacks": [xgb.callback.EarlyStopping(rounds=10)],
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
                "random_seed": 42,
                "eval_metric": "MAPE",
            },
            "XGBRegressor": {
                "verbosity": 0,
                "random_seed": 42,
                "verbose_eval": False,
                "eval_metric": "mape",
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
    print(mean_absolute_percentage_error(y_test, model.predict_ensemble_mean(X_test)))
    print(mean_absolute_percentage_error(y_test, model.predict_ensemble(X_test)))
