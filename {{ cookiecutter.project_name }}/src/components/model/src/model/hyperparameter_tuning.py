import logging
import sys
from copy import deepcopy
from pathlib import Path
from time import time

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
import yaml
from optuna.integration import (
    CatBoostPruningCallback,
    LightGBMPruningCallback,
    XGBoostPruningCallback,
)
from sklearn.model_selection import KFold, cross_val_score

from .constants import MODELS
from .model import split_data

start = time()


class HyperparameterTuning:
    """
    Class for hyperparameter tuning using Optuna.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment for storing the optimization results.

    Attributes
    ----------
    storage : str
        Storage URL for the Optuna study database.
    study : optuna.Study
        Optuna study for hyperparameter optimization.
    base_trials : dict
        Dictionary containing base trials for the study.
    hyperparams_grid : dict
        Dictionary containing the hyperparameter search space grid.
    """

    storage: str
    study: optuna.Study
    base_trials: dict
    hyperparams_grid: dict

    def __init__(self, model, metrics, model_init_params):
        """
        Initialize the HyperparameterTuning object.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment for storing the optimization results.
        """
        self.model = model
        self.metrics = metrics
        self.model_init_params = model_init_params
        self.model_name = type(model()).__name__

        optuna.logging.get_logger("optuna").addHandler(
            logging.StreamHandler(sys.stdout)
        )
        self.study = optuna.create_study(
            direction="minimize",
            study_name="",
            storage=f"sqlite:///{self.model_name}.db",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=0, n_warmup_steps=10, n_min_trials=5
            ),
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=100, multivariate=True, seed=42
            ),
        )
        self.pruning_callbacks = {
            "CatBoostRegressor": {
                "callback": CatBoostPruningCallback,
                "params": {"metric": "RMSE"},
            },
            "LGBMRegressor": {
                "callback": LightGBMPruningCallback,
                "params": {"metric": "mape", "report_interval": 1},
            },
            "XGBRegressor": {
                "callback": XGBoostPruningCallback,
                "params": {"observation_key": "validation_0-mape"},
            },
        }
        mlflow.set_experiment(self.model_name)

        hyperparams_path = Path(__file__).parent / "hyperparams" / "experiments.yml"
        experiments = yaml.safe_load(open(hyperparams_path, "r"))[self.model_name]

        self.base_trials = experiments["base_trials"]

        # Initializing the grid to match optuna's suggestions
        self.grid = {}
        for param, _range in experiments["grid"].items():
            self.grid[param] = {}
            # If any element is string
            if any(isinstance(v, str) for v in _range):
                self.grid[param]["type"] = "categorical"
                self.grid[param]["params"] = {"name": param, "choices": _range}
            # If all elements are int
            elif all(isinstance(v, int) for v in _range):
                self.grid[param]["type"] = "int"
                self.grid[param]["params"] = {
                    "name": param,
                    "low": min(_range),
                    "high": max(_range),
                }
            # If all elements are numeric
            elif all(isinstance(v, (float, int)) for v in _range):
                self.grid[param]["type"] = "float"
                self.grid[param]["params"] = {
                    "name": param,
                    "low": min(_range),
                    "high": max(_range),
                }
            else:
                raise TypeError("YAML suggestion grid has bad formatting")

    def suggest_from_hyperparams_grid(
        self, trial: optuna.trial
    ) -> dict[str, dict[str, str | int | float]]:
        """
        Suggest hyperparameters for the trial from the hyperparameter search space grid.

        Parameters
        ----------
        trial : optuna.trial
            Optuna trial object.

        Returns
        -------
        dict
            A dictionary containing suggested hyperparameters for the trial.
        """
        trial_suggest_functions = {
            "int": trial.suggest_int,
            "float": trial.suggest_float,
            "categorical": trial.suggest_categorical,
        }
        suggestions = {
            k: trial_suggest_functions[v["type"]](**v["params"])
            for k, v in self.grid.items()
        }
        return suggestions

    def _get_pruning_callback(self, trial):
        return self.pruning_callbacks[self.model_name]["callback"](
            trial,
            **self.pruning_callbacks[self.model_name]["params"],
        )

    def objective(self, trial, X_train, y_train, fit_params, cv_splits) -> float:
        """
        Objective function for optimization.

        Parameters
        ----------
        trial : optuna.trial
            Optuna trial object.
        model : sklearn.base.BaseEstimator
            The machine learning model to evaluate.
        data : pd.DataFrame
            The input dataset.
        cross_validation_splits : int
            Number of cross-validation splits.

        Returns
        -------
        float
            The objective value (evaluation metric) to be minimized.
        """
        with mlflow.start_run(nested=True):
            suggested_params = self.suggest_from_hyperparams_grid(trial)
            model = self.model(**suggested_params, **self.model_init_params)

            # Run a model and see if we need to prune
            fit_params_with_pruning = deepcopy(fit_params)
            fit_params_with_pruning["callbacks"].append(
                self._get_pruning_callback(trial)
            )

            model.fit(X_train, y_train, **fit_params_with_pruning)
            # Cross Validation
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            score = cross_val_score(
                estimator=self.model(**suggested_params, **self.model_init_params),
                X=X_train,
                y=y_train,
                scoring=self.metrics,
                fit_params=fit_params,
                cv=cv,
                n_jobs=-1,
            ).mean()
            score = score if score > 0 else -score
            mlflow.log_params(suggested_params)
            mlflow.log_metric(self.metrics, score)
            if len(self.study.trials) > 1:
                best_mape = (
                    score if score < self.study.best_value else self.study.best_value
                )
                mlflow.log_metric("best_mape", best_mape)
                mlflow.log_param("time", time() - start)
            else:
                mlflow.log_metric("best_mape", score)
                mlflow.log_param("time", time() - start)

        return score

    def run_tuning(
        self,
        X_train,
        y_train,
        fit_params,
        cv_splits,
        suggestion_trials,
        timeout_in_hours,
    ):
        """
        Run hyperparameter tuning.

        Parameters
        ----------
        data : pd.DataFrame
            The input dataset.
        cross_validation_splits : int
            Number of cross-validation splits.
        suggestion_trials : int
            Number of trials for hyperparameter suggestions.
        timeout_in_hours : int
            Timeout for the optimization process in hours.

        Returns
        -------
        optuna.Study
            The completed Optuna study after hyperparameter tuning.
        """
        trials_df = self.study.trials_dataframe().copy()
        for _, trial in self.base_trials.items():
            if not self.is_trial_in_database(trial, trials_df):
                self.study.enqueue_trial(trial)

        self.study.optimize(
            lambda trial: self.objective(
                trial, X_train, y_train, fit_params, cv_splits
            ),
            n_trials=suggestion_trials,
            timeout=int(round(60 * 60 * timeout_in_hours)),
        )
        return self.study

    def is_trial_in_database(self, trial, trials_df):
        trial_exists = (
            len(trials_df) > 0
            and trials_df["system_attrs_fixed_params"]
            .astype(str)
            .isin([str(trial)])
            .any()
        )
        return trial_exists


def run_hyperparameter_tuning(
    gcs_bucket: str,
    preprocessed_data_uri: str,
    split_ratio: str,
    model_str: str,
    suggestion_trials: int,
    timeout_in_hours: float,
):
    # Read data
    df = pd.read_parquet(preprocessed_data_uri)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df, split_ratio)
    model = MODELS[model_str]
    cv_splits = 5
    tuning = HyperparameterTuning(model, metrics="f1_macro", model_init_params={})
    fit_params = {
        "eval_set": (X_val, y_val),
        "callbacks": [
            lgb.early_stopping(stopping_rounds=5),
            # LightGBMPruningCallback(trial, "multi_logloss")
        ],
    }
    tuning.run_tuning(
        X_train, y_train, fit_params, cv_splits, suggestion_trials, timeout_in_hours
    )


# run_hyperparameter_tuning(
#    gcs_bucket=None,
#    preprocessed_data_uri="gs://python-project-bucket-test/preprocessed.gzip",
#    split_ratio="6:2:2",
#    model_str="LGBMClassifier",
#    suggestion_trials=200,
#    timeout_in_hours=0.25,
# )
