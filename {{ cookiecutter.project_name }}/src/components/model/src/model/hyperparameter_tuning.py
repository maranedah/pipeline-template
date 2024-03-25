from copy import deepcopy
from pathlib import Path

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
import yaml
from sklearn.model_selection import KFold, cross_val_score

from .callbacks import (
    MLFlowLogCallback,
    ModelPruningCallback,
    StopIfStudyDoesNotImproveCallback,
    StopWhenTrialKeepBeingPrunedCallback,
)
from .constants import MODELS
from .model import split_data


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
        self.metric = metrics
        self.model_init_params = model_init_params
        self.model_name = type(model()).__name__
        mlflow.set_experiment(self.model_name)

        self.study = optuna.create_study(
            direction="minimize",
            study_name=self.model_name,
            storage=f"sqlite:///{self.model_name}.db",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=0, n_warmup_steps=10, n_min_trials=10
            ),
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=100, multivariate=True, seed=42
            ),
        )

        hyperparams_path = Path(__file__).parent / "hyperparams" / "experiments.yml"
        experiments = yaml.safe_load(open(hyperparams_path, "r"))[self.model_name]

        self.base_trials = experiments["base_trials"]
        self.grid = self.parse_grid(experiments)

    def parse_grid(self, experiments):
        # Initializing the grid to match optuna's suggestions
        grid = {}
        for param, _range in experiments["grid"].items():
            grid[param] = {}
            # If any element is string
            if any(isinstance(v, str) for v in _range):
                grid[param]["type"] = "categorical"
                grid[param]["params"] = {"name": param, "choices": _range}
            # If all elements are int
            elif all(isinstance(v, int) for v in _range):
                grid[param]["type"] = "int"
                grid[param]["params"] = {
                    "name": param,
                    "low": min(_range),
                    "high": max(_range),
                }
            # If all elements are bool
            elif all(isinstance(v, bool) for v in _range):
                grid[param]["type"] = "categorical"
                grid[param]["params"] = {"name": param, "choices": _range}
            # If all elements are numeric
            elif all(isinstance(v, (float, int)) for v in _range):
                grid[param]["type"] = "float"
                grid[param]["params"] = {
                    "name": param,
                    "low": min(_range),
                    "high": max(_range),
                }
            else:
                raise TypeError("YAML suggestion grid has bad formatting")
        return grid

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

    def check_for_pruning(self, trial, X_train, y_train, suggested_params, fit_params):
        # Init pruning callback
        pruning_callback = ModelPruningCallback(self.model_name, trial)

        # Add pruning callback to fit_params
        fit_params_with_pruning = deepcopy(fit_params)
        fit_params_with_pruning["callbacks"].append(pruning_callback)

        # Init model
        model = self.model(**suggested_params, **self.model_init_params)

        # Fit model
        model.fit(X_train, y_train, **fit_params_with_pruning)

        # Catboost Pruning needs to be manually called
        if self.model_name == "CatBoostRegressor":
            pruning_callback.check_pruned()

    def score_model(self, X_train, y_train, suggested_params, fit_params):
        # Define Cross Validation folds
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Score model
        score = cross_val_score(
            estimator=self.model(**suggested_params, **self.model_init_params),
            X=X_train,
            y=y_train,
            scoring=self.metric,
            params=fit_params,
            cv=cv,
            n_jobs=1,
        ).mean()

        # sklearn makes positive functions to be negative
        score = score if score > 0 else -score
        return score

    def objective(self, trial, X_train, y_train, fit_params) -> float:
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

        # Parse hparams
        suggested_params = self.suggest_from_hyperparams_grid(trial)

        # Check if we need to prune the model
        self.check_for_pruning(trial, X_train, y_train, suggested_params, fit_params)

        # Score model
        score = self.score_model(X_train, y_train, suggested_params, fit_params)

        return score

    def run_tuning(
        self,
        X_train,
        y_train,
        fit_params,
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
        for _, trial in self.base_trials.items():
            self.study.enqueue_trial(trial, skip_if_exists=True)

        self.study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, fit_params),
            n_trials=suggestion_trials,
            timeout=int(round(60 * 60 * timeout_in_hours)),
            callbacks=[
                MLFlowLogCallback(metric_name=self.metric),
                StopWhenTrialKeepBeingPrunedCallback(threshold=1000),
                StopIfStudyDoesNotImproveCallback(threshold=2000),
            ],
            n_jobs=1,
        )
        return self.study


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
    tuning = HyperparameterTuning(model, metric="f1_macro", model_init_params={})
    fit_params = {
        "eval_set": (X_val, y_val),
        "callbacks": [
            lgb.early_stopping(stopping_rounds=5),
            # LightGBMPruningCallback(trial, "multi_logloss")
        ],
    }
    tuning.run_tuning(X_train, y_train, fit_params, suggestion_trials, timeout_in_hours)


# run_hyperparameter_tuning(
#    gcs_bucket=None,
#    preprocessed_data_uri="gs://python-project-bucket-test/preprocessed.gzip",
#    split_ratio="6:2:2",
#    model_str="LGBMClassifier",
#    suggestion_trials=200,
#    timeout_in_hours=0.25,
# )
