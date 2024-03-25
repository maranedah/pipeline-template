import mlflow
import optuna
from optuna.integration import (
    CatBoostPruningCallback,
    LightGBMPruningCallback,
    XGBoostPruningCallback,
)


class MLFlowLogCallback:
    def __init__(self, metric_name):
        self.metric_name = metric_name

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            with mlflow.start_run():
                mlflow.log_params(trial.params)
                mlflow.log_param("trial_id", trial._trial_id)
                mlflow.log_metric("time_in_seconds", trial.duration.total_seconds())
                mlflow.log_metric(self.metric_name, trial.value)


class StopWhenTrialKeepBeingPrunedCallback:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            study.stop()


class StopIfStudyDoesNotImproveCallback:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial._trial_id - study.best_trial._trial_id > self.threshold:
            study.stop()
        else:
            self._consequtive_pruned_count += 1


class ModelPruningCallback:
    def __new__(cls, model_name, trial):
        pruning_callbacks = {
            "CatBoostRegressor": {
                "callback": CatBoostPruningCallback,
                "params": {"metric": "MAPE"},
            },
            "LGBMRegressor": {
                "callback": LightGBMPruningCallback,
                "params": {"metric": "mape"},
            },
            "XGBRegressor": {
                "callback": XGBoostPruningCallback,
                "params": {"observation_key": "validation_0-mape"},
            },
        }
        callback = pruning_callbacks[model_name]["callback"]
        params = pruning_callbacks[model_name]["params"]
        return callback(trial=trial, **params)
