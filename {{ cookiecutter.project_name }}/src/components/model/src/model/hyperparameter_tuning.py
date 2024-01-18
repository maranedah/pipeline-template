import pathlib

import optuna
import yaml
from sklearn.model_selection import cross_val_score


class HyperparameterTuning:
    storage: str
    study: optuna.Study
    base_trials: dict
    hyperparams_grid: dict

    def __init__(self, experiment_name: str):
        self.study = optuna.create_study(
            direction="minimize",
            study_name="",
            storage=f"sqlite:///{experiment_name}.db",
            load_if_exists=True,
        )

        hyperparams_path = pathlib.Path(__file__).parent / "hyperparams"
        base_trials_path = hyperparams_path / "base_trials" / f"{experiment_name}.yml"
        grid_path = hyperparams_path / "grid" / f"{experiment_name}.yml"

        self.base_trials = yaml.safe_load(open(base_trials_path, "r"))
        self.grid = yaml.safe_load(open(grid_path, "r"))

    def suggest_from_hyperparams_grid(
        self, trial: optuna.trial
    ) -> dict[str, dict[str, str | int | float]]:
        trial_suggest_functions = {
            "int": trial.suggest_int,
            "float": trial.suggest_float,
            "categorical": trial.suggest_categorical,
        }
        suggestions = {
            k: trial_suggest_functions[v["type"]](**v["params"])
            for k, v in self.hyperparams_grid.items()
        }
        return suggestions

    def objective(self, trial, model, data, cross_validation_splits) -> float:
        suggested_params = self.suggest_from_hyperparams_grid(trial)
        score = cross_val_score(
            estimator=model(**suggested_params),
            X=data.drop(columns="y"),
            y=data["y"],
            scoring="f1_macro",
            cv=cross_validation_splits,
            n_jobs=1,
        ).mean()
        return score

    def run_tuning(
        self, data, cross_validation_splits, suggestion_trials, timeout_in_hours
    ):
        trials_df = self.study.trials_dataframe().copy()
        for _, trial in self.base_trials.items():
            if not self.is_trial_in_database(trial, trials_df):
                self.study.enqueue_trial(trial)

        self.study.optimize(
            lambda trial: self.objective(trial, data, cross_validation_splits),
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
