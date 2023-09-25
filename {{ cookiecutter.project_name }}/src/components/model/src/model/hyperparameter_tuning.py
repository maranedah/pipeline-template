import lightgbm as lgb 
import optuna 
import pandas as pd 
import sklearn 

from .constants import (
    HYPERPARAMETER_TUNING,
    N_SUGGESTION_TRIALS,
    OPTUNA_OPTIMIZATION_DIRECTION,
    OPTUNA_SCORE_FUNCTION,
    RANDOM_SEED,
    SUGGESTIONS_DICT,
    TIMEOUT_IN_HOURS,
)
from .storage import OptunaStorage

def parse_hyperparams_suggestions(
    params: str,
    trial: optuna.trial,
) -> dict[str, dict[str | int | float]]:
    params = SUGGESTIONS_DICT
    trial_suggest_functions = {
        "int": trial.suggest_int,
        "float": trial.suggest_float,
        "categorical": trial.suggest_categorical,
    }
    suggestions = {
        k: trial_suggest_functions[v["type"]](**v["params"])
        for k, v in params.items()
    }
    return suggestions 

def get_best_params_from_optuna(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    gcs_bucket: str,
):
    storage = OptunaStorage(gcs_bucket)
    current_best_params, gcs_params_url = storage.get_stored_params()
    if not HYPERPARAMETER_TUNING and current_best_params is not None:
        return current_best_params, gcs_params_url
    else:
        # split

        def objective(trial):
            suggestions = parse_hyperparams_suggestions(
                params = SUGGESTIONS_DICT, trial=trial
            )
            model = lgb.LGBMRegressor(random_state=[RANDOM_SEED], n_jobs=-1, **suggestions)
            score = sklearn.model_selection.cross_val_score(
                model,
                X_train,
                Y_train,
                n_jobs=-1,
                cv=split,
                scoring=OPTUNA_SCORE_FUNCTION,
            )
            mse = score.mean()
            return mse 
        
        study = optuna.create_study(direction=OPTUNA_OPTIMIZATION_DIRECTION)
        study.optimize(objective, n_trials=N_SUGGESTION_TRIALS, timeout=60*60*TIMEOUT_IN_HOURS)

        stored_url = storage.store_best_trial_params(study.best_trial.params)

        return study.best_trial.params, stored_url


