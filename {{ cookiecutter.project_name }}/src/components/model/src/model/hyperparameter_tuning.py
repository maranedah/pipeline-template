import optuna


def parse_hyperparams_suggestions(
    params: str,
    trial: optuna.trial,
    suggestions_dict: dict[str, any],
) -> dict[str, dict[str, str | int | float]]:
    params = suggestions_dict
    trial_suggest_functions = {
        "int": trial.suggest_int,
        "float": trial.suggest_float,
        "categorical": trial.suggest_categorical,
    }
    suggestions = {
        k: trial_suggest_functions[v["type"]](**v["params"]) for k, v in params.items()
    }
    return suggestions


# suggestions_dict = yaml.safe_load(open(ROOT_FOLDER / "hyperparams.yml", "r"))
