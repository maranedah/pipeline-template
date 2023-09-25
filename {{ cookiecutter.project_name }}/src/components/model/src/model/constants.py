import pathlib 

import yaml

HYPERPARAMETER_TUNING = True
N_SUGGESTION_TRIALS = 100
TIMEOUT_IN_HOURS = 2
RANDOM_SEED = 1234

ROOT_FOLDER = pathlib.Path(__file__).parent
SUGGESTIONS_DICT = yaml.safe_load(open(ROOT_FOLDER / "hyperparams.yml", "r")) 

OPTUNA_SCORE_FUNCTION = "neg_mean_absolute_error"
OPTUNA_OPTIMIZATION_DIRECTION = "maximize"