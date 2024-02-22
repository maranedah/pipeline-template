import lightgbm as lgb

RANDOM_STATE = 42
EVALUATION_METRICS = ["accuracy", "f1", "precision", "recall"]
MODELS = {
    "LGBMClassifier": lgb.LGBMClassifier,
    "LGBMRegressor": lgb.LGBMRegressor,
}
