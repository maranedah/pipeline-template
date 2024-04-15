import logging
from itertools import product

import numpy as np
import optuna
from model.callbacks import StopIfStudyDoesNotImproveCallback


class ModelEnsemble:
    def __init__(self, models, task, scoring, direction):
        self.models = models
        self.n_models = len(models)
        self.agg_function = self.get_agg_function(task)
        self.pred_function = self.get_pred_function(task)
        self.scoring = scoring
        self.direction = direction

    def get_agg_function(self, task):
        agg_functions = {
            "binary_classification": (lambda predictions: predictions.mean(axis=0)),
            "regression": (lambda predictions: predictions.mean(axis=0)),
        }
        return agg_functions[task]

    def get_pred_function(self, task):
        pred_functions = {
            "binary_classification": (lambda model, X: model.predict_proba(X)[:, 1]),
            "regression": (lambda model, X: model.predict(X)),
        }
        return pred_functions[task]

    def objective(self, trial, predictions, y_val):
        # Create coefficients
        coefs = [trial.suggest_float(f"x{i}", -1.0, 1.0) for i in range(self.n_models)]
        # Expand dims to match dims with predictions
        coefs = np.expand_dims(np.array(coefs), axis=(1))

        # Generate weighted prediction
        weighted_predictions = predictions * coefs

        # Aggregate predictions
        y_hat = self.agg_function(weighted_predictions)
        score = self.scoring(y_val, y_hat)

        return score

    def ensemble_tuning(self, eval_set):
        X_val, y_val = eval_set

        # We compute predictions for each model
        predictions = [self.pred_function(model, X_val) for model in self.models]

        logging.info("Ensemble tuning")

        study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(n_startup_trials=100),
        )

        # Generate grid of combinations of models
        values = [-1.0, 0.0, 1.0]
        combinations = list(product(values,repeat=self.n_models))
        for combination in combinations:
            trial = {f"x{i}": coef for i, coef in enumerate(combination)}
            study.enqueue_trial(trial, skip_if_exists=True)

        study.optimize(
            lambda trial: self.objective(trial, predictions, y_val),
            callbacks=[StopIfStudyDoesNotImproveCallback(threshold=1000)],
        )
        self.coefficients = np.expand_dims(
            np.array(list(study.best_params.values())), axis=(1)
        )

    def predict_ensemble(self, X):
        predictions = [self.pred_function(model, X) for model in self.models]
        weighted_predictions = predictions * self.coefficients
        return self.agg_function(weighted_predictions)
