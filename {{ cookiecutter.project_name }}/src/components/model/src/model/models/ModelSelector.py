import logging
from datetime import datetime, timezone

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from catboost import Pool
from model.hyperparameter_tuning import HyperparameterTuning
from model.models.Ensemble import ModelEnsemble
from model.models.Scoring import cross_validation_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelSelector:
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

    def score_models(self, X_train, y_train, eval_set):
        mlflow.set_experiment("All Models")

        experiment_timestamp = np.datetime64(datetime.now(timezone.utc), "ns")
        for model in self.models:
            with mlflow.start_run():
                logging.info(f"Training model {type(model()).__name__}")
                result = cross_validation_score(
                    estimator=model(**self.get_init_params(model)),
                    X=X_train,
                    y=y_train,
                    score_funcs=self.metrics,
                    n_splits=self.cv_splits,
                    fit_params=self.get_fit_params(type(model()), eval_set),
                    random_state=self.random_state,
                )
                mlflow.log_param("model", type(model()).__name__)
                mlflow.log_metrics(result)
        logging.info("Finished Training...")
        models_score = self.get_scores_from_mlflow(experiment_timestamp)

        return models_score

    def get_top_k_models(self, df_models_score, top_k):
        top_k_models = (
            df_models_score.nlargest(n=top_k, columns=self.tuning_metric.__name__)
            .index.map(lambda x: (len(self.models) - 1) - x)
            .map(self.models.__getitem__)
            .tolist()
        )
        return top_k_models

    def best_models_hparams_tuning(self, X_train, y_train, best_models, eval_set):
        mlflow.set_experiment("All Models")
        studies = []
        scores = []
        for model in best_models:
            study = HyperparameterTuning(
                model,
                metric=self.tuning_metric,
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

    def init_tuned_models(self, X_train, y_train, best_models, studies, eval_set):
        best_models = [
            model(**study.best_params, **self.get_init_params(model)).fit(
                X_train, y_train, **self.get_fit_params(model, eval_set)
            )
            for study, model in zip(studies, best_models)
        ]
        return best_models

    def set_data(self, X, y, eval_set):
        # CatBoost
        X_train = Pool(data=X, label=y)
        X_valid = Pool(data=eval_set[0], label=eval_set[1])

        self.catboost_data = {"train": (X_train, None), "valid": (X_valid, None)}

        # LGBM
        train_data = lgb.Dataset(X, label=y)
        valid_data = lgb.Dataset(eval_set[0], label=eval_set[1])
        self.lgbm_data = {
            "train": (train_data.data, train_data.label),
            "valid": (valid_data.data, valid_data.label),
        }

        # SKLearn
        self.sklearn_data = {"train": (X, y), "valid": eval_set}

    def get_data(self, model_name, set_):
        if "Catboost" in model_name:
            return self.catboost_data[set_]
        elif "LGBM" in model_name:
            return self.lgbm_data[set_]
        else:
            return self.sklearn_data[set_]

    def fit(self, X_train, y_train, eval_set):
        self.set_data(X_train, y_train, eval_set)
        df_models_score = self.score_models(X_train, y_train, eval_set)
        best_models = self.get_top_k_models(df_models_score, top_k=3)
        top_models_score, studies = self.best_models_hparams_tuning(
            X_train, y_train, best_models, eval_set
        )
        print(top_models_score)
        self.best_models = self.init_tuned_models(
            X_train, y_train, best_models, studies, eval_set
        )

        self.model_ensemble = ModelEnsemble(
            self.best_models, "binary_classification", self.tuning_metric, "maximize"
        )
        self.model_ensemble.ensemble_tuning(eval_set)

        # Get best model
        self.best_model = self.best_models[
            top_models_score.nlargest(1, "score").index.tolist()[0]
        ]

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_ensemble(self, X):
        return self.model_ensemble.predict_ensemble(X)
