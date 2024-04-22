import logging
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
from model.hyperparameter_tuning import HyperparameterTuning
from model.models.Ensemble import ModelEnsemble
from model.models.Scoring import cross_validation_score

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
mlflow.set_tracking_uri("http://192.168.1.200:5000")


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

    def best_models_hparams_tuning(
        self, X_train, y_train, best_models, eval_set, split
    ):
        mlflow.set_experiment("All Models")
        studies = []
        scores = []
        for model in best_models:
            study = HyperparameterTuning(
                model,
                metric=self.tuning_metric,
                model_init_params=self.get_init_params(model),
                split=split,
            ).run_tuning(
                X_train,
                y_train,
                None,
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

    def stratified_selection(self, labels, size):
        unique_labels = np.unique(labels)

        label_counts = {label: np.sum(labels == label) for label in unique_labels}

        desired_sample_size = {
            label: int(count * size) for label, count in label_counts.items()
        }

        stratified_indexes = []
        for label in unique_labels:
            indexes = np.where(labels == label)[0]
            selected_indexes = np.random.choice(  # noqa: NPY002
                indexes, size=desired_sample_size[label], replace=False
            )
            stratified_indexes.extend(selected_indexes)

        stratified_indexes = np.array(stratified_indexes)
        return stratified_indexes

    def reduce_dataset(self, X, y):
        dataset_in_mb = X.itemsize * X.size / 1024**2
        reduction_coefficient = 1
        while dataset_in_mb / reduction_coefficient > 20:
            reduction_coefficient *= 2
        small_len = len(X) // reduction_coefficient
        full_len = len(X)
        medium_len = int(np.exp((np.log(full_len) + np.log(small_len)) / 2))

        small = self.stratified_selection(y, small_len / full_len)
        medium = self.stratified_selection(y, medium_len / full_len)
        full = self.stratified_selection(y, full_len / full_len)

        # return indexes
        return small, medium, full

    def fit(self, X_train, y_train, eval_set):
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier

        # df_models_score = self.score_models(X_train, y_train, eval_set)
        # best_models = self.get_top_k_models(df_models_score, top_k=3)

        dataset_sizes = self.reduce_dataset(X_train, y_train)

        for split, dataset_size in zip(
            ["small", "medium", "full"][2:], dataset_sizes[2:]
        ):
            best_models = [XGBClassifier, LGBMClassifier, CatBoostClassifier]
            top_models_score, studies = self.best_models_hparams_tuning(
                X_train[dataset_size, 1:],
                y_train[dataset_size],
                best_models,
                eval_set,
                split,
            )
            print(top_models_score)
        self.best_models = self.init_tuned_models(
            X_train[:, 1:], y_train, best_models, studies, eval_set
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
