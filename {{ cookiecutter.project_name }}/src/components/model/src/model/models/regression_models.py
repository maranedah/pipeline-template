# regression_models.py

# Importing necessary libraries
import lightgbm as lgb
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from model.hyperparameter_tuning import HyperparameterTuning
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class RegressionModels:
    def __init__(self):
        self.models = [
            LinearRegression,
            Ridge,
            Lasso,
            ElasticNet,
            DecisionTreeRegressor,
            RandomForestRegressor,
            GradientBoostingRegressor,
            SVR,
            KNeighborsRegressor,
            MLPRegressor,
            XGBRegressor,
            LGBMRegressor,
            CatBoostRegressor,  # eval_set
        ]
        self.metrics = [
            "neg_mean_squared_error",
            "neg_root_mean_squared_error",
            "neg_mean_absolute_error",
            "r2",
            "neg_mean_absolute_percentage_error",
        ]
        self.tuning_metric = "neg_mean_absolute_percentage_error"
        self.cv_splits = 5

    def fit(self, X, y, eval_set):
        # Init all models with default params
        models = [model() for model in self.models]

        # Cross Validation for each default model
        cv_models_info = [
            cross_validate(
                estimator=model,
                X=X,
                y=y,
                scoring=self.metrics,
                cv=self.cv_splits,
                n_jobs=-1,
            )
            for model in models
        ]
        models_performance = [
            {
                "model_name": type(model).__name__,
                "test_score": cv_info["test_score"].mean(),
                "fit_time": cv_info["fit_time"].mean(),
            }
            for model, cv_info in zip(models, cv_models_info)
        ]

        df = pd.DataFrame(models_performance)
        df.to_csv("Regression Models Preliminary Results.csv")
        # Probably add a bar plot here
        ###

        # Get top 3 models
        top_3_indexes = df.nlargest(3, "test_score").index
        top_3_models = self.models.iloc[top_3_indexes]

        # Hyperparameter tuning of best 3 models
        suggestion_trials = 100
        timeout_in_hours = 2
        model_studies = [
            HyperparameterTuning(model, metrics=self.tuning_metric).run_tuning(
                X,
                y,
                self.get_fit_params(model, eval_set, early_stopping_rounds=5),
                self.cv_splits,
                suggestion_trials,
                timeout_in_hours,
            )
            for model in top_3_models
        ]

        # Assign optimized top 3 best models to object
        self.top_3_models = [
            model(study.best_params)
            for study, model in zip(model_studies, top_3_models)
        ]

        # Save top 3 models results
        top_3_performances = [
            {"model_name": type(model).__name__, self.tuning_metric: study.best_value}
            for study, model in zip(model_studies, top_3_models)
        ]
        top_3_df = pd.DataFrame(top_3_performances)
        top_3_df.to_csv("Top 3 Tuned Model Performances.csv")

        # Get best model
        self.best_model = self.top_3_models.iloc[
            top_3_df.nlargest(1, self.tuning_metric).index
        ]

    def predict(self, X):
        self.best_model.predict(X)

    def get_fit_params(model, eval_set, early_stopping_rounds):
        model_name = type(model()).__name__
        fit_params = {
            "LGBMRegressor": {
                "eval_set": eval_set,
                "callbacks": [
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds)
                ],
            },
            "CatBoostRegressor": {"eval_set": eval_set},
            "XGBRegressor": {"eval_set": eval_set},
        }
        return fit_params[model_name]


if __name__ == "__main__":
    # You can add some code here to test or use the models
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    regression_models = RegressionModels()
    data = fetch_california_housing()
    X = data.data
    y = data.target

    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.2)
    regression_models.fit(X_train, y_train, eval_set=(X_val, y_val))
