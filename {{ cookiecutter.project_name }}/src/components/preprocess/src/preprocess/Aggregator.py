import polars as pl
from DataTypeOptimizer import type_optimizer_decorator


class Aggregator:
    def __init__(self, key_column, ignore_columns):
        self.key_column = key_column
        self.ignore_columns = ignore_columns

    @type_optimizer_decorator
    def __call__(self, df):
        groups = df.group_by(self.key_column)
        # If n_groups == n_data, skip, nothing to aggregate
        if df.shape[0] == groups.len().shape[0]:
            return df
        else:
            # Aggregate n_count, mean, min, max, etc
            df = df.group_by(self.key_column).agg(self.get_exprs(df))
            return df

    def get_exprs(self, df):
        columns = [col for col in df.columns if col not in self.ignore_columns]
        numerical_columns = [
            col
            for col, dtype in zip(df[columns].columns, df[columns].dtypes)
            if "Float" in str(dtype) or "Int" in str(dtype)
        ]

        # Length of each group
        expr_len = [pl.len()]

        # Numerical columns
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in numerical_columns]
        expr_std = [pl.std(col).alias(f"std_{col}") for col in numerical_columns]
        return expr_len + expr_mean + expr_std
