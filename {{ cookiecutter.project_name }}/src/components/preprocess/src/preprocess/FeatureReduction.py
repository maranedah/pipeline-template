import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def dimensionality_reduction(self, df):
    bool_columns = df.select_dtypes(include=["bool"]).columns.tolist()
    n = len(bool_columns)
    sqrt_n = int(np.sqrt(n).round())

    pca = PCA(n_components=sqrt_n, random_state=42)
    pca_data = pca.fit_transform(df[bool_columns])
    pca_columns = pd.DataFrame(data=pca_data, columns=[f"x_{i}" for i in range(sqrt_n)])

    df.drop(columns=bool_columns, inplace=True)
    df = pd.concat([df, pca_columns], axis=1)

    return df


class FilterColumnsWithManyNulls:
    def __init__(self, ignore_columns=[], threshold=0.95):
        self.ignore_columns = ignore_columns
        self.threshold = threshold

    def __call__(self, df):
        columns = [col for col in df.columns if col not in self.ignore_columns]
        for col in columns:
            should_be_filtered = (
                df[col].dtype == pl.Null or df[col].is_null().mean() > self.threshold
            )
            if should_be_filtered:
                df = df.drop(col)
        return df


class FilterColumnsWithOnlyOneValue:
    def __init__(self, ignore_columns=[]):
        self.ignore_columns = ignore_columns

    def __call__(self, df):
        columns = [col for col in df.columns if col not in self.ignore_columns]
        for col in columns:
            should_be_filtered = df[col].drop_nulls().n_unique() == 1
            if should_be_filtered:
                df = df.drop(col)
        return df


class FilterColumnsWithTooManyCategories:
    def __init__(self, ignore_columns=[], threshold=10):
        self.ignore_columns = ignore_columns
        self.threshold = threshold

    def __call__(self, df):
        columns = [col for col in df.columns if col not in self.ignore_columns]
        for col in columns:
            should_be_filtered = (
                df[col].dtype == pl.Categorical and df[col].n_unique() >= self.threshold
            )
            if should_be_filtered:
                df = df.drop(col)
        return df


class FilterColumnsTooCorrelated:
    def __init__(self, ignore_columns=[], threshold=0.95):
        self.ignore_columns = ignore_columns
        self.threshold = threshold

    def __call__(self, df):
        columns = [col for col in df.columns if col not in self.ignore_columns]
        ignore_columns = [col for col in df.columns if col in self.ignore_columns]
        n_nulls = {col: df[col].is_null().sum() for col in columns}
        pairs = [
            [columns[i], columns[j], False]
            for i in range(len(columns))
            for j in range(i + 1, len(columns))
        ]

        should_remove = self.parallel_filtering(
            df, pairs, n_nulls, self.should_be_filtered
        )
        new_df = df[list(set(columns) - set(should_remove))]
        df = pl.concat((df[ignore_columns], new_df), how="horizontal")
        return df

    def should_be_filtered(self, df, col1, col2, n_nulls):
        if self.get_corr_between_two_columns(df, col1, col2) > self.threshold:
            if n_nulls[col1] <= n_nulls[col2]:
                return col1
            else:
                return col2
        else:
            return ""

    def get_corr_between_two_columns(self, df, col1, col2):
        corr = df.select(pl.corr(col1, col2)).item()
        return corr

    def process_pair(self, pair, df, should_remove, should_be_filtered, pairs, n_nulls):
        col1, col2, visited = pair
        if not visited:
            col_to_filter = should_be_filtered(df, col1, col2, n_nulls)
            if col_to_filter:
                should_remove.append(col_to_filter)
                for pair in pairs:
                    if col_to_filter in (pair[0], pair[1]):
                        pair[2] = True

    def parallel_filtering(self, df, pairs, n_nulls, should_be_filtered):
        should_remove = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.process_pair,
                    pair,
                    df,
                    should_remove,
                    should_be_filtered,
                    pairs,
                    n_nulls,
                )
                for pair in pairs
            ]
            for future in tqdm(futures):
                future.result()
        return should_remove
