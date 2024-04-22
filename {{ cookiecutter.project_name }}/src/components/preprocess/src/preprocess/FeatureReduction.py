import concurrent.futures
import logging
import os

import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA
from TypeHandling import TypeOptimization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Format for displaying the date and time
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


class FilterCorrelatedColumns:
    def __init__(self, threshold=0.9, logging=False, ignore_columns=[]):
        self.ignore_columns = ignore_columns
        self.threshold = threshold
        self.logging = logging

    def get_corr_matrix(self, df):
        corr_matrix = np.zeros((df.shape[1], df.shape[1]))
        for i in range(df.shape[1]):
            for j in range(i, df.shape[1]):
                if i == j:
                    corr_matrix[i][j] = 1
                    continue
                filtered_df = df.filter(
                    (pl.col(df.columns[i]).is_not_null())
                    & (pl.col(df.columns[j]).is_not_null())
                )[[df.columns[i], df.columns[j]]]
                corr_matrix[i][j] = filtered_df.corr()[0, 1]
        return corr_matrix

    def filter_cols(self, df_pd, correlation_matrix):
        groups = []
        remaining_cols = list(df_pd.columns)
        while remaining_cols:
            col = remaining_cols.pop(0)
            group = [col]
            correlated_cols = [col]
            for c in remaining_cols:
                if correlation_matrix.loc[col, c] >= self.threshold:
                    group.append(c)
                    correlated_cols.append(c)
            groups.append(group)
            remaining_cols = [c for c in remaining_cols if c not in correlated_cols]
        # Filter groups with length = 1
        groups = [group for group in groups if len(group) > 1]

        for group in groups:
            n_nulls = {col: df_pd[col].isna().sum() for col in group}
            ordered_nulls_cols = list(
                dict(sorted(n_nulls.items(), key=lambda item: item[1])).keys()
            )
            df_pd = df_pd.drop(columns=ordered_nulls_cols[1:])
        if logging:
            logging.info(f"Finished filtering correlated columns, shape: {df_pd.shape}")
        return list(df_pd.columns)

    def process_sub_df(self, sub_df):
        correlation_matrix = sub_df.corr()
        return self.filter_cols(sub_df, correlation_matrix)

    def parallel_process(self, df, n_jobs=None):
        if n_jobs is None:
            n_jobs = os.cpu_count()

        columns = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            splits = np.linspace(0, df.shape[1], n_jobs)
            for start, end in zip(splits[:-1], splits[1:]):
                sub_df = df.iloc[:, int(start) : int(end)]
                futures.append(executor.submit(self.process_sub_df, sub_df))

            for future in concurrent.futures.as_completed(futures):
                columns.append(future.result())

        columns = [item for sublist in columns for item in sublist]
        return columns

    def __call__(self, df):
        if logging:
            logging.info(f"Started filtering correlated columns, shape: {df.shape}")
        # correlation_matrix = self.get_corr_matrix(df)
        df_pd = TypeOptimization(
            ignore_columns=self.ignore_columns, logging=self.logging
        )(df.to_pandas())
        columns = [col for col in df.columns if col not in self.ignore_columns]

        df_pd = df_pd[columns]
        columns = self.parallel_process(df_pd, n_jobs=os.cpu_count() + 1)
        columns = self.parallel_process(df_pd[columns], n_jobs=2 + 1)
        columns = self.process_sub_df(df_pd[columns])
        columns = [col for col in df.columns if col in [*columns, *self.ignore_columns]]
        return df[columns]


class FilterColumns:
    def __init__(self, logging: bool, ignore_columns=[]):
        self.logging = logging
        self.ignore_columns = ignore_columns

    def __call__(self, df):
        if self.logging:
            logging.info(f"Started filtering with DataFrame of shape: {df.shape}")
        columns_to_filter = []
        columns = [col for col in df.columns if col not in self.ignore_columns]
        for col in columns:
            should_be_filtered = (
                self.too_many_nulls(df, col)
                or self.only_one_value(df, col)
                or self.too_many_categories(df, col)
            )
            if should_be_filtered:
                columns_to_filter.append(col)
        for col in columns_to_filter:
            df = df.drop(col)
        if self.logging:
            logging.info(f"Ended filtering with DataFrame of shape: {df.shape}")
        return df

    def too_many_nulls(self, df, col, threshold=0.8):
        return df[col].dtype == pl.Null or df[col].is_null().mean() > threshold

    def only_one_value(self, df, col):
        return df[col].n_unique() == 1

    def too_many_categories(self, df, col, threshold=10):
        return df[col].dtype == pl.Categorical and df[col].n_unique() > threshold
