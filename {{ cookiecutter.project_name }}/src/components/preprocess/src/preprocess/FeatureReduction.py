import logging

import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA

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
    def __init__(self,  ignore_columns=[], threshold=0.95):
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
    def __init__(self,  ignore_columns=[]):
        self.ignore_columns = ignore_columns

    def __call__(self, df):
        columns = [col for col in df.columns if col not in self.ignore_columns]
        for col in columns:
            should_be_filtered = (
                df[col].drop_nulls().n_unique() == 1
            )
            if should_be_filtered:
                df = df.drop(col)
        return df
    

class FilterColumnsWithTooManyCategories:
    def __init__(self,  ignore_columns=[], threshold=10):
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
    def __init__(self,  ignore_columns=[], threshold=0.95):
        self.ignore_columns = ignore_columns
        self.threshold = threshold

    def __call__(self, df):
        columns = [col for col in df.columns if col not in self.ignore_columns]
        for col1 in columns:
            for col2 in [col for col in df.columns if col not in self.ignore_columns]:
                if self.should_be_filtered(df, col1, col2):
                    df = df.drop(col1)
                    break
        return df
    
    def should_be_filtered(self, df, col1, col2):
        same_column = col1 == col2
        col1_has_more_nulls = df[col1].is_null().sum() >= df[col2].is_null().sum()
        return not same_column and col1_has_more_nulls and self.get_corr_between_two_columns(df, col1, col2) > self.threshold

    def get_corr_between_two_columns(self, df, col1, col2):
        filtered_df = df.filter((pl.col(col1).is_not_null()) & (pl.col(col2).is_not_null()))
        corr = filtered_df.corr()[0,1]
        return corr