import logging

import numpy as np
import pandas as pd
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Format for displaying the date and time
)


class TypeHandling:
    def __init__(self, ignore_columns=None, logging=False):
        self.type_optimization = TypeOptimization(
            ignore_columns=ignore_columns, logging=False
        )
        self.ignore_columns = ignore_columns
        self.logging = logging

    def __call__(self, df):
        if self.logging:
            logging.info(
                f"Starting type handling for df of {int(self.get_dataframe_size(df))}MB"
            )
        df = self.set_dates(df)
        df = self.set_categorical(df)
        df = self.type_optimization(df)
        if self.logging:
            logging.info(
                f"Ended type handling, new df of {int(self.get_dataframe_size(df))}MB"
            )
        return df

    def get_dataframe_size(self, df):
        if type(df) == pd.DataFrame:
            size_in_mb = df.memory_usage().sum() / 1024**2
        elif type(df) == pl.DataFrame:
            size_in_mb = df.estimated_size() / 1024**2
        return size_in_mb

    def set_dates(self, df):
        pattern = r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}"
        for col in df.columns:
            if df[col].dtype == pl.String and df[col].str.contains(pattern).all():
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    def set_categorical(self, df):
        str_columns = [
            col for col, dtype in zip(df.columns, df.dtypes) if "String" in str(dtype)
        ]
        for col in str_columns:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
        return df


class PandasMemoryReduction:
    def __init__(self, ignore_columns=None):
        self.ignore_columns = ignore_columns

    def optimize_float_dtype(self, df, col):
        c_min, c_max = df[col].min(), df[col].max()
        if (df[col].round() == df[col]).all():
            return df[col].astype(np.int64)
        elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
            return df[col].astype(np.float16)
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            return df[col].astype(np.float32)
        else:
            return df[col].astype(np.float64)

    def optimize_int_dtype(self, df, col):
        c_min, c_max = df[col].min(), df[col].max()
        if c_min is not None and c_min >= 0:
            if c_max < np.iinfo(np.uint8).max:
                return df[col].astype(np.uint8)
            elif c_max < np.iinfo(np.uint16).max:
                return df[col].astype(np.uint16)
            elif c_max < np.iinfo(np.uint32).max:
                return df[col].astype(np.uint32)
            else:
                return df[col].astype(np.uint64)
        else:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                return df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                return df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                return df[col].astype(np.int32)
            else:
                return df[col].astype(np.int64)

    def type_optimization(self, df):
        columns = [col for col in df.columns if col not in self.ignore_columns]
        float_columns = df[columns].select_dtypes(
            include=["float64", "float32", "float16"]
        )
        for col in float_columns:
            df[col] = self.optimize_float_dtype(df, col)

        int_columns = df[columns].select_dtypes(
            include=["int64", "int32", "int16", "int8"]
        )
        for col in int_columns:
            df[col] = self.optimize_int_dtype(df, col)

        object_columns = df[columns].select_dtypes(include=["object"])
        for col in object_columns:
            df[col].astype("category")
        return df


class PolarsMemoryReduction:
    def __init__(self, ignore_columns=None):
        self.ignore_columns = ignore_columns

    def optimize_int_dtype(self, df, col):
        c_min, c_max = df[col].min(), df[col].max()

        if c_min >= 0:
            if c_max < np.iinfo(np.uint8).max:
                return df.with_columns(pl.col(col).cast(pl.UInt8))
            elif c_max < np.iinfo(np.uint16).max:
                return df.with_columns(pl.col(col).cast(pl.UInt16))
            elif c_max < np.iinfo(np.uint32).max:
                return df.with_columns(pl.col(col).cast(pl.UInt32))
            else:
                return df.with_columns(pl.col(col).cast(pl.UInt64))
        else:
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                return df.with_columns(pl.col(col).cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                return df.with_columns(pl.col(col).cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                return df.with_columns(pl.col(col).cast(pl.Int32))
            else:
                return df.with_columns(pl.col(col).cast(pl.Int64))

    def optimize_float_dtype(self, df, col):
        c_min, c_max = df[col].min(), df[col].max()
        if (df[col].round() == df[col]).all():
            return df.with_columns(pl.col(col).cast(pl.Int64))
        elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
            return df.with_columns(pl.col(col).cast(pl.Float32))
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            return df.with_columns(pl.col(col).cast(pl.Float32))
        else:
            return df.with_columns(pl.col(col).cast(pl.Float64))

    def type_optimization(self, df):
        columns = [col for col in df.columns if col not in self.ignore_columns]
        float_columns = [
            col
            for col, dtype in zip(columns, df[columns].dtypes)
            if "Float" in str(dtype)
        ]
        for col in float_columns:
            df = self.optimize_float_dtype(df, col)

        int_columns = [
            col
            for col, dtype in zip(columns, df[columns].dtypes)
            if "Int" in str(dtype)
        ]
        for col in int_columns:
            if col == "case_id" or df[col].min() is None:
                continue
            df = self.optimize_int_dtype(df, col)

        str_columns = [
            col
            for col, dtype in zip(columns, df[columns].dtypes)
            if "String" in str(dtype)
        ]
        for col in str_columns:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
        return df


class TypeOptimization:
    def __init__(self, ignore_columns=None, logging=False):
        self.ignore_columns = ignore_columns
        self.logging = logging

    def __call__(self, df):
        if logging:
            logging.info(
                f"Optimizing dataframe with size {int(self.get_dataframe_size(df))}MB"
            )
        if type(df) == pd.DataFrame:
            df = PandasMemoryReduction(
                ignore_columns=self.ignore_columns
            ).type_optimization(df)
        elif type(df) == pl.DataFrame:
            df = PolarsMemoryReduction(
                ignore_columns=self.ignore_columns
            ).type_optimization(df)
        else:
            raise NotImplementedError
        if logging:
            logging.info(
                f"Optimization ended with size {int(self.get_dataframe_size(df))}MB"
            )
        return df

    def get_dataframe_size(self, df):
        if type(df) == pd.DataFrame:
            size_in_mb = df.memory_usage().sum() / 1024**2
        elif type(df) == pl.DataFrame:
            size_in_mb = df.estimated_size() / 1024**2
        return size_in_mb
