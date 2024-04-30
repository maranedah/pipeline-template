import logging
from math import isnan

import numpy as np
import pandas as pd
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Format for displaying the date and time
)


class DataTypeOptimizer:
    def __init__(self, ignore_columns=[], logging=False):
        self.ignore_columns = ignore_columns
        self.date_pattern = r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}"
        self.np_uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]
        self.np_int_types = [np.int8, np.int16, np.int32, np.int64]

    def optimize_numerical_column(self, df, col, np_types, df_types):
        c_min, c_max = df[col].min(), df[col].max()

        for df_dtype, dtype in zip(df_types, np_types):
            np_info = np.iinfo if np.issubdtype(dtype, np.integer) else np.finfo
            if c_min > np_info(dtype).min and c_max < np_info(dtype).max:
                return self.cast_column(df, col, df_dtype)

    def optimize_numerical_dtype(self, df, col):
        if self.is_null(df, col):
            df = self.cast_column(df, col, self.df_float_types[0])
        else:
            c_min = df[col].min()
            col_is_int = (df[col].round() == df[col]).all()
            col_is_positive = c_min > 0

            if col_is_int:
                if col_is_positive:
                    df = self.optimize_numerical_column(
                        df, col, self.np_uint_types, self.df_uint_types
                    )
                else:
                    df = self.optimize_numerical_column(
                        df, col, self.np_int_types, self.df_int_types
                    )
            else:
                df = self.optimize_numerical_column(
                    df, col, self.np_float_types, self.df_float_types
                )
        return df

    def __call__(self, df):
        df_ignore = df[self.ignore_columns]
        df = df[list(set(df.columns) - set(self.ignore_columns))]
        df = self.set_dates(df)
        df = self.set_categorical(df)
        df = self.set_numerical(df)
        df = self.concat_horizontally((df_ignore, df))
        return df

    def is_string(self, df, col):
        return df[col].dtype == self.string_type

    def is_date(self, df, col):
        return self.is_string(df, col) and df[col].str.contains(self.date_pattern).all()

    def is_categorical(self, df, col):
        return self.is_string(df, col) and not self.is_date(df, col)

    def set_categorical(self, df):
        for col in df.columns:
            if self.is_categorical(df, col):
                df = self.cast_column(df, col, self.category_type)
        return df

    def set_dates(self, df):
        for col in df.columns:
            if self.is_date(df, col):
                df = self.cast_column(df, col, self.date_type)
        return df

    def set_numerical(self, df):
        for col in df.columns:
            if self.is_numerical(df, col):
                df = self.optimize_numerical_dtype(df, col)
        return df

    def is_numerical(self, df, col):
        return df[col].dtype in self.df_numerical_types


class PandasDataTypeOptimizer(DataTypeOptimizer):
    def __init__(self, ignore_columns=[], logging=False):
        super().__init__(ignore_columns, logging)
        self.np_float_types = [np.float16, np.float32, np.float64]
        self.df_float_types = [*self.np_float_types]
        self.df_uint_types = [*self.np_uint_types]
        self.df_int_types = [*self.np_int_types]
        self.df_numerical_types = [
            *self.df_float_types,
            *self.df_uint_types,
            *self.df_int_types,
        ]
        self.category_type = "category"
        self.string_type = "object"
        self.date_type = "date"

    def is_null(self, df, col):
        return isnan(df[col].min())

    def concat_horizontally(self, items):
        return pd.concat(items, axis=1)

    def cast_column(self, df, col, dtype):
        if dtype == "date":
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = df[col].astype(dtype)
        return df

    def get_df_size(self, df):
        return df.memory_usage().sum() / 1024**2


class PolarsDataTypeOptimizer(DataTypeOptimizer):
    def __init__(self, ignore_columns=[], logging=False):
        super().__init__(ignore_columns, logging)
        self.np_float_types = [np.float32, np.float64]  # Polars doesn't accept float16
        self.df_float_types = [pl.Float32, pl.Float64]
        self.df_uint_types = [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
        self.df_int_types = [pl.Int8, pl.Int16, pl.Int32, pl.Int64]
        self.df_numerical_types = [
            *self.df_float_types,
            *self.df_uint_types,
            *self.df_int_types,
        ]
        self.category_type = pl.Categorical
        self.string_type = pl.String
        self.date_type = pl.Date

    def is_null(self, df, col):
        return df[col].dtype == pl.Null

    def concat_horizontally(self, items):
        return pl.concat(items, how="horizontal")

    def cast_column(self, df, col, dtype):
        return df.with_columns(pl.col(col).cast(dtype))

    def get_df_size(self, df):
        return df.estimated_size() / 1024**2
