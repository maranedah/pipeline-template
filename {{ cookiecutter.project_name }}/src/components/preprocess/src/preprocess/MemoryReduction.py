import numpy as np
import pandas as pd
import polars as pl
from sklearn.decomposition import PCA


class MemoryReduction:
    def pandas_optimize_float_dtype(self, df, col):
        c_min, c_max = df[col].min(), df[col].max()
        if (df[col].round() == df[col]).all():
            return df[col].astype(np.int64)
        elif c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
            return df[col].astype(np.float16)
        elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            return df[col].astype(np.float32)
        else:
            return df[col].astype(np.float64)

    def pandas_optimize_int_dtype(self, df, col):
        c_min, c_max = df[col].min(), df[col].max()
        if c_min > 0:
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

    def pandas_type_optimization(self, df):
        float_columns = df.select_dtypes(include=["float64", "float32", "float16"])
        for col in float_columns:
            df[col] = self.pandas_optimize_float_dtype(df, col)

        int_columns = df.select_dtypes(include=["int64", "int32", "int16", "int8"])
        for col in int_columns:
            df[col] = self.pandas_optimize_int_dtype(df, col)

        object_columns = df.select_dtypes(include=["object"])
        for col in object_columns:
            df[col].astype("category")
        return df

    def polars_type_optimization(self, df):
        for col in df.columns:
            col_type = df[col].dtype
            non_numeric_columns = ["Date", "Null"]
            if str(col_type) in non_numeric_columns or col == "case_id":
                continue

            if str(col_type) == "Boolean":
                df = df.with_columns(pl.col(col).cast(pl.Int8))
                continue

            if str(col_type) == "String":
                df = df.with_columns(pl.col(col).cast(pl.Categorical))
                continue

            if "Int" in str(col_type) or "Float" in str(col_type):
                c_min, c_max = df[col].min(), df[col].max()
                if c_min is None:
                    continue
                if "Int" in str(col_type) or df[col].eq(df[col].round()).all():
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df = df.with_columns(pl.col(col).cast(pl.Int8))
                    elif (
                        c_min > np.iinfo(np.int16).min
                        and c_max < np.iinfo(np.int16).max
                    ):
                        df = df.with_columns(pl.col(col).cast(pl.Int16))
                    elif (
                        c_min > np.iinfo(np.int32).min
                        and c_max < np.iinfo(np.int32).max
                    ):
                        df = df.with_columns(pl.col(col).cast(pl.Int32))
                    elif (
                        c_min > np.iinfo(np.int64).min
                        and c_max < np.iinfo(np.int64).max
                    ):
                        df = df.with_columns(pl.col(col).cast(pl.Int64))
                else:
                    if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                    ):
                        df = df.with_columns(pl.col(col).cast(pl.Float32))
                    elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                    ):
                        df = df.with_columns(pl.col(col).cast(pl.Float32))
                    else:
                        df = df.with_columns(pl.col(col).cast(pl.Float64))
            else:
                continue
        return df

    def type_optimization(self, df):
        if type(df) == pd.DataFrame:
            df = self.pandas_type_optimization(df)
        elif type(df) == pl.DataFrame:
            df = self.polars_type_optimization(df)
        else:
            raise NotImplementedError
        return df

    def pandas_filter_cols(self, df):
        raise NotImplementedError

    def filter_cols(self, df):
        if type(df) == pd.DataFrame:
            df = self.pandas_filter_cols(df)
        elif type(df) == pl.DataFrame:
            df = self.polars_filter_cols(df)
        else:
            raise NotImplementedError
        return df

    def dimensionality_reduction(self, df):
        bool_columns = df.select_dtypes(include=["bool"]).columns.tolist()
        n = len(bool_columns)
        sqrt_n = int(np.sqrt(n).round())

        pca = PCA(n_components=sqrt_n, random_state=42)
        pca_data = pca.fit_transform(df[bool_columns])
        pca_columns = pd.DataFrame(
            data=pca_data, columns=[f"x_{i}" for i in range(sqrt_n)]
        )

        df.drop(columns=bool_columns, inplace=True)
        df = pd.concat([df, pca_columns], axis=1)

        return df

    def polars_filter_cols(self, df):
        numeric_columns = [
            pl.Int64,
            pl.Int32,
            pl.Int16,
            pl.Int8,
            pl.Float64,
            pl.Float32,
        ]

        # Filter out the columns with numeric data types
        ignore_columns = ["target", "case_id", "WEEK_NUM"]
        numeric_columns = [
            column
            for column, dtype in zip(df.columns, df.dtypes)
            if dtype in numeric_columns
        ]

        for col in df.columns:
            if col not in ignore_columns:
                isnull = df[col].is_null().mean()
                if isnull > 0.8:
                    df = df.drop(col)

        for col in df.columns:
            freq = df[col].n_unique()
            if freq == 1:
                df = df.drop(col)

        for col in df.columns:
            if (col not in ignore_columns) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if freq > 200:
                    df = df.drop(col)
        return df
