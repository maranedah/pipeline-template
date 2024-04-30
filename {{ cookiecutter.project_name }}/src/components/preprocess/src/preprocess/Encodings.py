import polars as pl

from .DataTypeOptimizer import PolarsDataTypeOptimizer


class Encodings:
    def __init__(
        self, weekday=False, day_of_month=False, day_of_year=False, date_as_unix=True
    ):
        self.weekday = weekday
        self.day_of_month = day_of_month
        self.day_of_year = day_of_year
        self.date_as_unix = date_as_unix

    def __call__(self, df):
        df = self.get_categorical_encodings(df)
        df = self.get_dates_encodings(df)
        optimizer = PolarsDataTypeOptimizer()
        df = optimizer.type_optimization(df)
        return df

    def get_categorical_encodings(self, df):
        for col in df.columns:
            if df[col].dtype == pl.Categorical:
                encoding = df[col].to_dummies()
                df = df.drop(col)
                df = df.hstack(encoding)
        return df

    def get_dates_encodings(self, df):
        for col in df.columns:
            if df[col].dtype == pl.Date:
                if self.weekday:
                    df = df.with_columns(
                        pl.col(col).dt.weekday().alias(f"{col}_weekday")
                    )
                if self.day_of_month:
                    df = df.with_columns(
                        pl.col(col).dt.day().alias(f"{col}_day_of_month")
                    )
                if self.day_of_year:
                    df = df.with_columns(
                        pl.col(col).dt.ordinal_day().alias(f"{col}_day_of_year")
                    )
                if self.date_as_unix:
                    df = df.with_columns(
                        pl.col(col).dt.epoch(time_unit="d").alias(f"{col}_unix")
                    )
                df = df.drop(col)
        return df
