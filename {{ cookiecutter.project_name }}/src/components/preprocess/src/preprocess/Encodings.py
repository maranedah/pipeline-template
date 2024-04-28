import polars as pl


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

        return df

    def get_columns_where(self, df, col_type):
        column_types = zip(df.columns, df.dtypes)
        return [col for col, dtype in column_types if dtype == col_type]

    def get_categorical_encodings(self, df):
        categorical_columns = self.get_columns_where(df, col_type=pl.Categorical)
        if categorical_columns:
            for categorical_column in categorical_columns:
                print(categorical_column)
                encoding = df[categorical_column].to_dummies()
                df = df.drop(categorical_column)
                df = df.hstack(encoding)
        return df

    def get_dates_encodings(self, df):
        date_columns = self.get_columns_where(df, col_type=pl.Date)
        for col in date_columns:
            if self.weekday:
                df = df.with_columns(pl.col(col).dt.weekday().alias(f"{col}_weekday"))
            if self.day_of_month:
                df = df.with_columns(
                    pl.col(col).dt.weekday().alias(f"{col}_day_of_month")
                )
            if self.day_of_year:
                df = df.with_columns(
                    pl.col(col).dt.weekday().alias(f"{col}_day_of_year")
                )
            if self.date_as_unix:
                df = df.with_columns(
                    pl.col(col).dt.epoch(time_unit="d").alias(f"{col}_unix")
                )
            df = df.drop(col)
        return df
