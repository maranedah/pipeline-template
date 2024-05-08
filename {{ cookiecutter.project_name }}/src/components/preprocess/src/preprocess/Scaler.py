import polars as pl
from sklearn.preprocessing import StandardScaler


class Scaler:
    def __init__(self, ignore_columns=[]):
        self.ignore_columns = ignore_columns
        self.scalers = None
        self.fitted = False

    def __call__(self, df):
        if self.fitted:
            df = self.transform(df)
        else:
            df = self.fit_transform(df)
        return df

    def fit_transform(self, df):
        def scaler_fit_transform(s: pl.Series, scaler) -> pl.Series:
            return pl.Series(
                scaler.fit_transform(s.to_numpy().reshape(-1, 1)).flatten()
            )

        columns = [col for col in df.columns if col not in self.ignore_columns]
        self.scalers = []
        for col in columns:
            scaler = StandardScaler()
            df = df.with_columns(
                pl.col(col).map_batches(lambda x: scaler_fit_transform(x, scaler))
            )
            self.scalers.append({col: scaler})
        self.fitted = True
        return df

    def transform(self, df):
        def apply_scaler(s: pl.Series, scaler) -> pl.Series:
            return pl.Series(scaler.transform(s.to_numpy().reshape(-1, 1)).flatten())

        for scaler_dict in self.scalers:
            for col, scaler in scaler_dict.items():
                if col in df.columns:
                    df = df.with_columns(
                        pl.col(col).map_batches(lambda x: apply_scaler(x, scaler))
                    )
        return df
