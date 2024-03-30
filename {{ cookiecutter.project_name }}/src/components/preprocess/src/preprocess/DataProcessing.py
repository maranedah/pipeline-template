import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

class DataProcessing:
    df: pd.DataFrame
    replace_data: list[dict[str, any]]
    rename: dict[str, str]

    def __init__(
        self,
        df: pd.DataFrame,
        remove_rows_where=[],
        replace_data=[],
        rename={},
        encode=[],
    ):
        self.df = df
        if remove_rows_where:
            for setting in remove_rows_where:
                self.remove_rows_where(**setting)
        if replace_data:
            for setting in replace_data:
                self.replace(**setting)
        if encode:
            self.encoders = [self.encode_column(**setting) for setting in encode]
        if rename:
            self.df = self.df.rename(columns=rename)
            
        self.scalers = self.scale()
        self.to_half_precision()

    def remove_rows_where(self, columns: list[str], condition: callable):
        for column in columns:
            mask = self.df[column].apply(condition)
            self.df = self.df[~mask].reset_index(drop=True)

    def to_half_precision(self):
        float64_columns = self.df.select_dtypes(include=["float64", "float32"]).columns
        for column in float64_columns:
            self.df[column].astype("float16")

    def replace(self, columns: list[str], condition: callable, new_value: any):
        self.df[columns] = self.df[columns].map(
            lambda x: new_value if condition(x) else x
        )
        return self.df

    def encode_column(self, column, encoder):
        if isinstance(encoder, LabelEncoder):
            self.df[f"encoded_{column}"] = encoder.fit_transform(self.df[column])
        elif isinstance(encoder, OneHotEncoder):
            print(column, self.df[[column]].nunique())
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(
                        encoder.fit_transform(self.df[[column]]).astype(bool),
                        columns=encoder.get_feature_names_out([column]),
                    ),
                ],
                axis=1,
            )
        self.df = self.df.drop(columns=[column])
        return {"column": column, "encoder": encoder}

    def scale(self):
        scalers = []
        float64_columns = X_train.select_dtypes(include=["float64", "float32"]).columns.tolist()
        for column in float64_columns:
            scaler = StandardScaler()
            self.df[float64_columns] = scaler.fit_transform(df[float64_columns])
            scalers.append(scaler)
        return scalers