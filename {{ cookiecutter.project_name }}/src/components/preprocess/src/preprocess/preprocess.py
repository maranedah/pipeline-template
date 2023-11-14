import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class DataProcessing:
    df: pd.DataFrame
    replace_data: list[dict[str, any]]
    rename: dict[str, str]

    def __init__(self, df, replace_data, rename, encode):
        self.df = df
        for setting in replace_data:
            self.replace(**setting)
        self.encoders = [self.encode_column(**setting) for setting in encode]
        self.df = self.df.rename(columns=rename)

    def replace(self, columns: list[str], condition: callable, new_value: any):
        self.df[columns] = self.df[columns].map(
            lambda x: new_value if condition(x) else x
        )
        return self.df

    def encode_column(self, column, encoder):
        if isinstance(encoder, LabelEncoder):
            self.df[f"encoded_{column}"] = encoder.fit_transform(self.df[column])
        elif isinstance(encoder, OneHotEncoder):
            self.df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(
                        encoder.fit_transform(self.df[[column]]),
                        columns=encoder.get_feature_names_out([column]),
                    ),
                ],
                axis=1,
            )
        self.df.drop(columns=[column])
        return {"column": column, "encoder": encoder}


def run_preprocess(project_id: str, gcs_bucket: str) -> list[pd.DataFrame]:
    data = {
        "A": [1, 2, 3, 4],
        "B": ["apple", "banana", "cherry", "date"],
        "C": [0.1, 0.5, 0.8, 1.0],
    }

    df = pd.DataFrame(data)

    my_data = DataProcessing(
        df=df,
        replace_data=[
            {
                "columns": ["C"],
                "condition": lambda x: (x < 0.6) or (x is None),
                "new_value": 0,
            },
        ],
        rename={"C": "y", "A": "ts"},
        encode=[{"column": "B", "encoder": OneHotEncoder(sparse_output=False)}],
    )
    return my_data.df, my_data.encoders


if __name__ == "__main__":
    run_preprocess(None, None)
