import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .DataProcessing import DataProcessing


def run_preprocess(project_id: str, palmer_penguins_uri: str) -> list[pd.DataFrame]:
    palmer_penguins = pd.read_parquet(palmer_penguins_uri)
    my_data = DataProcessing(
        df=palmer_penguins,
        remove_rows_where=[
            {
                "columns": [
                    "bill_length_mm",
                    "bill_depth_mm",
                    "flipper_length_mm",
                    "body_mass_g",
                    "sex",
                ],
                "condition": pd.isna,
            }
        ],
        encode=[
            {"column": "island", "encoder": OneHotEncoder(sparse_output=False)},
            {"column": "sex", "encoder": OneHotEncoder(sparse_output=False)},
            {"column": "species", "encoder": LabelEncoder()},
        ],
        rename={"encoded_species": "y"},
    )
    return my_data.df, my_data.encoders


if __name__ == "__main__":
    run_preprocess(None, None)
