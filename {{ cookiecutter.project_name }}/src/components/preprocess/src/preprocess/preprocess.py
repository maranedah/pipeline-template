import pandas as pd
from DataProcessing import DataProcessing
from sklearn.preprocessing import OneHotEncoder


def run_preprocess(project_id: str, palmer_penguins_uri: str) -> list[pd.DataFrame]:
    palmer_penguins = pd.read_parquet(palmer_penguins_uri)
    my_data = DataProcessing(
        df=palmer_penguins,
        encode=[
            {"column": column, "encoder": OneHotEncoder(sparse_output=False)}
            for column in palmer_penguins.select_dtypes(
                include=["category", "object"]
            ).columns
        ],
        rename={"encoded_species": "y"},
    )
    return my_data.df, my_data.encoders


if __name__ == "__main__":
    run_preprocess(None, None)
