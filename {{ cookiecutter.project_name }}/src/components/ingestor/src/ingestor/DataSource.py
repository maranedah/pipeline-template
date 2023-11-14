import pandas as pd
import pandera as pa


class DataSource:
    name: str
    read_function: callable
    read_params: dict[str, any]
    schema: pa.DataFrameModel
    columns: list[str]
    cast_columns: dict[str, callable]
    df: pd.DataFrame

    def __init__(
        self,
        name: str,
        read_function: callable,
        read_params: dict[str, any],
        schema: pa.DataFrameModel,
        columns: list[str] = None,
        cast_columns: dict[str, callable] = None,
    ):
        self.name = name
        self.read_function = read_function
        self.read_params = read_params
        self.schema = schema
        self.columns = columns
        self.cast_columns = cast_columns
        self.df = pd.DataFrame()

    def get_data(self) -> pd.DataFrame:
        self.df = self.read_function(**self.read_params)
        if self.columns:
            self.df = self.df[self.columns]
        if self.cast_columns:
            for column_name, cast_function in self.cast_columns.items():
                self.df[column_name] = cast_function(self.df[column_name])
        return self.df

    def validate_schema(self) -> None:
        self.schema.validate(self.df)

    def save_data(self, gcs_bucket_path: str) -> None:
        if gcs_bucket_path:
            self.df.to_parquet(
                path=f"{gcs_bucket_path}/{self.name}.gzip", compression="gzip"
            )
