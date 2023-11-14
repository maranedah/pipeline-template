import pandas as pd

from .DataSource import DataSource
from .schemas import MyDataFrameSchema


def run_ingestor(project_id: str, gcs_bucket: str) -> list[pd.DataFrame]:
    my_data = DataSource(
        name="dummy",
        read_function=MyDataFrameSchema.example,
        read_params={"size": 100},
        schema=MyDataFrameSchema,
    )

    data_sources = [my_data]

    for data_source in data_sources:
        data_source.get_data()
        data_source.validate_schema()
        data_source.save_data(gcs_bucket)

    return my_data.df
