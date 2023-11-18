import pandas as pd

from .DataSource import DataSource
from .pandera_sampler import get_schema_sample
from .schemas import MyDataFrameSchema


def run_ingestor(project_id: str, gcs_bucket: str) -> list[pd.DataFrame]:
    my_data = DataSource(
        name="dummy",
        read_function=get_schema_sample,
        read_params={"schema": MyDataFrameSchema, "size": 100},
        schema=MyDataFrameSchema,
    )

    data_sources = [my_data]

    for data_source in data_sources:
        data_source.get_data()
        data_source.validate_schema()
        data_source.save_data(gcs_bucket)

    return my_data.df

if __name__ == "__main__":
    run_ingestor(None, None)
