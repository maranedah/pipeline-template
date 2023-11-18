import pandas as pd
from palmerpenguins import load_penguins

from .DataSource import DataSource
from .schemas import PalmerPenguinsSchema


def run_ingestor(project_id: str, gcs_bucket: str) -> list[pd.DataFrame]:
    my_data = DataSource(
        name="palmer_penguins",
        read_function=load_penguins,
        read_params={},
        schema=PalmerPenguinsSchema,
    )

    data_sources = [my_data]

    for data_source in data_sources:
        data_source.get_data()
        data_source.validate_schema()
        data_source.save_data(gcs_bucket)

    return my_data.df


if __name__ == "__main__":
    run_ingestor(
        project_id="ml-projects-399119", gcs_bucket="gs://pipeline-template-dev"
    )
