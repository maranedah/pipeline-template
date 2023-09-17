import numpy as np
import pandas as pd

from .schemas import MyDataFrameSchema


def run_ingestor(gcs_bucket: str) -> list[pd.DataFrame]:
    data = {
        "A": np.random.randint(1, 100, 10),  # 10 random integers between 1 and 100
        "B": np.random.rand(10),  # 10 random float numbers between 0 and 1
        "C": np.random.choice(["X", "Y", "Z"], 10),  # 10 random choices from a list
    }

    df = pd.DataFrame(data)
    MyDataFrameSchema.validate(df)
    return df


if __name__ == "__main__":
    run_ingestor("asd")