
import pandas as pd
import numpy as np

def run_ingestor(gcs_bucket: str) -> list[pd.DataFrame]:
    data = {
        'A': np.random.randint(1, 100, 10),  # 10 random integers between 1 and 100
        'B': np.random.rand(10),            # 10 random float numbers between 0 and 1
        'C': np.random.choice(['X', 'Y', 'Z'], 10)  # 10 random choices from a list
    }

    df = pd.DataFrame(data)
    return (
        df
    )