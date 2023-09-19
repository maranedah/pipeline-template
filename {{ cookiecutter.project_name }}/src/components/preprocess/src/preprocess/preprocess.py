import pandas as pd


def run_preprocess(gcs_bucket: str) -> pd.DataFrame:
    df = pd.read_parquet(f"gs://{gcs_bucket}/example.gzip")
    return df


if __name__ == "__main__":
    gcs_bucket = "ml-projects-dev-bucket"
    run_preprocess(gcs_bucket)
