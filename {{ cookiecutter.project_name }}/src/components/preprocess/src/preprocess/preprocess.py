import logging
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from constants import (
    ignore_columns,
    scaler,
    step_1_processing
)
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Format for displaying the date and time
)


def get_paths(set_):
    dir_ = {
        "train": Path(__file__).parent / "parquet_files" / "train",
        "test": Path(__file__).parent / "parquet_files" / "test",
    }
    paths = [
        f"{set_}_base.parquet",
        f"{set_}_static_cb_0.parquet",
        f"{set_}_static_0_*.parquet",
        f"{set_}_applprev_1_*.parquet",
        f"{set_}_tax_registry_a_1.parquet",
        f"{set_}_tax_registry_b_1.parquet",
        f"{set_}_tax_registry_c_1.parquet",
        f"{set_}_credit_bureau_a_1_*.parquet",
        f"{set_}_credit_bureau_b_1.parquet",
        f"{set_}_other_1.parquet",
        f"{set_}_person_1.parquet",
        f"{set_}_deposit_1.parquet",
        f"{set_}_debitcard_1.parquet",
        f"{set_}_credit_bureau_b_2.parquet",
        f"{set_}_credit_bureau_a_2_*.parquet",
        f"{set_}_applprev_2.parquet",
        f"{set_}_person_2.parquet",
    ]
    return [dir_[set_] / path for path in paths]


def read_parquet(filepath):
    logging.info(f"Reading data at path {filepath}")
    if "*" not in str(filepath):
        df = process_data(pl.read_parquet(filepath))
    else:
        paths = glob(str(filepath))
        df = process_data(pl.read_parquet(paths[0]))
        for path in paths[1:]:
            logging.info(f"Reading {path}...")
            new_df = process_data(pl.read_parquet(path))
            df = pl.concat((df, new_df), how="diagonal_relaxed")
    logging.info(
        f"""
        Finished reading {filepath.stem}
        with df of size {int(df.estimated_size() / 1024 ** 2)}MB
        """
    )
    return df


def process_data(df):
    return step_1_processing(df)


def process_submission_data(df):
    train_columns = pd.read_csv("columns.txt", header=None)[0].to_list()
    train_columns = set(train_columns)
    submission_columns = set(df.columns)
    shared_columns = train_columns & submission_columns
    not_in_submission_columns = train_columns - submission_columns
    df = df[list(shared_columns)]
    # df = df.fill_null(0)
    for col in list(not_in_submission_columns):
        default_value = 1.0 if "null" in col else None
        df = df.with_columns(pl.lit(default_value).alias(col))
    return df


def run_preprocess(project_id: str, palmer_penguins_uri: str) -> list[pd.DataFrame]:
    for split in ["train", "test"]:
        for path in get_paths(split):
            output_file = f"data/{split}/{''.join(path.stem.split('*'))}.parquet"
            if not os.path.exists(output_file):
                df = read_parquet(path)
                df.write_parquet(output_file)

    # Consolidate multiple files into a single one

    for split in ["train", "test"]:
        if not os.path.exists(f"output/{split}/consolidated_dataset.parquet"):
            df = pl.read_parquet(f"data/{split}/{split}_base.parquet")
            for i, file in enumerate(
                [
                    f
                    for f in os.listdir(f"data/{split}")
                    if "base" not in f and "dataset" not in f
                ]
            ):
                new_df = pl.read_parquet(f"data/{split}/{file}")
                if len(new_df) == 0:
                    continue
                df.with_columns(pl.col("case_id").cast(pl.UInt64))
                df = df.join(new_df, how="left", on="case_id", suffix=f"_{i}")

            if split == "train":
                df = df.drop(["MONTH"])
                columns = pd.read_csv("columns.txt", header=None)[0].to_list()
                df = df[columns]
                # df = df.fill_null(0)
                df = scaler.fit_transform(df)
                #df = type_optimization(df.to_pandas())
                print("df shape", df.shape)
            elif split == "test":
                df = process_submission_data(df)
                df = scaler.transform(df)
                #df = type_optimization(df.to_pandas())
                np.save("processed/X_submission_case_ids.npy", df["case_id"].values)
                columns = [col for col in df.columns if col not in ignore_columns]
                df = df[columns]
                np.save("processed/X_submission.npy", df.values)
                print(df.shape)
            df.to_parquet(f"output/{split}/consolidated_dataset.parquet")

    # Train, valid, test splits
    split = "train"
    df = pl.read_parquet("output/train/consolidated_dataset.parquet")
    case_ids = df["case_id"].unique().shuffle(seed=1).to_frame()
    case_ids_train, case_ids_test = train_test_split(
        case_ids, train_size=0.6, random_state=1
    )
    case_ids_valid, case_ids_test = train_test_split(
        case_ids_test, train_size=0.5, random_state=1
    )

    def from_polars_to_pandas(case_ids: pl.DataFrame) -> pl.DataFrame:
        columns = [col for col in df.columns if col not in ignore_columns]
        return (
            df.filter(pl.col("case_id").is_in(case_ids))[
                ["case_id", "WEEK_NUM", "target"]
            ].to_pandas(),
            df.filter(pl.col("case_id").is_in(case_ids))[columns]
            .to_pandas()
            .values.astype(np.float16),
            df.filter(pl.col("case_id").is_in(case_ids))["target"]
            .to_pandas()
            .values.astype(np.uint8),
        )

    base_train, X_train, y_train = from_polars_to_pandas(case_ids_train)
    base_valid, X_valid, y_valid = from_polars_to_pandas(case_ids_valid)
    base_test, X_test, y_test = from_polars_to_pandas(case_ids_test)

    base_train.to_parquet("processed/base_train.parquet")
    base_valid.to_parquet("processed/base_valid.parquet")
    base_test.to_parquet("processed/base_test.parquet")

    np.save("processed/X_train.npy", X_train)
    np.save("processed/X_valid.npy", X_valid)
    np.save("processed/X_test.npy", X_test)

    np.save("processed/y_train.npy", y_train)
    np.save("processed/y_valid.npy", y_valid)
    np.save("processed/y_test.npy", y_test)


if __name__ == "__main__":
    run_preprocess(None, None)
