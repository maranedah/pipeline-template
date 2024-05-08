import logging
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from .constants import (
    ignore_columns,
    step_1_processing,
)

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
    logging.info(f"Reading data at path {filepath.stem}")
    if "*" not in str(filepath):
        df = process_data(pl.read_parquet(filepath))
    else:
        paths = glob(str(filepath))
        df = process_data(pl.read_parquet(paths[0]))
        for path in paths[1:]:
            logging.info(f"Reading {Path(path).stem}...")
            new_df = process_data(pl.read_parquet(path))
            if len(new_df):
                df = pl.concat((df, new_df), how="diagonal_relaxed")
    # df  = step_2_processing(df)
    logging.info(
        f"""
        Finished reading {filepath.stem}
        with df of size {int(df.estimated_size() / 1024 ** 2)}MB
        """
    )
    return df


def process_data(df):
    return step_1_processing(df)


def process_submission_data(df, train_columns):
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


def merge_files(output_paths):
    base_file = next(f for f in output_paths if "base" in str(f))
    df = pl.read_parquet(base_file)
    files = [f for f in output_paths if f != base_file]
    for i, file in enumerate(files):
        new_df = pl.read_parquet(file)
        if len(new_df):
            df = df.join(new_df, how="left", on="case_id", suffix=f"_{i}")
    # df = step_2_processing(df)
    return df


def separate_features(df, id_col, group_col, target_col):
    columns = [col for col in df.columns if col not in ignore_columns]
    id_col = df[id_col]
    group_col = df[group_col]
    features = df[columns]
    target_col = df[target_col]
    return id_col, group_col, features, target_col


def scale(df):
    # mean_values = df.mean()
    # std_values = df.std()
    df = df.select((pl.all() - pl.all().mean()) / pl.all().std())
    return df


def process_files(input_paths, output_paths):
    for input_path, output_path in zip(input_paths, output_paths):
        print(input_path)
        if not os.path.exists(output_path):
            df = read_parquet(input_path)
            df.write_csv(output_path)


def get_output_paths(split, input_files):
    output_folder = Path(__file__).parent / "data" / split
    output_paths = [output_folder / f"{f.stem.split('*')[0]}.csv" for f in input_files]
    return output_paths


def run_preprocess(project_id: str, palmer_penguins_uri: str) -> list[pd.DataFrame]:
    # Process data to files
    train_paths = get_paths("train")
    output_train_paths = get_output_paths("train", train_paths)
    process_files(train_paths, output_train_paths)
    df_train = merge_files(output_train_paths)

    # test_files = get_paths("test")

    # Consolidate multiple files into a single one
    df_train = merge_files("train")
    df_test = merge_files("test")

    df_test = process_submission_data(df_test, train_columns=df_train.columns)

    case_ids, week_nums, np_test, test_target = separate_features(
        df_test, "case_id", "WEEK_NUM", "target"
    )

    # Train, valid, test splits
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
