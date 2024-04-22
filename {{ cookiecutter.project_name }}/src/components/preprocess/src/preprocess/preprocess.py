import logging
import os
from glob import glob
from pathlib import Path

import pandas as pd
import polars as pl
from constants import (
    aggregate_rows,
    filter_columns,
    filter_correlated_columns,
    get_encodings,
    type_handling,
    type_optimization,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


def read_parquet(filepath, filter_cols=True):
    logging.info(f"Reading data at path {filepath}")
    if "*" not in str(filepath):
        df = process_data(pl.read_parquet(filepath), should_filter_cols=filter_cols)
    else:
        paths = glob(str(filepath))
        df = process_data(pl.read_parquet(paths[0]), should_filter_cols=True)
        for path in paths[1:]:
            logging.info(f"Reading {path}...")
            new_df = process_data(pl.read_parquet(path), should_filter_cols=True)
            df = pl.concat((df, new_df), how="diagonal_relaxed")
    logging.info(
        f"""
        Finished reading {filepath}
        with df of size {int(df.estimated_size() / 1024 ** 2)}MB
        """
    )
    return df


def process_data(df, should_filter_cols=True):
    return filter_columns(
        aggregate_rows(get_encodings(filter_columns(type_handling(df))))
    )


def date_to_number(df):
    df = df.drop("date_decision", "date_decision_2", "MONTH")
    return df


def scale_data(df, scalers=None):
    def my_scaler(s: pl.Series, scaler) -> pl.Series:
        return pl.Series(scaler.fit_transform(s.to_numpy().reshape(-1, 1)).flatten())

    def apply_scaler(s: pl.Series, scaler) -> pl.Series:
        return pl.Series(scaler.transform(s.to_numpy().reshape(-1, 1)).flatten())

    columns = [
        col for col in df.columns if col not in ["case_id", "WEEK_NUM", "target"]
    ]
    if scalers:
        for scaler_dict in scalers:
            for col, scaler in scaler_dict.items():
                if col in df.columns:
                    df = df.with_columns(
                        pl.col(col).map_batches(lambda x: apply_scaler(x, scaler))
                    )
    else:
        scalers = []
        for col in columns:
            scaler = StandardScaler()
            df = df.with_columns(
                pl.col(col).map_batches(lambda x: my_scaler(x, scaler))
            )
            scalers.append({col: scaler})
    return df, scalers


def run_preprocess(project_id: str, palmer_penguins_uri: str) -> list[pd.DataFrame]:
    for split in ["train", "test"]:
        for path in get_paths(split):
            output_file = f"data/{split}/{''.join(path.stem.split('*'))}.parquet"
            if not os.path.exists(output_file):
                df = read_parquet(path, filter_cols=True)
                if split == "train":
                    df = filter_correlated_columns(df)
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
                df = filter_columns(df)
            df = date_to_number(df)
            # df = type_optimization(df)
            df = df.fill_null(0)
            if split == "train":
                df, scalers = scale_data(df)
                import pickle

                pickle.dump(scalers, open("scalers.pickle", "wb"))
                df = type_optimization(df)
                df = filter_correlated_columns(df)
            elif split == "test":
                train_df = read_parquet("output/train/consolidated_dataset.parquet")
                import pickle

                scalers = pickle.load(open("scalers.pickle", "rb"))
                columns_to_select = [
                    col for col in df.columns if col in train_df.columns
                ]
                df = df[columns_to_select]
                columns_to_fill = [
                    col
                    for col in train_df.columns
                    if col not in df.columns and col != "target" and col != "case_id"
                ]
                for col in columns_to_fill:
                    df = df.with_columns(pl.lit(0.0).alias(col))

                df, scalers = scale_data(df, scalers)
                import numpy as np

                np.save("processed/X_submission.npy", df.values.astype(np.float16))
            df.write_parquet(f"output/{split}/consolidated_dataset.parquet")

    # Train, valid, test splits
    split = "train"
    df = pl.read_parquet("output/train/consolidated_dataset.parquet")
    df = filter_columns(df)
    case_ids = df["case_id"].unique().shuffle(seed=1).to_frame()
    case_ids_train, case_ids_test = train_test_split(
        case_ids, train_size=0.6, random_state=1
    )
    case_ids_valid, case_ids_test = train_test_split(
        case_ids_test, train_size=0.5, random_state=1
    )

    def from_polars_to_pandas(case_ids: pl.DataFrame) -> pl.DataFrame:
        columns = [col for col in df.columns if col not in ["target", "case_id"]]
        return (
            df.filter(pl.col("case_id").is_in(case_ids))[
                ["case_id", "WEEK_NUM", "target"]
            ].to_pandas(),
            df.filter(pl.col("case_id").is_in(case_ids))[columns].to_pandas(),
            df.filter(pl.col("case_id").is_in(case_ids))["target"].to_pandas(),
        )

    base_train, X_train, y_train = from_polars_to_pandas(case_ids_train)
    base_valid, X_valid, y_valid = from_polars_to_pandas(case_ids_valid)
    base_test, X_test, y_test = from_polars_to_pandas(case_ids_test)

    base_train.to_parquet("processed/base_train.parquet")
    base_valid.to_parquet("processed/base_valid.parquet")
    base_test.to_parquet("processed/base_test.parquet")

    import numpy as np

    np.save("processed/X_train.npy", X_train.values.astype(np.float16))
    np.save("processed/X_valid.npy", X_valid.values.astype(np.float16))
    np.save("processed/X_test.npy", X_test.values.astype(np.float16))

    np.save("processed/y_train.npy", y_train.values.astype(np.float16))
    np.save("processed/y_valid.npy", y_valid.values.astype(np.float16))
    np.save("processed/y_test.npy", y_test.values.astype(np.float16))


if __name__ == "__main__":
    run_preprocess(None, None)
