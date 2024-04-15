import os
from glob import glob
from pathlib import Path

import pandas as pd
import polars as pl
from MemoryReduction import MemoryReduction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def handle_dates(df):
    for col in df.columns:
        if col[-1] in ("D",):
            df = df.with_columns(pl.col(col) - pl.col("date_decision"))  #!!?
            df = df.with_columns(pl.col(col).dt.total_days())  # t - t-1
    df = df.drop("date_decision", "MONTH")
    return df


class Aggregator:
    def get_exprs(df, ignore_columns=[]):
        columns = [col for col in df.columns if col not in ignore_columns]
        numerical_columns = [
            col
            for col, dtype in zip(df[columns].columns, df[columns].dtypes)
            if "Float" in str(dtype) or "Int" in str(dtype)
        ]
        date_columns = [
            col
            for col, dtype in zip(df[columns].columns, df[columns].dtypes)
            if "Date" in str(dtype)
        ]

        # Length of each group
        expr_len = [pl.len()]

        # Numerical columns
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in numerical_columns]
        expr_std = [pl.std(col).alias(f"std_{col}") for col in numerical_columns]

        # Date columns
        expr_min = [pl.min(col).alias(f"min_{col}") for col in date_columns]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in date_columns]
        return expr_len + expr_mean + expr_std + expr_min + expr_max


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
    if "*" not in str(filepath):
        df = process_data(pl.read_parquet(filepath), filter_cols)
    else:
        paths = glob(str(filepath))
        df = process_data(pl.read_parquet(paths[0]), filter_cols)
        print(df.shape)
        for path in paths[1:]:
            new_df = process_data(pl.read_parquet(path), filter_cols)
            print(new_df.shape)
            df = pl.concat((df, new_df), how="diagonal_relaxed")
    return df


def set_dates(df):
    pattern = r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}"
    for col in df.columns:
        if df[col].dtype == pl.String and df[col].str.contains(pattern).all():
            df = df.with_columns(pl.col(col).cast(pl.Date))
    return df


def set_categorical(df):
    str_columns = [
        col for col, dtype in zip(df.columns, df.dtypes) if "String" in str(dtype)
    ]
    for col in str_columns:
        df = df.with_columns(pl.col(col).cast(pl.Categorical))
    return df


def too_many_nulls(df, col, threshold=0.8):
    return df[col].is_null().mean() > threshold


def only_one_value(df, col):
    return df[col].n_unique() == 1


def too_many_categories(df, col, threshold=10):
    return df[col].dtype == pl.Categorical and df[col].n_unique() > threshold


def filter_cols(df):
    columns_to_filter = []
    for col in df.columns:
        should_be_filtered = (
            too_many_nulls(df, col)
            or only_one_value(df, col)
            or too_many_categories(df, col)
        )
        if should_be_filtered:
            columns_to_filter.append(col)
    for col in columns_to_filter:
        df = df.drop(col)
    print(len(columns_to_filter))
    return df


def get_encodings(df):
    categorical_columns = [
        col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Categorical
    ]
    if categorical_columns:
        encoding = df[categorical_columns].to_dummies()
        df = df.drop(categorical_columns)
        df = df.hstack(encoding)
    return df


# Function to group columns by correlation
def group_columns_by_correlation(df, threshold=0.9):
    print("shape1", df.shape)
    correlation_matrix = df.corr()
    groups = []
    remaining_cols = list(df.columns)
    while remaining_cols:
        col = remaining_cols.pop(0)
        group = [col]
        correlated_cols = [col]
        for c in remaining_cols:
            if correlation_matrix.loc[col, c] >= threshold:
                group.append(c)
                correlated_cols.append(c)
        groups.append(group)
        remaining_cols = [c for c in remaining_cols if c not in correlated_cols]
    # Filter groups with length = 1
    groups = [group for group in groups if len(group) > 1]

    for group in groups:
        n_nulls = {col: df[col].isna().sum() for col in group}
        ordered_nulls_cols = list(
            dict(sorted(n_nulls.items(), key=lambda item: item[1])).keys()
        )
        df = df.drop(columns=ordered_nulls_cols[1:])
    print("shape2", df.shape)
    return list(df.columns)


def aggregate_rows(df):
    groups = df.group_by("case_id")
    # If n_groups == n_data, skip, nothing to aggregate
    if df.shape[0] == groups.len().shape[0]:
        return df
    else:
        # Aggregate n_count, mean, min, max, etc
        df = df.group_by("case_id").agg(
            Aggregator.get_exprs(
                df, ignore_columns=["case_id", "num_group1", "num_group2"]
            )
        )
        return df


def process_data(df, should_filter_cols=True):
    if should_filter_cols:
        return MemoryReduction().type_optimization(
            aggregate_rows(get_encodings(filter_cols(set_categorical(set_dates(df)))))
        )
    else:
        return MemoryReduction().type_optimization(
            aggregate_rows(get_encodings(set_categorical(set_dates(df))))
        )


def date_to_number(df):
    date_columns = [
        col
        for col, dtype in zip(df.columns, df.dtypes)
        if "Date" in str(dtype) and "date_decision" not in col
    ]
    for col in date_columns:
        df = df.with_columns(pl.col(col) - pl.col("date_decision"))
        df = df.with_columns(pl.col(col).dt.total_days())
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
    # Run processing for each individual file
    split = "train"
    for split in ["train", "test"]:
        for path in get_paths(split):
            print(path)
            output_file = f"data/{split}/{''.join(path.stem.split('*'))}.parquet"
            if not os.path.exists(output_file):
                df = read_parquet(path, filter_cols=True)
                if "base" in str(path):
                    df.write_parquet(output_file)
                    continue
                if split == "train":
                    columns = group_columns_by_correlation(df.to_pandas())
                    df = df[columns]
                df.write_parquet(output_file)

    # Consolidate multiple files into a single one
    split = "train"
    for split in ["train", "test"]:
        if not os.path.exists(f"data/{split}/consolidated_dataset.parquet"):
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
                df = filter_cols(df)
            df = date_to_number(df)
            df = MemoryReduction().type_optimization(df)
            df = df.fill_null(0)
            if split == "train":
                df, scalers = scale_data(df)
                import pickle

                pickle.dump(scalers, open("scalers.pickle", "wb"))
                df = MemoryReduction().type_optimization(df)
                columns = group_columns_by_correlation(df.to_pandas())
                df = MemoryReduction().type_optimization(df[columns].to_pandas())
            elif split == "test":
                train_df = read_parquet("data/train/consolidated_dataset.parquet")
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

                df = df.drop("case_id")
                df, scalers = scale_data(df, scalers)
                df = MemoryReduction().type_optimization(df.to_pandas())
                import numpy as np

                np.save("processed/X_submission.npy", df.values.astype(np.float16))

            df.to_parquet(f"data/{split}/consolidated_dataset.parquet")

    # Train, valid, test splits
    split = "train"
    df = pl.read_parquet("data/train/consolidated_dataset.parquet")
    df = filter_cols(df)
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
