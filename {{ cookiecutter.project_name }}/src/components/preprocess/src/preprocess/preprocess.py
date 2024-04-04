import gc
from glob import glob
from pathlib import Path

import pandas as pd
import polars as pl
from DataProcessing import DataProcessing
from MemoryReduction import MemoryReduction
from sklearn.preprocessing import OneHotEncoder


def set_table_dtypes(df):
    for col in df.columns:
        if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
            df = df.with_columns(pl.col(col).cast(pl.Int64))
        elif col in ["date_decision"]:
            df = df.with_columns(pl.col(col).cast(pl.Date))
        elif col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64))
        elif col[-1] in ("M",):
            df = df.with_columns(pl.col(col).cast(pl.String))
        elif col[-1] in ("D",):
            df = df.with_columns(pl.col(col).cast(pl.Date))
    return df


def handle_dates(df):
    for col in df.columns:
        if col[-1] in ("D",):
            df = df.with_columns(pl.col(col) - pl.col("date_decision"))  #!!?
            df = df.with_columns(pl.col(col).dt.total_days())  # t - t-1
    df = df.drop("date_decision", "MONTH")
    return df


class Aggregator:
    def pandas_get_aggregations(df):
        numeric_columns = df.select_dtypes(include="number").columns
        numeric_funcs = ["mean", "max", "min", "first", "last"]
        non_numeric_funcs = ["max", "min", "first", "last"]
        columns = [
            col
            for col in df.columns
            if col not in ["case_id", "target", "date_decision", "MONTH", "WEEK_NUM"]
        ]
        for col in columns:
            funcs = numeric_funcs if col in numeric_columns else non_numeric_funcs
            df = MemoryReduction().type_optimization(df)
            df_aggregations = (
                df[["case_id", col]].groupby("case_id").agg(funcs).reset_index()
            )
            df_aggregations.columns = [
                f"{col}_{func}" if col != "case_id" else col
                for col, func in df_aggregations.columns
            ]
            df_aggregations = MemoryReduction().type_optimization(df_aggregations)
            df = df.merge(df_aggregations, how="left", on="case_id")
        return df

    def polars_get_aggregations(df):
        expr_mean = [
            pl.mean(col).alias(f"mean_{col}")
            for col in df.columns
            if "Int" in col or "Float" in col
        ]
        expr_min = [pl.min(col).alias(f"min_{col}") for col in df.columns]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in df.columns]
        expr_first = [pl.first(col).alias(f"first_{col}") for col in df.columns]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in df.columns]
        return expr_mean + expr_min + expr_max + expr_first + expr_last


def get_paths(set_):
    dir_ = {
        "train": Path(__file__).parent / "parquet_files" / "train",
        "test": Path(__file__).parent / "parquet_files" / "train",
    }
    df_base = f"{set_}_base.parquet"
    depth_0 = [
        f"{set_}_static_cb_0.parquet",
        f"{set_}_static_0_*.parquet",
    ]
    depth_1 = [
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
    ]
    depth_2 = [
        f"{set_}_credit_bureau_b_2.parquet",
        f"{set_}_credit_bureau_a_2_*.parquet",
        f"{set_}_applprev_2.parquet",
        f"{set_}_person_2.parquet",
    ]
    return {
        "df_base": dir_[set_] / df_base,
        "depth_0": [dir_[set_] / file for file in depth_0],
        "depth_1": [dir_[set_] / file for file in depth_1],
        "depth_2": [dir_[set_] / file for file in depth_2],
    }


def read_file(path, depth=None):
    print(path)
    df = pl.read_parquet(path)
    df = set_table_dtypes(df)
    df = MemoryReduction().type_optimization(df)
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.polars_get_aggregations(df))
        df = MemoryReduction().type_optimization(df)
    return df


def read_files(regex_path, depth=None):
    chunks = []

    for path in glob(str(regex_path)):
        df = read_file(path, depth)
        chunks.append(df)

    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    return df


def read_data(path, depth):
    if "*" in str(path):
        df = read_files(path, depth)
    else:
        df = read_file(path, depth)
    return df


def merge_files(df_base, depth_0, depth_1, depth_2):
    df_base = read_data(df_base, depth=0)
    df_base = df_base.with_columns(
        month_decision=pl.col("date_decision").dt.month(),
        weekday_decision=pl.col("date_decision").dt.weekday(),
    )
    depth_df = df_base
    i = 0
    for depth, depth_path_list in zip([2, 1, 0], [depth_2, depth_1, depth_0]):
        for path in depth_path_list:
            df = read_data(path, depth=depth)
            print(depth_df.shape)
            depth_df = depth_df.join(df, how="left", on="case_id", suffix=f"_{i}")
            depth_df = MemoryReduction().filter_cols(depth_df)
            del df
            gc.collect()
            i += 1
    depth_df = handle_dates(depth_df)
    return depth_df


def run_preprocess(project_id: str, palmer_penguins_uri: str) -> list[pd.DataFrame]:
    # palmer_penguins = pd.read_parquet(palmer_penguins_uri)

    # df = merge_files(**get_paths("train"))
    # df = df.to_pandas()
    # df = MemoryReduction().type_optimization(df)
    df = pd.read_parquet("processed_df_train.parquet")
    df = df.drop(columns=["date_decision", "MONTH"])
    my_data = DataProcessing(
        df=df,
        remove_columns_where=[
            {
                "condition": lambda df, col: df[col].isna().mean() > 0.8,
                "ignore_columns": ["target", "case_id", "WEEK_NUM"],
            },
            {
                "condition": lambda df, col: df[col].nunique() == 1,
                "ignore_columns": ["target", "case_id", "WEEK_NUM"],
            },
            # {
            #    "condition": lambda df, col: df[col].nunique() > 200,
            #    "ignore_columns": df.select_dtypes(include='number').columns.tolist()
            # }
        ],
        # replace_data=[
        #    {
        #        "columns": [],
        #        "condition": lambda df, col: col[-1] in ("D",),
        #        "new_value": pd.to_datetime()
        #    }
        # ],
        encode=[
            {"column": column, "encoder": OneHotEncoder(sparse_output=False)}
            for column in df.select_dtypes(include=["category", "object"]).columns
        ],
        # rename={"encoded_species": "y"},
    )
    return my_data.df, my_data.encoders


if __name__ == "__main__":
    run_preprocess(None, None)
