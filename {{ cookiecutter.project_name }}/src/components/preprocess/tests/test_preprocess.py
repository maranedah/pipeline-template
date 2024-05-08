import unittest
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
from preprocess.preprocess import (
    get_output_paths,
    get_paths,
    merge_files,
    process_files,
)


class TestEncodings(unittest.TestCase):
    def setUp(self):
        self.case_ids = [39, 1338, 1531, 1526173, 2592396, 925488, 925437]

    def mock_read_parquet(original):
        def wrapped_method(data):
            case_ids = [39, 1338, 1531, 1526173, 2592396, 925488, 925437]
            df = original(data)
            return df.filter(df["case_id"].is_in(case_ids))

        return wrapped_method

    @patch("polars.read_parquet", mock_read_parquet(pl.read_parquet))
    def test_process_files(self):
        train_paths = get_paths("train")
        output_train_paths = get_output_paths("dummy", train_paths)
        process_files(train_paths, output_train_paths)
        df = merge_files(output_train_paths)
        df.write_parquet("dummy.parquet")
        self.assertEqual(df.shape[1], 954)

    def test_process_train_files(self):
        train_paths = get_paths("train")
        output_train_paths = get_output_paths("train", train_paths)
        process_files(train_paths, output_train_paths)
        train_df = merge_files(output_train_paths)

        train_df = train_df.filter(train_df["case_id"].is_in(self.case_ids))
        dummy_df = pl.read_parquet("dummy.parquet")
        shared_columns = list(set(dummy_df.columns) & set(train_df.columns))
        shared_columns.remove("mean_currdebt_94A")
        shared_columns.remove("mean_education_1138M_a55475b1")
        shared_columns.remove("disbursementtype_67L_GBA")
        assert_frame_equal(
            train_df[shared_columns], dummy_df[shared_columns], check_dtype=False
        )
        self.assertTrue(set(dummy_df.columns).issubset(set(train_df.columns)))
