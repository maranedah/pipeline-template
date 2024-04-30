import unittest

import polars as pl
from preprocess.DataTypeOptimizer import PolarsDataTypeOptimizer


class TestPandasDataTypeOptimizer(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.df = pl.DataFrame(
            {
                "A": [1.0, 2.0, 3.0],
                "B": [1.0, 2.5, 3.8],
                "C": ["2021-01-01", "2022-02-02", "2023-03-03"],
                "D": ["apple", "banana", "orange"],
                "E": [None, None, None],
            }
        )
        self.optimizer = PolarsDataTypeOptimizer(ignore_columns=[])

    def test_optimize_float_to_uint(self):
        df_float_to_int = self.df.clone()
        optimized_column = self.optimizer.optimize_numerical_dtype(df_float_to_int, "A")
        self.assertEqual(optimized_column["A"].dtype, pl.UInt8)

    def test_optimize_float_to_float(self):
        df_float_to_float = self.df.clone()
        optimized_column = self.optimizer.optimize_numerical_dtype(
            df_float_to_float, "B"
        )
        self.assertEqual(optimized_column["B"].dtype, pl.Float32)

    def test_null_to_float(self):
        df_null_to_float = self.df.clone()
        optimized_column = self.optimizer.optimize_numerical_dtype(
            df_null_to_float, "E"
        )
        self.assertEqual(optimized_column["E"].dtype, pl.Float32)

    def test_set_dates(self):
        df_with_dates = self.df.clone()
        optimized_df = self.optimizer.set_dates(df_with_dates)
        self.assertEqual(optimized_df["C"].dtype, pl.Date)

    def test_set_categorical(self):
        optimized_df = self.optimizer.set_categorical(self.df)
        self.assertEqual(optimized_df["D"].dtype, pl.Categorical)

    def test_set_numerical(self):
        optimized_df = self.optimizer.set_numerical(self.df)
        self.assertEqual(optimized_df["A"].dtype, pl.UInt8)
        self.assertEqual(optimized_df["B"].dtype, pl.Float32)

    def test_type_optimization(self):
        optimized_df = self.optimizer(self.df)
        self.assertEqual(optimized_df["A"].dtype, pl.UInt8)
        self.assertEqual(optimized_df["B"].dtype, pl.Float32)
        self.assertEqual(optimized_df["C"].dtype, pl.Date)
        self.assertEqual(optimized_df["D"].dtype, pl.Categorical)
