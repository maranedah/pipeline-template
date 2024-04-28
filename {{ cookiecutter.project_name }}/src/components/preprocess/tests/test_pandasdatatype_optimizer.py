import unittest

import numpy as np
import pandas as pd
from preprocess.DataTypeOptimizer import PandasDataTypeOptimizer


class TestPandasDataTypeOptimizer(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.df = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0],
                "B": [1.0, 2.5, 3.8],
                "C": ["2021-01-01", "2022-02-02", "2023-03-03"],
                "D": ["apple", "banana", "orange"],
                "E": [None, None, None],
            }
        )
        self.optimizer = PandasDataTypeOptimizer(ignore_columns=[])

    def test_optimize_float_to_uint(self):
        df_float_to_int = self.df.copy()
        optimized_column = self.optimizer.optimize_numerical_dtype(df_float_to_int, "A")
        self.assertEqual(optimized_column["A"].dtype, np.uint8)

    def test_optimize_float_to_float(self):
        df_float_to_float = self.df.copy()
        optimized_column = self.optimizer.optimize_numerical_dtype(
            df_float_to_float, "B"
        )
        self.assertEqual(optimized_column["B"].dtype, np.float16)

    def test_null_to_float(self):
        df_null_to_float = self.df.copy()
        optimized_column = self.optimizer.optimize_numerical_dtype(
            df_null_to_float, "E"
        )
        self.assertEqual(optimized_column["E"].dtype, np.float16)

    def test_set_dates(self):
        df_with_dates = self.df.copy()
        optimized_df = self.optimizer.set_dates(df_with_dates)
        self.assertEqual(optimized_df["C"].dtype, np.dtype("<M8[ns]"))

    def test_set_categorical(self):
        optimized_df = self.optimizer.set_categorical(self.df)
        self.assertEqual(optimized_df["D"].dtype, "category")

    def test_set_numerical(self):
        optimized_df = self.optimizer.set_numerical(self.df)
        self.assertEqual(optimized_df["A"].dtype, np.uint8)
        self.assertEqual(optimized_df["B"].dtype, np.float16)

    def test_type_optimization(self):
        optimized_df = self.optimizer.type_optimization(self.df)
        self.assertEqual(optimized_df["A"].dtype, np.uint8)
        self.assertEqual(optimized_df["B"].dtype, np.float16)
        self.assertEqual(optimized_df["C"].dtype, np.dtype("<M8[ns]"))
        self.assertEqual(optimized_df["D"].dtype, "category")
