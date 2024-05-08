import unittest

import polars as pl
from preprocess.DataTypeOptimizer import PolarsDataTypeOptimizer
from preprocess.Encodings import Encodings


class TestEncodings(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            "category_col": ["A", "B", "C"],
            "date_col": ["2023-01-01", "2023-02-01", "2023-03-01"],
        }

        self.df = pl.DataFrame(data)
        optimizer = PolarsDataTypeOptimizer()
        self.df = optimizer(self.df)
        self.encodings = Encodings(
            weekday=True, day_of_month=True, day_of_year=True, date_as_unix=True
        )

    def test_get_categorical_encodings(self):
        # Test categorical encoding
        encoded_df = self.encodings.get_categorical_encodings(self.df)
        self.assertTrue("category_col_A" in encoded_df.columns)
        self.assertTrue("category_col_B" in encoded_df.columns)
        self.assertTrue("category_col_C" in encoded_df.columns)
        self.assertTrue("category_col" not in encoded_df.columns)

    def test_get_dates_encodings(self):
        # Test date encoding
        encoded_df = self.encodings.get_dates_encodings(self.df)
        self.assertTrue("date_col_unix" in encoded_df.columns)
        self.assertTrue("date_col_weekday" in encoded_df.columns)
        self.assertTrue("date_col_day_of_month" in encoded_df.columns)
        self.assertTrue("date_col_day_of_year" in encoded_df.columns)

    def test_call(self):
        # Test __call__ method
        processed_df = self.encodings(self.df)
        self.assertTrue("category_col_A" in processed_df.columns)
        self.assertTrue("category_col_B" in processed_df.columns)
        self.assertTrue("category_col_C" in processed_df.columns)
        self.assertTrue("date_col_unix" in processed_df.columns)
        self.assertTrue("date_col_weekday" in processed_df.columns)
        self.assertTrue("date_col_day_of_month" in processed_df.columns)
        self.assertTrue("date_col_day_of_year" in processed_df.columns)
        self.assertTrue("category_col" not in processed_df.columns)
        self.assertTrue("date_col" not in processed_df.columns)
