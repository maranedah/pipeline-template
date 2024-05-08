import unittest

import polars as pl
from preprocess.FeatureReduction import FilterColumnsWithManyNulls


class TestFilterColumnsWithManyNulls(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            "col1": [None, None, None, None],
            "col2": [None, None, 3, 4],
            "col3": [1, 2, 3, 4],
            "col4": [1, 2, None, None],
        }
        self.df = pl.DataFrame(data)
        self.filter_obj = FilterColumnsWithManyNulls()

    def test_call(self):
        # Test __call__ method
        filtered_df = self.filter_obj(self.df)
        self.assertEqual(
            len(filtered_df.columns), 3
        )  # Only col3 and col4 should remain
        self.assertTrue("col2" in filtered_df.columns)
        self.assertTrue("col4" in filtered_df.columns)

    def test_call_with_threshold(self):
        # Test __call__ method with threshold parameter
        filter_obj_with_threshold = FilterColumnsWithManyNulls(threshold=0.5)
        filtered_df = filter_obj_with_threshold(self.df)
        self.assertEqual(
            len(filtered_df.columns), 3
        )  # col3 should remain as it has less than 50% nulls

    def test_call_with_ignore_columns(self):
        # Test __call__ method with ignore_columns parameter
        filter_obj_with_ignore = FilterColumnsWithManyNulls(ignore_columns=["col2"])
        filtered_df = filter_obj_with_ignore(self.df)
        self.assertEqual(len(filtered_df.columns), 3)  # col2 should not be filtered
