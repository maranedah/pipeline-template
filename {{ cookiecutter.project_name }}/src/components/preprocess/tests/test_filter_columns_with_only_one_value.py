import unittest

import polars as pl
from preprocess.FeatureReduction import FilterColumnsWithOnlyOneValue


class TestFilterColumnsWithOnlyOneValue(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            "col1": [1, 1, 1, 1],
            "col2": [None, None, None, None],
            "col3": ["A", "A", "A", "A"],
            "col4": ["A", "B", "C", "D"],
        }
        self.df = pl.DataFrame(data)
        self.filter_obj = FilterColumnsWithOnlyOneValue()

    def test_call(self):
        # Test __call__ method
        filtered_df = self.filter_obj(self.df)
        self.assertEqual(len(filtered_df.columns), 2)  # Only col4 should remain

    def test_call_with_ignore_columns(self):
        # Test __call__ method with ignore_columns parameter
        filter_obj_with_ignore = FilterColumnsWithOnlyOneValue(ignore_columns=["col3"])
        filtered_df = filter_obj_with_ignore(self.df)
        self.assertEqual(len(filtered_df.columns), 3)  # col3 should not be filtered
        self.assertTrue("col3" in filtered_df.columns)
