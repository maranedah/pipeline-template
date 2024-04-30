import unittest
import polars as pl
from preprocess.FeatureReduction import FilterColumnsTooCorrelated

class TestFilterColumnsTooCorrelated(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'col1': [1, 2, 3, 4, None],
            'col2': [1, 2, None, 4, 5],
            'col3': [1, 2, 3, 4, 5]
        }
        self.df = pl.DataFrame(data)
        self.filter_obj = FilterColumnsTooCorrelated()

    def test_should_be_filtered(self):
        # Test should_be_filtered method
        result = self.filter_obj.should_be_filtered(self.df, 'col1', 'col2')
        self.assertTrue(result)  # col1 should be filtered

    def test_get_corr_between_two_columns(self):
        # Test get_corr_between_two_columns method
        corr = self.filter_obj.get_corr_between_two_columns(self.df, 'col1', 'col2')
        self.assertAlmostEqual(corr, 1.0)  # col1 and col2 are perfectly correlated

    def test_call(self):
        # Test __call__ method
        filtered_df = self.filter_obj(self.df)
        self.assertFalse('col1' in filtered_df.columns)  # col1 should be filtered
        self.assertFalse('col2' in filtered_df.columns)  # col2 should be filtered
