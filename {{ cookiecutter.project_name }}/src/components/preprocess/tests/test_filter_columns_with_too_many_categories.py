import unittest
import polars as pl
from preprocess.FeatureReduction import FilterColumnsWithTooManyCategories
from preprocess.DataTypeOptimizer import PolarsDataTypeOptimizer

class TestFilterColumnsWithTooManyCategories(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'col1': ['A', 'B', 'C', 'D'],
            'col2': ['A', 'B', 'C', 'C'],
            'col3': ['A', 'B', 'C', 'A']
        }
        self.df = pl.DataFrame(data)
        optimizer = PolarsDataTypeOptimizer()
        self.df = optimizer(self.df)
        self.filter_obj = FilterColumnsWithTooManyCategories(threshold=4)

    def test_call(self):
        # Test __call__ method
        filtered_df = self.filter_obj(self.df)
        self.assertEqual(len(filtered_df.columns),2)  # One columns should be filtered

