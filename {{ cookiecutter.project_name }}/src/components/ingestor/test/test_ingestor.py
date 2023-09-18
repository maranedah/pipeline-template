import unittest

import ingestor


class TestIngestor(unittest.TestCase):
    def setUp(self):
        self.num1 = 2
        self.num2 = 3

    def test_run_ingestor(self):
        df = ingestor.run_ingestor("bucket_name")
        assert df.shape[0] > 0
