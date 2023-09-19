import unittest

import preprocess


class TestIngestor(unittest.TestCase):
    def setUp(self):
        self.gcs_bucket = "ml-projects-dev-bucket"

    def test_run_ingestor(self):
        df = preprocess.run_preprocess(self.gcs_bucket)
        assert df.shape[0] > 0
