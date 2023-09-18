import unittest

import ingestor


class TestIngestor(unittest.TestCase):
    def setUp(self):
        self.gcs_bucket = "ml-projects-dev-bucket"

    def test_run_ingestor(self):
        (df) = ingestor.run_ingestor(self.gcs_bucket)
        assert df.shape[0] > 0
