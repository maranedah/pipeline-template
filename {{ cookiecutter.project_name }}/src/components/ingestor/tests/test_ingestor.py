import unittest

from ingestor import run_ingestor


class TestIngestor(unittest.TestCase):
    def setUp(self):
        self.project_id = "ml-projects-399119"
        self.gcs_bucket = "gs://pipeline-template-dev"

    def test_run_ingestor(self):
        (df) = run_ingestor(
            self.project_id,
            self.gcs_bucket,
        )
        assert df.shape[0] > 0
