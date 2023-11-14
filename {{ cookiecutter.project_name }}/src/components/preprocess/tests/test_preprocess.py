import unittest

from preprocess import run_preprocess


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.project_id = None
        self.gcs_bucket = None

    def test_run_preprocess(self):
        (df, encoders) = run_preprocess(
            self.project_id,
            self.gcs_bucket,
        )
        assert df.shape[0] > 0
        assert encoders is not None
