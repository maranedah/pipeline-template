import unittest

from preprocess import run_preprocess


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.project_id = "ml-projects-399119"
        self.palmer_penguins_uri = "gs://pipeline-template-dev/palmer_penguins.gzip"
        self.preprocessed_dataset_uri = "gs://pipeline-template-dev/preprocessed.gzip"

    def test_run_preprocess(self):
        (df, encoders) = run_preprocess(
            self.project_id,
            self.palmer_penguins_uri,
        )
        df.to_parquet(self.preprocessed_dataset_uri)
        assert df.shape[0] > 0
        assert encoders is not None
