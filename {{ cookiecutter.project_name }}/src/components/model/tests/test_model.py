import unittest

from model import run_model


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.project_id = "ml-projects-399119"
        self.preprocessed_dataset_uri = "gs://pipeline-template-dev/preprocessed.gzip"
        self.predictions_uri = "gs://pipeline-template-dev/predictions.gzip"
        self.model_uri = ""
        self.metrics_uri = ""

    def test_run_preprocess(self):
        (df, model, metrics) = run_model(
            self.project_id,
            self.preprocessed_dataset_uri,
        )
        df.to_parquet(self.predictions_uri)
        assert df.shape[0] > 0
