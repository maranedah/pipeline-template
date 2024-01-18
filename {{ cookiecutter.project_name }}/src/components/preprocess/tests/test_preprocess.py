import unittest

from preprocess import run_preprocess


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.project_id = "{{ cookiecutter.project_id }}"
        self.palmer_penguins_uri = (
            "gs://{{ cookiecutter.project_name }}-test/palmer_penguins.gzip"
        )
        self.preprocessed_dataset_uri = (
            "gs://{{ cookiecutter.project_name }}-test/preprocessed.gzip"
        )

    def test_run_preprocess(self):
        (df, encoders) = run_preprocess(
            self.project_id,
            self.palmer_penguins_uri,
        )
        df.to_parquet(self.preprocessed_dataset_uri)
        assert df.shape[0] > 0
        assert encoders is not None
