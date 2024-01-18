import unittest
from unittest.mock import patch

from model import run_model
from test_utils import SklearnModelMock


class TestModel(unittest.TestCase):
    def setUp(self):
        self.project_id = "{{ cookiecutter.project_id }}"
        self.dataset_uri = "{{ cookiecutter.gcs_bucket }}-test/preprocessed.gzip"
        self.model_uri = ""
        self.metrics_uri = ""

    @patch.object("lightgbm.LGBMRegressor", SklearnModelMock)
    def test_run_model(self):
        model, metrics = run_model(
            self.project_id,
            self.preprocessed_dataset_uri,
        )
        assert isinstance(metrics, dict)
