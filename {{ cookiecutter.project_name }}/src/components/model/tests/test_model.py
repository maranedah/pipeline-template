import unittest
from unittest.mock import patch

from model import run_model
from test_utils import SklearnModelMock


class TestModel(unittest.TestCase):
    def setUp(self):
        self.project_id = "{{ cookiecutter.project_id }}"
        self.dataset_uri = (
            "gs://{{ cookiecutter.project_name }}-bucket-test/preprocessed.gzip"
        )
        self.split_ratio = "6:2:2"
        self.model_str = "LGBMClassifier"
        self.model_uri = ""
        self.metrics_uri = ""

    @patch("lightgbm.LGBMClassifier", SklearnModelMock)
    def test_run_model(self):
        model, metrics = run_model(
            self.project_id,
            self.dataset_uri,
            self.split_ratio,
            self.model_str,
        )
        assert isinstance(metrics, dict)
