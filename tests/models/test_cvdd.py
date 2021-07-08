from pathlib import Path

from allennlp.common.testing import ModelTestCase


class TestMetricRankingClassifier(ModelTestCase):
    def setup_method(self) -> None:
        super().setup_method()  # type: ignore
        self.fixture_root = Path("./tests/fixtures")
        self.set_up_model(
            self.fixture_root / "configs" / "cvdd.jsonnet",
            self.fixture_root / "data" / "imdb_corpus.jsonl",
        )

    def test_model_batch_norm_verification(self) -> None:
        pass

    def test_metric_ranking_classifier_can_save_and_load(self) -> None:
        self.set_up_model(
            self.fixture_root / "configs" / "cvdd.jsonnet",
            self.fixture_root / "data" / "imdb_corpus.jsonl",
        )
        self.ensure_model_can_train_save_and_load(self.param_file)
