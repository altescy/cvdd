from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


def test_anomaly_detector() -> None:
    archive = load_archive("tests/fixtures/models/cvdd.tar.gz")
    predictor = Predictor.from_archive(
        archive, "anomaly_detector", extra_args={"threshold": 0.3}
    )

    inputs = {"text": "this is a test sentence"}
    outputs = predictor.predict_json(inputs)

    assert "anomaly" in outputs
    assert isinstance(outputs["anomaly"], bool)
