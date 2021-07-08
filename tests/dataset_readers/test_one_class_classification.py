from cvdd.datast_readers import OneClassClassification


def test_one_class_classification_can_read_file() -> None:
    reader = OneClassClassification(
        normal_labels="pos",
        anomalous_labels="neg",
    )

    instances = list(reader.read("tests/fixtures/data/imdb_corpus.jsonl"))
    assert len(instances) == 3
