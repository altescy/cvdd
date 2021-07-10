from typing import List, Optional

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register("anomaly_detector")
class AnomarlyDetector(Predictor):
    def __init__(
        self,
        model: Model,
        dataset_reader: DatasetReader,
        frozen: bool = True,
        threshold: Optional[float] = None,
    ) -> None:
        super().__init__(model, dataset_reader, frozen)
        self._threshold = threshold

    def detect_anomaly(self, json_dict: JsonDict) -> JsonDict:
        if self._threshold is not None:
            json_dict["anomaly"] = json_dict["anomaly_scores"] > self._threshold
        return json_dict

    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = super().predict_instance(instance)
        outputs = self.detect_anomaly(outputs)
        return outputs

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = super().predict_batch_instance(instances)
        outputs = [self.detect_anomaly(x) for x in outputs]
        return outputs

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict["text"]
        reader_has_tokenizer = (
            getattr(self._dataset_reader, "tokenizer", None) is not None
            or getattr(self._dataset_reader, "_tokenizer", None) is not None
        )
        if not reader_has_tokenizer:
            tokenizer = SpacyTokenizer()
            text = tokenizer.tokenize(text)
        return self._dataset_reader.text_to_instance(text)
