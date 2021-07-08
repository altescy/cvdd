import json
from typing import Any, Dict, Iterator, List, Optional, Union

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import Field, LabelField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Tokenizer


@DatasetReader.register("one_class_classification")
class OneClassClassification(DatasetReader):
    def __init__(
        self,
        normal_labels: Union[str, List[str]],
        anomalous_labels: Union[str, List[str]],
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        text_key: str = "text",
        label_key: str = "label",
        label_namespace: str = "labels",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )

        if isinstance(normal_labels, str):
            normal_labels = [normal_labels]
        if isinstance(anomalous_labels, str):
            anormalous_labels = [anomalous_labels]

        self._normal_labels = normal_labels
        self._anomalous_labels = anormalous_labels
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._text_key = text_key
        self._label_key = label_key
        self._label_namespace = label_namespace

    def _read(self, file_path: str) -> Iterator[Instance]:
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            for line_num, line in self.shard_iterable(enumerate(data_file)):
                row = json.loads(line)

                text = row.get(self._text_key)
                label = row.get(self._label_key)

                if text is None:
                    raise ValueError(
                        f"Invalid line format: {row} (line number {line_num + 1}"
                    )

                # Convert label into binary class: normal/anomalous
                if label is not None:
                    if label in self._normal_labels:
                        label = "normal"
                    elif label in self._anomalous_labels:
                        label = "anomalous"
                    else:
                        continue

                yield self.text_to_instance(
                    text=text,
                    label=label,
                )

    def text_to_instance(  # type: ignore[override]
        self,
        text: str,
        label: Optional[str] = None,
    ) -> Instance:
        tokens = self._tokenizer.tokenize(text)
        text_field = TextField(tokens, self._token_indexers)

        fields: Dict[str, Field[Any]] = {}
        fields["tokens"] = text_field

        metadata: Dict[str, Any] = {}
        metadata["text"] = text

        if label is not None:
            label_field = LabelField(label, label_namespace=self._label_namespace)
            fields["label"] = label_field
            metadata["label"] = label

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
