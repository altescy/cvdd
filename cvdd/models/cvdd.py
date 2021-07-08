from typing import Any, Dict, List, Optional, cast

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import Auc

from cvdd.modules.distances import CosineDistance, Distance


@Model.register("cvdd")
class CVDD(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        anomaly_label: str,
        text_field_embedder: TextFieldEmbedder,
        context_encoder: Seq2SeqEncoder,
        attention_encoder: Seq2SeqEncoder,
        distance: Optional[Distance] = None,
        alpha: float = 0.0,
        dropout: float = 0.0,
        namespace: str = "tokens",
        label_namespace: str = "labels",
        initializer: Optional[InitializerApplicator] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self._anomaly_label = anomaly_label
        self._text_field_embbedder = text_field_embedder
        self._context_encoder = context_encoder
        self._attention_encoder = attention_encoder
        self._distance = TimeDistributed(distance or CosineDistance())  # type: ignore

        self._alpha = alpha

        self._context_vectors = torch.nn.parameter.Parameter(
            torch.FloatTensor(
                1,
                self._attention_encoder.get_output_dim(),
                self._context_encoder.get_input_dim(),
            ),
            requires_grad=True,
        )

        self._namespace = namespace
        self._label_namespace = label_namespace
        self._anomaly_label = anomaly_label
        self._anomaly_label_index = self.vocab.get_token_index(
            anomaly_label,
            label_namespace,
        )

        self._auc = Auc(self._anomaly_label_index)  # type: ignore

        initializer = initializer or InitializerApplicator()
        initializer(self)

    def forward(  # type: ignore[override]
        self,
        tokens: TextFieldTensors,
        label: Optional[torch.LongTensor] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, max_length, embedding_dim)
        embeddings = self._text_field_embbedder(tokens)
        # Shape: (batch_size, max_length, 1)
        mask = util.get_text_field_mask(tokens)

        # Shape: (batch_size, max_length, encoding_dim)
        encodings = self._context_encoder(embeddings, mask)

        batch_size, _, encoding_dim = encodings.size()

        # Shape: (batch_size, max_length, num_heads)
        attentions = self._attention_encoder(embeddings, mask)
        attentions = util.masked_softmax(
            attentions,
            mask=cast(torch.BoolTensor, mask.unsqueeze(-1)),
            dim=1,
        )

        num_heads = attentions.size(-1)

        # Shape: (batch_size, num_heads, encoding_dim)
        matrix = torch.bmm(
            attentions.transpose(1, 2),
            encodings,
        )
        # Shape: (batch_size, num_heads, encoding_dim)
        context_vectors = self._context_vectors.expand(
            batch_size, num_heads, encoding_dim
        )

        # Shape: (batch_size, num_heads)
        distances = self._distance(matrix, context_vectors).reshape(
            batch_size, num_heads
        )
        # Shape: (batch_size, num_heads)
        sigmas = (self._alpha * distances).softmax(1)

        loss = (sigmas * distances).sum(1).mean()

        # Shape: (batch_size, )
        anomaly_scores = distances.mean(-1)

        output_dict: Dict[str, torch.Tensor] = {}
        output_dict["loss"] = loss
        output_dict["anomaly_score"] = anomaly_scores
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)

        if label is not None:
            binary_label = (label == self._anomaly_label_index).long()
            self._auc(anomaly_scores, binary_label)

        return output_dict

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        tokens: List[List[str]] = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(
                        int(token_id.item()), namespace=self._namespace
                    )
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens  # type: ignore
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"auc": self._auc.get_metric(reset)}
        return metrics
