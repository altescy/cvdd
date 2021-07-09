from typing import Optional, cast

import torch
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from overrides import overrides
from xallennlp.modules.feedforward import FeedForward


@Seq2SeqEncoder.register("feedforward", exist_ok=True)
class FeedForwardEncoder(Seq2SeqEncoder):
    """
    This class applies the `FeedForward` to each item in sequences.

    Registered as a `Seq2SeqEncoder` with name "feedforward".
    """

    def __init__(self, feedforward: FeedForward) -> None:
        super().__init__()
        self._feedforward = feedforward

    @overrides
    def get_input_dim(self) -> int:
        return self._feedforward.get_input_dim()  # type: ignore

    @overrides
    def get_output_dim(self) -> int:
        return self._feedforward.get_output_dim()  # type: ignore

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            return self._feedforward(inputs)  # type: ignore
        else:
            outputs = self._feedforward(inputs)
            return cast(torch.Tensor, outputs * mask.unsqueeze(dim=-1))
