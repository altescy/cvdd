from typing import cast

import torch
from allennlp.common.registrable import Registrable
from allennlp.nn.util import tiny_value_of_dtype


class Distance(torch.nn.Module, Registrable):
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


@Distance.register("cosine")
class CosineDistance(Distance):
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Shape: (batch_size, embedding_dim)
        source_norm = source / (
            source.norm(p=2, dim=-1, keepdim=True)  # type: ignore
            + tiny_value_of_dtype(source.dtype)
        )
        # Shape: (batch_size, embedding_dim)
        target_norm = target / (
            target.norm(p=2, dim=-1, keepdim=True)  # type: ignore
            + tiny_value_of_dtype(target.dtype)
        )
        # Shape: (batch_size, )
        distances = (source_norm * target_norm).sum(-1)
        distances = 0.5 * (1 - distances)
        return cast(torch.Tensor, distances)
