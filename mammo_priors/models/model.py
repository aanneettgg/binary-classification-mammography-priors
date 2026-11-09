"""An encoder applied to every visible patch, an aggregator over the result.

**Only the present cells reach the encoder.** A masked patch therefore cannot
influence a logit even through a bug, and A1 on a woman with six exams costs one
exam's encoding rather than six.
"""

from contextlib import contextmanager

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from mammo_priors.models.aggregator import AGGREGATORS
from mammo_priors.models.encoder import ENCODERS


class GridModel(nn.Module):
    """One patient's grid of patches in, one logit out.

    `chunk` is the most patches the encoder holds activations for at once; a
    patient with more is encoded in checkpointed chunks (see `_encoded_in_chunks`).
    """

    def __init__(self, encoder, aggregator, chunk: int = None):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.chunk = chunk

    def forward(self, patches, present, gap_months):
        shape = present.shape
        visible = present.flatten()
        selected = patches.flatten(0, 3)[visible].unsqueeze(1)
        encoded = (self.encoder(selected) if self.chunk is None
                   else _encoded_in_chunks(self.encoder, selected, self.chunk))

        features = torch.zeros(len(visible), encoded.shape[-1],
                               dtype=encoded.dtype, device=encoded.device)
        features[visible] = encoded

        return self.aggregator(features.view(*shape, -1), present, gap_months)

    def groups(self, backbone_lr: float, head_lr: float) -> list:
        return [{"params": self.encoder.parameters(), "lr": backbone_lr},
                {"params": self.aggregator.parameters(), "lr": head_lr}]


def build(run) -> GridModel:
    """The model a RunConfig names; the one place the encoder's width meets the head."""
    encoder = ENCODERS.build(run.encoder)
    aggregator = AGGREGATORS.build(run.aggregator, dimension=encoder.dimension)

    return GridModel(encoder, aggregator,
                     chunk=run.max_cells if run.checkpoint_oversized else None)


@contextmanager
def _frozen_statistics(module: nn.Module):
    """BatchNorm normalising as usual but recording nothing (momentum zero).

    Checkpointing runs every chunk forward twice; the recomputation must not move
    the running statistics a second time, or an oversized patient would count
    twice in the statistics validation and test are normalised by.
    """
    layers = [layer for layer in module.modules()
              if isinstance(layer, nn.modules.batchnorm._BatchNorm)]
    saved = [(layer.momentum, None if layer.num_batches_tracked is None
              else layer.num_batches_tracked.clone()) for layer in layers]
    for layer in layers:
        layer.momentum = 0.0
    try:
        yield
    finally:
        for layer, (momentum, tracked) in zip(layers, saved):
            layer.momentum = momentum
            if tracked is not None:
                layer.num_batches_tracked.copy_(tracked)


def _recomputable(module: nn.Module):
    calls = 0

    def run(patches):
        nonlocal calls
        calls += 1
        if calls == 1:
            return module(patches)
        with _frozen_statistics(module):
            return module(patches)

    return run


def _encoded_in_chunks(module: nn.Module, patches: torch.Tensor, chunk: int) -> torch.Tensor:
    """`module(patches)` holding at most `chunk` patches' activations at a time.

    For a patient whose grid does not fit on the GPU even alone. Without a
    gradient there is nothing to keep, and the chunked result is exact.
    """
    if len(patches) <= chunk:
        return module(patches)

    parts = patches.split(chunk)
    if not torch.is_grad_enabled():
        return torch.cat([module(part) for part in parts])

    return torch.cat([checkpoint(_recomputable(module), part, use_reentrant=False)
                      for part in parts])
