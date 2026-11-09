"""One patch in, one feature vector out, chosen by name from a registry.

To add an encoder (a mammography foundation model, say), subclass `Encoder`,
report its feature width in `dimension`, and add it to `ENCODERS`; nothing else
has to change.
"""

from abc import ABC, abstractmethod
from functools import partial

import torch
from torch import nn
from torchvision import models

from mammo_priors.config import Registry


class Encoder(nn.Module, ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Width of the feature vector, read from the model rather than declared."""

    @abstractmethod
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """[B, 1, H, W] -> [B, D]."""


class ResNetEncoder(Encoder):
    """A torchvision ResNet with its classifier removed and one input channel.

    Pretrained colour weights of the first convolution are summed over the
    channels: a grey image fed to a three-channel filter is exactly that sum, so
    the response the weights were trained to give is kept.
    """

    def __init__(self, depth: int = 18, weights: str = None):
        super().__init__()
        self.depth = depth
        resnet = getattr(models, f"resnet{depth}")()
        if weights is not None:
            state = models.get_model_weights(f"resnet{depth}")[weights].get_state_dict(progress=True)
            resnet.load_state_dict(state)
        first = resnet.conv1
        resnet.conv1 = nn.Conv2d(1, first.out_channels, first.kernel_size, first.stride,
                                 first.padding, bias=False)
        with torch.no_grad():
            resnet.conv1.weight.copy_(first.weight.sum(dim=1, keepdim=True))
        self._dimension = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.resnet = resnet

    @property
    def dimension(self) -> int:
        return self._dimension

    def forward(self, patches):
        return self.resnet(patches)


# One weight recipe for every depth, so the depths differ in depth alone.
IMAGENET = "IMAGENET1K_V1"

ENCODERS = Registry("encoder", {
    # Random initialisation: for plumbing tests, where pretrained weights would
    # hide an error behind a plausible loss curve.
    **{f"resnet{depth}": partial(ResNetEncoder, depth=depth) for depth in (18, 34, 50)},
    **{f"resnet{depth}-imagenet": partial(ResNetEncoder, depth=depth, weights=IMAGENET)
       for depth in (18, 34, 50)},
})
