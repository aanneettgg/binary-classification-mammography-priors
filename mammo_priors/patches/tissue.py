"""Which pixels of an image are breast, and which edge the breast lies against.

A threshold, not a segmentation: what it separates is exposed detector from
breast, which differ by most of the dynamic range. The background level is read
off the image border rather than assumed dark, so it works whatever the storage
polarity.
"""

import numpy as np

# How far from the border level a pixel must sit to count as breast, as a share
# of the image's own range, so it means the same at every bit depth.
TISSUE_TOLERANCE = 0.05
BORDER_PX = 8


class BreastTissue:
    """Answers questions about pixels it is given; holds no image."""

    def __init__(self, tolerance: float = TISSUE_TOLERANCE):
        self.tolerance = tolerance

    @staticmethod
    def _background(pixels: np.ndarray) -> float:
        """The median of the border ring, which a burnt-in corner label cannot move."""
        border = np.concatenate([pixels[:BORDER_PX].ravel(), pixels[-BORDER_PX:].ravel(),
                                 pixels[:, :BORDER_PX].ravel(), pixels[:, -BORDER_PX:].ravel()])
        return float(np.median(border))

    def mask(self, pixels: np.ndarray) -> np.ndarray:
        span = float(pixels.max()) - float(pixels.min())
        if span <= 0:
            return np.zeros(pixels.shape, dtype=bool)

        return np.abs(pixels.astype(np.float32) - self._background(pixels)) > self.tolerance * span

    @staticmethod
    def against_left(mask: np.ndarray):
        """Whether the breast lies against the left edge; None when there is no breast."""
        columns = np.nonzero(mask)[1]
        if len(columns) == 0:
            return None

        return bool(columns.mean() < mask.shape[1] / 2)
