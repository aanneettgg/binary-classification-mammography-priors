"""Where a window drawn on the index exam falls on an earlier exam of the same breast.

Findings are annotated on the index exam only, so every prior window has to be
placed by a method, and the method is a choice worth comparing: each one is a
`Correspondence` and they run under an otherwise identical pipeline.

**A prior stored as the left-right mirror of its index is placed on its mirror.**
Two exams of one breast are not always stored facing the same way, and no
method finds the right tissue across a mirror - registration can translate,
rotate, scale and bend, but not reflect. So the window is placed on the prior
turned to face as the index does, and the patch is mirrored back when cut.
Which way an image faces is read off its pixels: the edge the breast lies
against.
"""

from abc import ABC, abstractmethod

import numpy as np

from mammo_priors.config import Registry
from mammo_priors.patches.geometry import PixelWindow
from mammo_priors.patches.tissue import BreastTissue
from mammo_priors.recent import Recent

ORIENTATION_PX = 640


class Orientation:
    """Which edge each image's breast lies against, read once per image."""

    def __init__(self, reader, tissue: BreastTissue = None):
        self.reader = reader
        self.tissue = tissue or BreastTissue()
        self._known = Recent(self._against_left, 64)

    def _against_left(self, image: str):
        pixels = self.reader.read(image)
        step = max(1, int(np.ceil(max(pixels.shape) / ORIENTATION_PX)))
        return self.tissue.against_left(self.tissue.mask(pixels[::step, ::step]))

    def mirrored(self, index: str, prior: str) -> bool:
        """Whether `prior` faces the other way; False when either holds no breast."""
        index_left, prior_left = self._known[index], self._known[prior]
        return None not in (index_left, prior_left) and index_left != prior_left


class Correspondence(ABC):
    """One way of placing an index window on a prior image, clamped into its grid."""

    def __init__(self, reader):
        self.reader = reader
        self.orientation = Orientation(reader)

    def mirrored(self, index: str, prior: str) -> bool:
        return self.orientation.mirrored(index, prior)

    def locate(self, window: PixelWindow, index: str, prior: str) -> PixelWindow:
        """Where the window falls on `prior`, in the pixels of `prior` as stored."""
        grid = self.reader.grid(prior)
        mirrored = self.mirrored(index, prior)
        clamped = self._placed(window, index, prior, mirrored).grown(window.size_px, grid)

        return clamped.mirrored(grid) if mirrored else clamped

    @abstractmethod
    def _placed(self, window: PixelWindow, index: str, prior: str,
                mirrored: bool) -> PixelWindow:
        """The window on `prior` turned to face as `index` does, before clamping."""


class SameLocation(Correspondence):
    """The same pixel coordinates as on the index image.

    Assumes only that two exams of one breast are roughly aligned; the wider
    prior margin in the cache absorbs what that misses. The baseline any
    registration has to beat.
    """

    def _placed(self, window, index, prior, mirrored):
        return window


class Registered(Correspondence):
    """The window moved to where a registration carries its centre.

    The window keeps its size, so the prior patch has the magnification of the
    index patch. One transform per pair of images, however many tiles share it.
    """

    def __init__(self, reader, alignment: str):
        super().__init__(reader)
        from mammo_priors.patches.registration import ALIGNMENTS, Thumbnails

        self.alignment = ALIGNMENTS[alignment]()
        self.thumbnails = Thumbnails(reader)
        self.transforms = Recent(self._fitted, 32)

    def _fitted(self, pair: tuple):
        index, prior, mirrored = pair
        return self.alignment.fit(self.thumbnails.of(index), self.thumbnails.of(prior, mirrored))

    def _placed(self, window, index, prior, mirrored):
        from mammo_priors.patches.registration import place

        centre_y, centre_x = place(self.transforms[(index, prior, mirrored)], window.centre_yx)
        size = window.size_px

        return PixelWindow((centre_y - size / 2, centre_x - size / 2), size)


SAME_LOCATION = "same-location"

CORRESPONDENCES = Registry("correspondence", {
    SAME_LOCATION: SameLocation,
    **{name: (lambda reader, name=name: Registered(reader, name))
       for name in ("centroid", "affine", "deformable")},
})
