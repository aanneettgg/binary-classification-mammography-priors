"""Where the tissue under the index window moved to on a prior of the same breast.

Needs SimpleITK (`pip install mammo-priors[registration]`); same-location
placement does not import this module.

A transform maps a point of the index (fixed) image to the prior (moving) image.
Images are registered as thumbnails whose physical space *is* their
full-resolution pixel grid - BinShrink keeps the physical extent and sets the
spacing to the shrink factor - so a transform found on thumbnails takes and
returns full-resolution pixels. SimpleITK points are (x, y); `place` is the only
place that flip is written.

Mutual information rather than correlation, because two exams years apart -
possibly from two machines - do not share one intensity mapping. Every
intensity method starts from the centroid alignment and keeps its result only
if it beats that start on its own metric, so a wandering optimiser can never
make a method worse than the cheap one it refines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

from mammo_priors.patches.tissue import BreastTissue
from mammo_priors.recent import Recent

# The longest side of a thumbnail. Registration aligns the breast, not its
# microcalcifications, and a full-resolution image would make every metric
# evaluation tens of times dearer for no better placement of the window.
THUMBNAIL_PX = 640
THUMBNAILS_KEPT = 32


@dataclass(frozen=True)
class Thumbnail:
    """One image shrunk for registration, with its breast mask (the metric's support)."""

    pixels: sitk.Image
    mask: sitk.Image
    holds_breast: bool


class Thumbnails(Recent):
    """Thumbnails of images read through a reader, keyed by (image, mirrored).

    A mirrored thumbnail is shrunk from the mirrored full image, so its physical
    space is that image's pixel grid exactly.
    """

    def __init__(self, reader, longest_px: int = THUMBNAIL_PX, tissue: BreastTissue = None):
        super().__init__(self._made, THUMBNAILS_KEPT)
        self.reader = reader
        self.longest_px = longest_px
        self.tissue = tissue or BreastTissue()

    def of(self, image: str, mirrored: bool = False) -> Thumbnail:
        return self[(image, mirrored)]

    def _made(self, key: tuple) -> Thumbnail:
        image, mirrored = key
        pixels = self.reader.read(image)
        if mirrored:
            pixels = pixels[:, ::-1]
        full = sitk.GetImageFromArray(np.ascontiguousarray(pixels, dtype=np.float32))
        factor = max(1, int(np.ceil(max(full.GetSize()) / self.longest_px)))
        shrunk = sitk.BinShrink(full, [factor, factor])

        array = sitk.GetArrayViewFromImage(shrunk)
        breast = self.tissue.mask(array)
        mask = sitk.GetImageFromArray(breast.astype(np.uint8))
        mask.CopyInformation(shrunk)

        return Thumbnail(_rescaled(shrunk, array, breast), mask, bool(breast.any()))


def _rescaled(image: sitk.Image, array: np.ndarray, breast: np.ndarray) -> sitk.Image:
    """The breast's own 1st-99th percentile stretched to [0, 1], background 0."""
    inside = array[breast] if breast.any() else array.ravel()
    low, high = np.percentile(inside, [1, 99])
    scaled = np.clip((array - low) / max(high - low, 1e-6), 0, 1) * breast
    result = sitk.GetImageFromArray(scaled.astype(np.float32))
    result.CopyInformation(image)

    return result


def place(transform: sitk.Transform, centre_yx: tuple) -> tuple:
    """Where an index point (row, column) falls on the prior, as (row, column)."""
    x, y = transform.TransformPoint((float(centre_yx[1]), float(centre_yx[0])))
    return (y, x)


class Alignment(ABC):
    """One way of finding the transform from an index image to a prior."""

    @abstractmethod
    def fit(self, fixed: Thumbnail, moving: Thumbnail) -> sitk.Transform:
        """The transform taking points of `fixed` (the index) onto `moving`."""


class CentroidAlignment(Alignment):
    """A translation laying the breast masks' centres of mass on each other.

    Corrects where the breast sits on the detector, not how it was compressed.
    The identity where either image holds no breast.
    """

    def fit(self, fixed, moving):
        if not (fixed.holds_breast and moving.holds_breast):
            return sitk.Euler2DTransform()

        return sitk.CenteredTransformInitializer(
            sitk.Cast(fixed.mask, sitk.sitkFloat32), sitk.Cast(moving.mask, sitk.sitkFloat32),
            sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)


class IntensityAlignment(Alignment):
    """A transform optimised on masked mutual information, on top of `start`'s."""

    SHRINK_FACTORS = [4, 2, 1]
    SMOOTHING_SIGMAS = [2, 1, 0]
    HISTOGRAM_BINS = 32

    def __init__(self, start: Alignment, iterations: int):
        self.start = start
        self.iterations = iterations

    @abstractmethod
    def _fresh(self, fixed: Thumbnail) -> sitk.Transform:
        """The identity of the transform family this method optimises."""

    @abstractmethod
    def _optimise_with(self, method: sitk.ImageRegistrationMethod):
        """Sets the optimiser suited to that family."""

    def _method(self, fixed, moving) -> sitk.ImageRegistrationMethod:
        method = sitk.ImageRegistrationMethod()
        method.SetMetricAsMattesMutualInformation(self.HISTOGRAM_BINS)
        # Every pixel rather than a random sample: a sample is drawn per thread,
        # so even a seeded one places the window differently from run to run.
        method.SetMetricSamplingStrategy(method.NONE)
        method.SetMetricFixedMask(fixed.mask)
        method.SetMetricMovingMask(moving.mask)
        method.SetInterpolator(sitk.sitkLinear)
        method.SetShrinkFactorsPerLevel(self.SHRINK_FACTORS)
        method.SetSmoothingSigmasPerLevel(self.SMOOTHING_SIGMAS)
        self._optimise_with(method)

        return method

    def fit(self, fixed, moving):
        initial = self.start.fit(fixed, moving)
        refined = self._fresh(fixed)

        method = self._method(fixed, moving)
        method.SetMovingInitialTransform(initial)
        method.SetInitialTransform(refined, inPlace=True)
        try:
            before = method.MetricEvaluate(fixed.pixels, moving.pixels)
            method.Execute(fixed.pixels, moving.pixels)
            after = method.MetricEvaluate(fixed.pixels, moving.pixels)
        except RuntimeError:
            # Masks that never meet leave the metric nothing to score; the
            # start is still a placement.
            return initial
        if not after < before:
            return initial

        composite = sitk.CompositeTransform(initial)
        composite.AddTransform(refined)

        return composite


class AffineAlignment(IntensityAlignment):
    """Translation, rotation, scale and shear: positioning and compression, globally."""

    FIRST_STEP_PX = 32.0

    def __init__(self, iterations: int = 200):
        super().__init__(CentroidAlignment(), iterations)

    def _fresh(self, fixed):
        transform = sitk.AffineTransform(2)
        transform.SetCenter(_centre_of(fixed.mask))
        return transform

    def _optimise_with(self, method):
        method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=self.FIRST_STEP_PX, minStep=0.5,
            numberOfIterations=self.iterations, relaxationFactor=0.5)
        method.SetOptimizerScalesFromPhysicalShift()


class DeformableAlignment(IntensityAlignment):
    """A coarse B-spline field on top of the affine: tissue that moved by different amounts."""

    # A coarse mesh learns nothing from the finest level, which would be most
    # of the cost.
    SHRINK_FACTORS = [4, 2]
    SMOOTHING_SIGMAS = [2, 1]

    def __init__(self, mesh: int = 6, iterations: int = 100):
        super().__init__(AffineAlignment(), iterations)
        self.mesh = mesh

    def _fresh(self, fixed):
        return sitk.BSplineTransformInitializer(fixed.pixels, [self.mesh, self.mesh])

    def _optimise_with(self, method):
        method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                                    numberOfIterations=self.iterations)


def _centre_of(mask: sitk.Image) -> tuple:
    statistics = sitk.LabelShapeStatisticsImageFilter()
    statistics.Execute(mask)
    if statistics.HasLabel(1):
        return statistics.GetCentroid(1)

    return mask.TransformContinuousIndexToPhysicalPoint([size / 2 for size in mask.GetSize()])


ALIGNMENTS = {
    "centroid": CentroidAlignment,
    "affine": AffineAlignment,
    "deformable": DeformableAlignment,
}
