"""Reads the pixels of one image, whatever format it is stored in.

DICOM is read with pydicom and every other format with Pillow (16-bit PNG and
TIFF keep their full depth). A reader holds the images it read most recently,
because a patient's tiles, her priors and a registration all ask for the same
few images.
"""

import os

import numpy as np
from PIL import Image

from mammo_priors.recent import Recent

DICOM_SUFFIXES = (".dcm", ".dicom", "")


class ImageReader:
    """Images under `root`, addressed by the path the image table states.

    **MONOCHROME1 DICOM is inverted** so that breast is bright in every image.
    Polarity can differ even between two exams of one woman, and left as stored
    it would differ between her index exam and her own prior.
    """

    MONOCHROME1 = "MONOCHROME1"

    def __init__(self, root: str = "", kept: int = 16):
        self.root = root
        self._held = Recent(self._read, kept)

    def _path(self, image: str) -> str:
        return image if os.path.isabs(image) else os.path.join(self.root, image)

    def _read(self, image: str) -> np.ndarray:
        path = self._path(image)
        if os.path.splitext(path)[1].lower() in DICOM_SUFFIXES:
            return self._dicom(path)

        with Image.open(path) as opened:
            pixels = np.asarray(opened)
        if pixels.ndim == 3:
            pixels = pixels[..., :3].mean(axis=-1)

        return pixels

    def _dicom(self, path: str) -> np.ndarray:
        import pydicom

        stored = pydicom.dcmread(path)
        pixels = stored.pixel_array
        if getattr(stored, "PhotometricInterpretation", None) == self.MONOCHROME1:
            pixels = pixels.max() - pixels

        return pixels

    def read(self, image: str) -> np.ndarray:
        return self._held[image]

    def grid(self, image: str) -> tuple:
        """The (rows, columns) of one image: windows are clamped into the real grid."""
        return tuple(self.read(image).shape[:2])
