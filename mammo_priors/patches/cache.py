"""The patch cache: one file per patient, under a folder named by the patch config.

Decoding a mammogram is most of the cost of a patch and every patch is read once
per epoch, so patches are cut once. Each crop is stored with a margin around the
window, so training-time jitter and smaller model windows need no re-cut, and
pixels keep their stored bit depth: normalisation happens at load.

**Only the cells that exist are stored**, flat, in grid order, with the
presence mask saying where they go back. Grids are mostly holes - a woman whose
finding was drawn on one view has an empty column at every exam - and padding
them on disk would store mostly zeros.
"""

import hashlib
import os

import numpy as np

from mammo_priors.data.tables import RIGHT
from mammo_priors.patches.grid import SampleGrid
from mammo_priors.workspace import safe_name

INDEX_PATCHES, PRIOR_PATCHES, PRESENT = "index", "priors", "present"
STORED_DTYPE = np.uint16


class PatchCache:
    """Reads and writes the patient files of one patch config."""

    def __init__(self, root: str, config):
        self.root = root
        self.config = config

    @property
    def folder(self) -> str:
        return os.path.join(self.root, self.config.digest())

    def path_of(self, sample) -> str:
        # A short hash beside the readable name, so two ids that render to the
        # same safe name cannot share a file.
        tag = hashlib.sha256(sample.key.encode()).hexdigest()[:8]
        return os.path.join(self.folder, safe_name(sample.source),
                            f"{safe_name(sample.patient_id)}-{tag}.npz")

    def holds(self, sample) -> bool:
        return os.path.exists(self.path_of(sample))

    def read(self, sample, *members) -> dict:
        """Named members only: deciding whether an arm can score a patient needs
        the mask, not the megabytes behind it, and npz decompresses per member."""
        with np.load(self.path_of(sample)) as stored:
            return {name: stored[name] for name in members or stored.files}

    def _cut(self, cell, reader) -> np.ndarray:
        margin = self.config.margin_px(prior=cell.timepoint > 0)
        y_min, x_min, y_max, x_max = cell.window.grown(margin, cell.grid).bounds_yx
        patch = reader.read(cell.image)[y_min:y_max, x_min:x_max]
        if patch.shape != (margin, margin):
            raise ValueError(f"{cell.image} is {reader.grid(cell.image)} px, smaller than "
                             f"the {margin} px crop the cache stores; every image must be "
                             f"at least {self.config.prior_margin_px} px on each side")
        if cell.mirrored:
            patch = patch[:, ::-1]
        # Right breasts mirrored so the chest wall is always on the same side:
        # a model that had to learn both would spend capacity on the mirror.
        if self.config.canonicalise_laterality and cell.laterality == RIGHT:
            patch = patch[:, ::-1]

        return np.clip(patch, 0, np.iinfo(STORED_DTYPE).max).astype(STORED_DTYPE)

    def write(self, sample, reader, correspondence) -> int:
        """Cut one patient's grid into one file; returns how many patches it holds."""
        grid = SampleGrid(sample, reader, correspondence, self.config.tile_px)
        patches = {(cell.timepoint, cell.view, cell.tile): self._cut(cell, reader)
                   for cell in grid.cells()}

        present = np.zeros(grid.shape, dtype=bool)
        for key in patches:
            present[key] = True

        def stacked(prior: bool) -> np.ndarray:
            margin = self.config.margin_px(prior)
            keys = sorted(key for key in patches if (key[0] > 0) == prior)
            stack = np.zeros((len(keys), margin, margin), dtype=STORED_DTYPE)
            for at, key in enumerate(keys):
                stack[at] = patches[key]
            return stack

        path = self.path_of(sample)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Written under a temporary name and moved into place, so an interrupted
        # extraction never leaves a half-written file that looks finished.
        partial = path + ".partial.npz"
        np.savez(partial, **{INDEX_PATCHES: stacked(False), PRIOR_PATCHES: stacked(True),
                             PRESENT: present})
        os.replace(partial, path)

        return len(patches)
