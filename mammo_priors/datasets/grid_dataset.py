"""One patient's grid as tensors, out of the cohort and the patch cache.

A patient is one sample and her cells are addressed by (timepoint, view, tile),
so an arm is a mask and a batch is a stack of masks.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from mammo_priors.datasets.gaps import ROUNDS, gaps_between
from mammo_priors.datasets.preprocessing import GridPreprocess
from mammo_priors.patches.cache import INDEX_PATCHES, PRESENT, PRIOR_PATCHES


@dataclass
class GridSample:
    patches: torch.Tensor      # [T, V, K, H, W] float
    present: torch.Tensor      # [T, V, K] bool
    gap_months: torch.Tensor   # [T]
    label: torch.Tensor        # scalar float, 1 for malignant
    key: str


def collate(batch: list) -> dict:
    """Stack patients of different history length and tile count.

    Padded with `present = False`: a padded cell is exactly a missing cell, which
    the aggregator already has to ignore.
    """
    timepoints = max(sample.patches.shape[0] for sample in batch)
    tiles = max(sample.patches.shape[2] for sample in batch)
    views, size = batch[0].patches.shape[1], batch[0].patches.shape[-1]

    patches = torch.zeros(len(batch), timepoints, views, tiles, size, size)
    present = torch.zeros(len(batch), timepoints, views, tiles, dtype=torch.bool)
    gaps = torch.zeros(len(batch), timepoints)
    for at, sample in enumerate(batch):
        t, _, k = sample.present.shape
        patches[at, :t, :, :k] = sample.patches
        present[at, :t, :, :k] = sample.present
        gaps[at, :t] = sample.gap_months

    return {"patches": patches, "present": present, "gap_months": gaps,
            "label": torch.stack([sample.label for sample in batch]),
            "key": [sample.key for sample in batch]}


def sliced(batch: dict, max_cells: int = None):
    """A batch as consecutive runs of whole patients, each within `max_cells` patches.

    A batch's cost is its patches, not its patients, and a patient holds one to
    dozens of them. A patient is never cut, since the aggregator needs her whole
    grid; one larger than the budget is a slice of her own.
    """
    if max_cells is None:
        yield batch
        return

    cells = batch["present"].flatten(1).sum(dim=1).tolist()
    start, held = 0, 0
    for at, count in enumerate(cells):
        if at > start and held + count > max_cells:
            yield {key: value[start:at] for key, value in batch.items()}
            start, held = at, 0
        held += count
    yield {key: value[start:] for key, value in batch.items()}


class GridDataset(Dataset):
    """The patients of one arm, over a cache that has already been cut.

    Patients the arm leaves nothing of are dropped and counted: an arm scored
    on a different set of patients than the arm it is subtracted from would
    make the difference meaningless (the report pairs on shared patients).
    """

    def __init__(self, samples: list, cache, arm, size_px: int, augment=None,
                 preprocess: GridPreprocess = None, interval: str = ROUNDS, control=None):
        self.cache = cache
        self.arm = arm
        self.size_px = size_px
        self.augment = augment
        self.preprocess = preprocess or GridPreprocess()
        self.interval = interval
        self.samples = [sample for sample in samples
                        if arm.mask(cache.read(sample, PRESENT)[PRESENT]).any()]
        if len(self.samples) < len(samples):
            print(f"  {arm}: {len(samples) - len(self.samples)} of {len(samples)} "
                  f"patients have no cell this arm can see")
        if control is not None and augment is not None:
            raise ValueError("a control changes a set that is scored, never one trained on")
        self.control = control
        if control is not None:
            control.bind(self)

    def __len__(self):
        return len(self.samples)

    def _centred(self, patches: np.ndarray, offset: tuple) -> np.ndarray:
        """The model's window out of the stored margin, moved by `offset`."""
        margin = patches.shape[-1]
        rows, columns = ((margin - self.size_px) // 2 + shift for shift in offset)
        if min(rows, columns) < 0 or max(rows, columns) + self.size_px > margin:
            raise ValueError(f"a {self.size_px} px window shifted by {offset} does not "
                             f"fit the {margin} px the cache holds")

        return patches[..., rows:rows + self.size_px, columns:columns + self.size_px]

    def _scattered(self, patches: np.ndarray, present: np.ndarray, offset: tuple) -> np.ndarray:
        grid = np.zeros((*present.shape, self.size_px, self.size_px), dtype=patches.dtype)
        grid[present] = self._centred(patches, offset)
        return grid

    def visible(self, at: int, offset: tuple = (0, 0)) -> tuple:
        """Patient `at`'s raw grid with only the arm's cells, and the arm's mask.

        Masked right after reading, so a hidden cell cannot reach the model even
        if something downstream forgot the mask.
        """
        stored = self.cache.read(self.samples[at])
        on_disk = stored[PRESENT]
        grid = np.concatenate([self._scattered(stored[INDEX_PATCHES], on_disk[:1], offset),
                               self._scattered(stored[PRIOR_PATCHES], on_disk[1:], offset)])
        present = self.arm.mask(on_disk)

        return grid * present[..., None, None], present

    def prepared(self, at: int, offset: tuple = (0, 0)) -> tuple:
        """Patient `at`'s grid as the model sees it before augmentation, and her mask."""
        grid, present = self.visible(at, offset)
        return self.preprocess(grid, present), present

    def __getitem__(self, at: int) -> GridSample:
        sample = self.samples[at]
        offset = self.augment.offset() if self.augment is not None else (0, 0)
        patches, present = self.prepared(at, offset)
        if self.control is not None:
            patches, present = self.control.apply(self, at, patches, present)

        mask = torch.from_numpy(present)
        if self.augment is not None:
            patches = self.augment(patches, mask)

        # Zero padding, unconditionally: nothing reads those cells, and padding
        # that is not zero would make a masking bug invisible.
        return GridSample(
            patches=patches * mask[..., None, None],
            present=mask,
            gap_months=torch.from_numpy(gaps_between(sample.dates, present, self.interval)),
            label=torch.tensor(float(sample.label)),
            key=sample.key,
        )
