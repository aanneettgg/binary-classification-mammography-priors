"""What every grid goes through before the model, and what only training adds.

**Preprocessing is deterministic and shared** by every arm, encoder and set, so
an arm difference never contains a normalisation difference. It depends on the
patient in hand only - never on a dataset or batch statistic, which would
change the moment a source was added.

**Augmentation is random and training-only**, and one draw covers a patient's
whole grid: a flip or a brightness shift drawn per cell would perturb an exam
differently from its own prior and erase the change between them.
"""

import numpy as np
import torch

from mammo_priors.config import AugmentConfig, PreprocessConfig

# The eight orientations a quarter turn and a flip reach, which test-time
# augmentation averages over: exactly the ones training was augmented with.
DIHEDRAL = tuple((turns, flipped) for turns in range(4) for flipped in (False, True))


def dihedral_views(patches: torch.Tensor):
    for turns, flipped in DIHEDRAL:
        view = torch.rot90(patches, turns, dims=[-2, -1])
        yield torch.flip(view, dims=[-1]) if flipped else view


class GridPreprocess:
    """One window over the present cells of the whole grid, then standardised.

    One window for the index exam and every prior together, so a lesion is
    comparable with its own earlier self. Absent cells are excluded from the
    percentiles: a grid is mostly padding.
    """

    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()

    def __call__(self, grid: np.ndarray, present: np.ndarray) -> torch.Tensor:
        """[T, V, K, H, W] raw -> the same shape, float32, standardised."""
        cells = grid[present]
        low, high = (np.percentile(cells, self.config.window_percentiles)
                     if cells.size else (0.0, 0.0))
        if high > low:
            scaled = (np.clip(grid.astype(np.float32), low, high) - low) / (high - low)
        else:
            scaled = np.full(grid.shape, 0.5, dtype=np.float32)

        return torch.from_numpy(((scaled - self.config.mean) / self.config.std).astype(np.float32))


class Augment:
    """Flips, quarter turns, brightness and contrast, and the window's shift.

    Applied to the standardised tensor, so the percentile window cannot undo a
    brightness shift. Its randomness comes from a seeded generator, so a run is
    reproducible from its seed.
    """

    def __init__(self, config: AugmentConfig = None, generator: torch.Generator = None):
        self.config = config or AugmentConfig()
        self.generator = generator or torch.Generator()

    def _chance(self) -> float:
        return float(torch.rand(1, generator=self.generator).item())

    def _uniform(self, span: float) -> float:
        return 1.0 + span * (2.0 * self._chance() - 1.0)

    def offset(self) -> tuple:
        """How far the window sits off the crop's centre; applied where it is cut."""
        if not self.config.shift_px:
            return 0, 0
        span = 2 * self.config.shift_px + 1
        return tuple(int(self._chance() * span) - self.config.shift_px for _ in range(2))

    def __call__(self, patches: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        if self._chance() < self.config.flip_probability:
            patches = torch.flip(patches, dims=[-1])
        if self._chance() < self.config.flip_probability:
            patches = torch.flip(patches, dims=[-2])
        if self.config.quarter_turns:
            patches = torch.rot90(patches, int(self._chance() * 4) % 4, dims=[-2, -1])
        if self.config.contrast:
            centre = patches[present].mean() if present.any() else patches.mean()
            patches = (patches - centre) * self._uniform(self.config.contrast) + centre
        if self.config.brightness:
            patches = patches + (self._uniform(self.config.brightness) - 1.0)

        return patches
