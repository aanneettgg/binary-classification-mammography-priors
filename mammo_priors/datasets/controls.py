"""Negative controls: test-time changes that ask what a trained model reads.

A gain of A3 over A1 says the priors change the score, not *how*: the model may
compare the finding with the same tissue years earlier, or merely average more
patches, or use the priors as an intensity reference. Each control changes the
input of a model that is not retrained, on the same patients:

| Control | If the priors are read as the woman's own history | If not |
|---|---|---|
| `temporal-shuffle` | a sequence model loses AUC (the masked mean must not move) | exams are ensembled |
| `prior-swap` | AUC falls towards A1 | priors are not read as her own anatomy |

Applied after the arm's mask and the preprocessing, so the index exam reaches
the model exactly as in the plain scoring. Neither changes the presence mask or
the gaps, so a change in score is a change in what the pixels said.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

from mammo_priors.config import Registry


def _timepoints(present: np.ndarray) -> np.ndarray:
    return present.any(axis=(1, 2)).nonzero()[0]


class Control(ABC):
    def bind(self, dataset):
        """Prepare whatever the control needs from the patients it will change."""

    @abstractmethod
    def apply(self, dataset, at: int, patches: torch.Tensor, present: np.ndarray) -> tuple:
        """`(patches, present)` of patient `at` as the control changes them."""

    @abstractmethod
    def changed(self, dataset) -> int:
        """How many patients the control actually changes, reported beside its score."""


class TemporalShuffle(Control):
    """The visible exams put in an order that is never the true one.

    Whole exams move, with their views and tiles. Drawn per patient from the
    seed, so a rescore gives the same order. A patient with one visible exam has
    no order to shuffle and is left as she is.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _order(self, at: int, count: int) -> np.ndarray:
        generator = np.random.default_rng((self.seed, at))
        while True:
            order = generator.permutation(count)
            if (order != np.arange(count)).any():
                return order

    def apply(self, dataset, at, patches, present):
        timepoints = _timepoints(present)
        if len(timepoints) < 2:
            return patches, present

        moved = timepoints[self._order(at, len(timepoints))]
        patches, present = patches.clone(), present.copy()
        patches[torch.from_numpy(timepoints)] = patches[torch.from_numpy(moved)]
        present[timepoints] = present[moved]

        return patches, present

    def changed(self, dataset):
        return sum(len(_timepoints(dataset.visible(at)[1])) >= 2 for at in range(len(dataset)))


class PriorSwap(Control):
    """A patient's priors replaced by another woman's; her index exam kept.

    The donor comes from the same source and set, with visible priors in exactly
    the same cells, so the mask and the gaps are unchanged. Among those, the one
    nearest in mean raw prior intensity - a stand-in for breast density. Chosen
    without regard to the label, which a same-label donor would pass on. A
    patient without a donor keeps her own priors and stays in the evaluation.
    """

    def __init__(self):
        self.donors = {}

    @staticmethod
    def _signature(present: np.ndarray):
        """Which prior cells are visible, trimmed of each axis's empty end."""
        priors = present[1:]
        if not priors.any():
            return None
        timepoints = _timepoints(priors)[-1] + 1
        tiles = priors.any(axis=(0, 1)).nonzero()[0][-1] + 1
        trimmed = priors[:timepoints, :, :tiles]

        return trimmed.shape, trimmed.tobytes()

    def bind(self, dataset):
        pool = {}
        for at, sample in enumerate(dataset.samples):
            grid, present = dataset.visible(at)
            signature = self._signature(present)
            if signature is not None:
                pool.setdefault((sample.source, signature), []).append(
                    (at, float(grid[1:][present[1:]].mean())))

        self.donors = {}
        for members in pool.values():
            for at, intensity in members:
                others = [(abs(intensity - theirs), donor)
                          for donor, theirs in members if donor != at]
                if others:
                    self.donors[at] = min(others)[1]

    def apply(self, dataset, at, patches, present):
        if at not in self.donors:
            return patches, present

        donated, _ = dataset.prepared(self.donors[at])
        timepoints, _, tiles = self._signature(present)[0]
        patches = patches.clone()
        patches[1:1 + timepoints, :, :tiles] = donated[1:1 + timepoints, :, :tiles]

        return patches, present

    def changed(self, dataset):
        return len(self.donors)


CONTROLS = Registry("control", {
    "temporal-shuffle": TemporalShuffle,
    "prior-swap": PriorSwap,
})
