"""The ablation arms, as masks over one grid and never as models.

If the arms ran through different code paths, a difference between two of them
would also contain whatever else differed between those paths. So an arm is a
mask: every arm runs the same encoder, aggregator and loss, and only the visible
cells change. A new arm is one more `Arm(...)` line, not new code.
"""

from dataclasses import dataclass, replace

import numpy as np

from mammo_priors.data.tables import VIEWS


@dataclass(frozen=True)
class Arm:
    """Which cells of a (timepoint, view, tile) grid an arm may see.

    t = 0 is the index exam and t grows into the past, so "index only" is a
    prefix of the time axis and "index plus priors" a longer one.

    `single_view` keeps the first view a patient actually has, so a patient
    annotated only in MLO still has an A0 grid; an arm scored on fewer patients
    than its neighbour could not be subtracted from it. `view` closes the other
    view at every timepoint, to ask the ladder of CC or MLO alone.
    """

    name: str
    timepoints: slice
    single_view: bool = False
    view: str = None

    def mask(self, present: np.ndarray) -> np.ndarray:
        """The presence mask this arm leaves: a copy, never larger than the input."""
        allowed = np.zeros_like(present)
        allowed[self.timepoints] = present[self.timepoints]

        if self.view is not None:
            allowed[:, np.arange(allowed.shape[1]) != VIEWS.index(self.view)] = False

        if self.single_view:
            holding = allowed.any(axis=(0, 2)).nonzero()[0]
            single = np.zeros_like(allowed)
            if len(holding):
                single[:, holding[0]] = allowed[:, holding[0]]
            return single

        return allowed

    @property
    def sees_priors(self) -> bool:
        return any(timepoint > 0 for timepoint in range(*self.timepoints.indices(2)))

    def through(self, view: str = None) -> "Arm":
        """This arm with only `view` open; the arm itself when `view` is None."""
        return self if view is None else replace(self, view=view)

    def __str__(self):
        return self.name if self.view is None else f"{self.name} {self.view}"


A0 = Arm("A0", slice(0, 1), single_view=True)   # index exam, one view: the floor
A1 = Arm("A1", slice(0, 1))                     # index exam, both views: no history
A2 = Arm("A2", slice(0, 2))                     # index exam + the most recent prior
A3 = Arm("A3", slice(None))                     # index exam + every prior

ARMS = {arm.name: arm for arm in (A0, A1, A2, A3)}

# Which arm each arm is measured against: A1 - A0 is the second view, A2 - A1
# one prior, A3 - A1 the whole history (the primary comparison), A3 - A2 the
# priors beyond the most recent one.
COMPARISONS = (("A1", "A0"), ("A2", "A1"), ("A3", "A1"), ("A3", "A2"))
