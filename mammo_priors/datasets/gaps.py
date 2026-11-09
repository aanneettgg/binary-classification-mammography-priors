"""What the model is told about time between exams.

`rounds` (the default) encodes every gap between two consecutive present exams
as one nominal screening round: the order survives, the duration does not. The
measured interval can carry the label on its own - it can differ between the
classes within a programme, and it identifies the programme across sources - so
a model given `months` may read the label from when exams happened rather than
from what they show. Run the availability probe before trusting `months`.
"""

from datetime import date

import numpy as np

MONTHS, ROUNDS = "months", "rounds"
INTERVALS = (MONTHS, ROUNDS)
DAYS_PER_MONTH = 30.4375

# Not zero: zero would tell a time-aware model every exam is the newest.
NOMINAL_ROUND_MONTHS = 24.0


def gaps_between(dates: list, present: np.ndarray, interval: str = ROUNDS) -> np.ndarray:
    """The gap in months from each present exam to the next older present one.

    Consecutive gaps between *present* exams, so masking an exam out never
    shifts the time of the others. The oldest present exam gets zero.
    """
    if interval not in INTERVALS:
        raise ValueError(f"interval is one of {INTERVALS}, not {interval!r}")

    gaps = np.zeros(len(dates), dtype=np.float32)
    at = present.reshape(len(dates), -1).any(axis=1).nonzero()[0]
    for newer, older in zip(at, at[1:]):
        days = (date.fromisoformat(dates[newer]) - date.fromisoformat(dates[older])).days
        gaps[newer] = NOMINAL_ROUND_MONTHS if interval == ROUNDS else days / DAYS_PER_MONTH

    return gaps
