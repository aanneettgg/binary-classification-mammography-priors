"""How much of the label an arm could read without looking at a single pixel.

An arm that sees priors is also told which exams exist and how far apart they
are. If those alone predict the label - a history's length can differ between
the classes - a with-prior "gain" may be read from the structure of the input
rather than from the priors' pixels. Run this before interpreting any
with-prior result: a gain is attributable to what the priors *show* only where
this probe is at chance (AUC 0.5).

The arm's own mask and the model's own gap computation are used, so the probe
reads exactly the signal the model is offered. An exam counts as present when
it exists; which views or how many tiles it holds is not seen, so the probe is
a lower bound on the label information in the input's shape.

Logistic regression on purpose: if a linear model separates the labels from a
few counts and durations, so does the aggregator.
"""

import numpy as np

from mammo_priors.data.balance import balance_named
from mammo_priors.data.splits import TEST, TRAIN
from mammo_priors.datasets.gaps import ROUNDS, gaps_between
from mammo_priors.evaluation.metrics import bootstrap_auc


class AvailabilityProbe:
    """A ridge-penalised logistic regression on one arm's exam mask and gaps."""

    RIDGE = 1.0
    STEPS = 50

    def __init__(self, arm, interval: str = ROUNDS):
        self.arm = arm
        self.interval = interval
        self.width = None
        self.centre, self.scale, self.weights = 0.0, 1.0, None

    def _row(self, sample) -> np.ndarray:
        exams = len(sample.exams)
        present = self.arm.mask(np.ones((exams, 1, 1), dtype=bool))
        shown = present.any(axis=(1, 2)).astype(float)
        gaps = gaps_between(sample.dates, present, self.interval).astype(float)

        row = np.zeros(2 * self.width)
        kept = min(exams, self.width)
        row[:kept] = shown[:kept]
        row[self.width:self.width + kept] = gaps[:kept]
        return row

    def _design(self, samples: list) -> np.ndarray:
        rows = (np.stack([self._row(sample) for sample in samples]) - self.centre) / self.scale
        return np.hstack([np.ones((len(samples), 1)), rows])

    def fit(self, samples: list, weights: np.ndarray = None) -> "AvailabilityProbe":
        """Fitted as training draws the patients, so it sees what the model sees."""
        labels = np.array([sample.label for sample in samples], dtype=float)
        weights = np.ones(len(samples)) if weights is None else weights / np.mean(weights)
        self.width = max(len(sample.exams) for sample in samples)
        self.centre, self.scale = 0.0, 1.0
        raw = self._design(samples)[:, 1:]
        self.centre = raw.mean(axis=0)
        # A constant column (the index exam is always shown) stays constant.
        self.scale = np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1.0)

        design = self._design(samples)
        penalty = self.RIDGE * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        fitted = np.zeros(design.shape[1])
        for _ in range(self.STEPS):
            chance = 1.0 / (1.0 + np.exp(-design @ fitted))
            gradient = design.T @ (weights * (chance - labels)) + penalty @ fitted
            hessian = (design * (weights * chance * (1 - chance))[:, None]).T @ design + penalty
            fitted -= np.linalg.solve(hessian, gradient)
        self.weights = fitted

        return self

    def logits(self, samples: list) -> np.ndarray:
        return self._design(samples) @ self.weights


def probe(cohort, split, arm, interval: str = ROUNDS) -> list:
    """Fitted on the split's training set; scored on its test set and each external source."""
    balance = balance_named(split.balance)
    train = split.samples_of(cohort, TRAIN)
    fitted = AvailabilityProbe(arm, interval).fit(train, balance.weights(train))

    rows = []
    for name, samples, groups in (
            [(TEST, split.samples_of(cohort, TEST), balance)]
            + [(source, cohort.of_sources([source]), balance_named("none"))
               for source in split.external]):
        keys = [sample.key for sample in samples]
        result = bootstrap_auc(fitted.logits(samples),
                               np.array([sample.label for sample in samples]),
                               groups=groups.groups(keys))
        rows.append({"arm": str(arm), "set": name, "patients": len(samples),
                     "auc": result["auc"], "low": result["low"], "high": result["high"]})

    return rows
