"""Patient-level AUC, its bootstrap interval, and the paired difference between two runs.

One implementation shared by training, the probe, the controls and the report:
an arm difference is a difference of two numbers, and a metric computed two ways
would hide a second difference inside it.

The patient is the resampling unit, because one sample is one patient.
"""

import numpy as np

RESAMPLES = 2000
CONFIDENCE = 0.95


def _ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, tied values sharing their average rank."""
    order = np.argsort(values, kind="mergesort")
    _, group, counts = np.unique(values[order], return_inverse=True, return_counts=True)
    shared = np.cumsum(counts) - (counts - 1) / 2
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = shared[group.reshape(-1)]

    return ranks


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """The Mann-Whitney AUC; NaN when the set holds one class only."""
    positive = labels == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    return float((_ranks(scores)[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def grouped_auc(scores, labels, groups=None) -> float:
    """The AUC within each group, averaged; the plain AUC without groups.

    For sets pooled from sources with different class balances, where one AUC
    over the pool would credit a model for recognising the source.
    """
    if groups is None:
        return auc(scores, labels)

    return float(np.mean([auc(scores[groups == group], labels[groups == group])
                          for group in np.unique(groups)]))


def _resample(generator, size: int, groups=None) -> np.ndarray:
    """One bootstrap draw of row positions, within each group when there are groups."""
    if groups is None:
        return generator.integers(0, size, size)

    return np.concatenate([members[generator.integers(0, len(members), len(members))]
                           for members in (np.flatnonzero(groups == group)
                                           for group in np.unique(groups))])


def _interval(statistic, size: int, groups, seed: int, resamples: int, confidence: float):
    generator = np.random.default_rng(seed)
    drawn = [value for value in (statistic(_resample(generator, size, groups))
                                 for _ in range(resamples)) if not np.isnan(value)]
    if not drawn:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0

    return float(np.quantile(drawn, tail)), float(np.quantile(drawn, 1.0 - tail))


def bootstrap_auc(scores, labels, groups=None, seed: int = 0,
                  resamples: int = RESAMPLES, confidence: float = CONFIDENCE) -> dict:
    """The AUC with a percentile interval over resampled patients."""
    def statistic(at):
        return grouped_auc(scores[at], labels[at], None if groups is None else groups[at])

    low, high = _interval(statistic, len(labels), groups, seed, resamples, confidence)

    return {"auc": grouped_auc(scores, labels, groups), "low": low, "high": high}


def paired_delta(first, second, labels, groups=None, seed: int = 0,
                 resamples: int = RESAMPLES, confidence: float = CONFIDENCE) -> dict:
    """AUC of `first` minus AUC of `second`, scored on the same patients in the same order.

    **Paired**: each resample draws patients once and recomputes both AUCs on that
    draw, so the patient-to-patient difficulty common to both arms cancels and
    the interval measures the difference, not the sum of two uncertainties.
    """
    def statistic(at):
        within = None if groups is None else groups[at]
        return (grouped_auc(first[at], labels[at], within)
                - grouped_auc(second[at], labels[at], within))

    low, high = _interval(statistic, len(labels), groups, seed, resamples, confidence)

    return {"delta": statistic(np.arange(len(labels))), "low": low, "high": high,
            "excludes_zero": bool(low > 0 or high < 0)}


def paired_runs(one: dict, two: dict, groups_of=None, seed: int = 0) -> dict:
    """The paired difference of two scored sets, over the patients both scored.

    A scored set holds `scored`: patient key -> [logit, label], so arms that
    dropped different patients are still lined up woman by woman. None when the
    two share no patient.
    """
    left, right = one["scored"], two["scored"]
    shared = sorted(set(left) & set(right))
    if not shared:
        return None

    delta = paired_delta(np.array([left[key][0] for key in shared]),
                         np.array([right[key][0] for key in shared]),
                         np.array([left[key][1] for key in shared]),
                         groups=None if groups_of is None else groups_of(shared), seed=seed)

    return {**delta, "patients": len(shared)}


def scored_set(logits: np.ndarray, labels: np.ndarray, keys: list, groups=None,
               seed: int = 0) -> dict:
    """What a run records for one set: the AUC, its interval, and every patient's score.

    The per-patient scores are kept so any later paired comparison needs no
    rerun. Accuracy and recall are at the logit's own threshold of zero.
    """
    interval = bootstrap_auc(logits, labels, groups, seed)
    predicted = (logits > 0).astype(int)

    def recall(label):
        mine = labels == label
        return float((predicted[mine] == label).mean()) if mine.any() else float("nan")

    return {
        "auc": interval["auc"], "auc_low": interval["low"], "auc_high": interval["high"],
        "accuracy": float((predicted == labels).mean()),
        "sensitivity": recall(1), "specificity": recall(0),
        "patients": int(len(labels)), "malignant": int(labels.sum()),
        "scored": {key: [float(logit), int(label)]
                   for key, logit, label in zip(keys, logits, labels)},
    }
