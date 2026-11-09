"""Fast checks of the pieces every arm difference depends on."""

import numpy as np
import pytest
import torch

from mammo_priors.config import ComponentConfig
from mammo_priors.data.cohort import Exam, Finding, Sample
from mammo_priors.data.splits import FOLDS, TEST, Split
from mammo_priors.datasets.arms import ARMS
from mammo_priors.datasets.gaps import MONTHS, NOMINAL_ROUND_MONTHS, ROUNDS, gaps_between
from mammo_priors.evaluation.metrics import auc, paired_delta
from mammo_priors.models.aggregator import AGGREGATORS
from mammo_priors.patches.geometry import tile


def test_tile_covers_a_large_box_and_stays_inside():
    windows = tile((100, 100, 1300, 400), (1500, 1000), 512)
    assert len(windows) == 3
    for window in windows:
        y0, x0, y1, x1 = window.bounds_yx
        assert y0 >= 0 and x0 >= 0 and y1 <= 1500 and x1 <= 1000
    assert min(w.bounds_yx[0] for w in windows) <= 100
    assert max(w.bounds_yx[2] for w in windows) >= 1300


def test_arms_are_masks():
    present = np.ones((3, 2, 2), dtype=bool)
    present[0, 0] = False                       # index exam without CC
    assert ARMS["A1"].mask(present)[1:].sum() == 0
    assert ARMS["A2"].mask(present)[2:].sum() == 0
    assert ARMS["A3"].mask(present).sum() == present.sum()
    a0 = ARMS["A0"].mask(present)
    assert a0.sum() == 2 and a0[0, 1].all()     # the first view she has: MLO
    assert ARMS["A3"].through("CC").mask(present)[:, 1].sum() == 0


def test_gaps_skip_absent_exams():
    dates = ["2024-01-01", "2022-01-01", "2020-01-01"]
    present = np.array([True, False, True]).reshape(3, 1, 1)
    rounds = gaps_between(dates, present, ROUNDS)
    assert rounds.tolist() == [NOMINAL_ROUND_MONTHS, 0, 0]
    months = gaps_between(dates, present, MONTHS)
    assert 47 < months[0] < 49


@pytest.mark.parametrize("name", list(AGGREGATORS))
def test_aggregator_reads_only_the_mask(name):
    torch.manual_seed(0)
    model = AGGREGATORS.build(ComponentConfig(name), dimension=16).eval()
    features = torch.randn(4, 3, 2, 2, 16)
    present = torch.rand(4, 3, 2, 2) > 0.3
    present[:, 0, 0, 0] = True
    gaps = torch.full((4, 3), 24.0)
    with torch.no_grad():
        base = model(features, present, gaps)
        assert base.shape == (4,)
        poisoned = torch.where(present.unsqueeze(-1), features, torch.full_like(features, 1e3))
        assert torch.allclose(base, model(poisoned, present, gaps), atol=1e-5)
        one = torch.zeros_like(present)
        one[:, 0] = present[:, 0]
        assert model(features, one, gaps).shape == (4,)


def test_auc_and_paired_delta():
    labels = np.array([0, 0, 1, 1])
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert auc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5
    scores = np.random.default_rng(0).normal(size=200)
    labels = (np.random.default_rng(1).random(200) < 0.5).astype(int)
    same = paired_delta(scores, scores, labels, resamples=200)
    assert same["delta"] == 0 and not same["excludes_zero"]


def _samples(count: int) -> list:
    generator = np.random.default_rng(0)
    return [Sample(f"P{at}", "s", int(generator.random() < 0.5),
                   [Exam(f"202{4 - t}-01-01") for t in range(1 + int(generator.integers(1, 4)))],
                   [Finding("L", "CC", (0, 0, 10, 10), 0)])
            for at in range(count)]


def test_split_is_deterministic_disjoint_and_folds_rotate_the_test_set():
    samples = _samples(200)
    first = Split("x", ["s"]).assign(samples)
    assert first == Split("x", ["s"]).assign(list(reversed(samples)))
    keys = [set(first[name]) for name in first]
    assert sum(map(len, keys)) == 200 and not set.intersection(*keys)

    tested = [set(Split("x", ["s"], fold=fold).assign(samples)[TEST])
              for fold in range(1, FOLDS + 1)]
    assert set.union(*tested) == {sample.key for sample in samples}
    assert sum(map(len, tested)) == 200
