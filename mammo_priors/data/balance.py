"""How a split's training set is balanced, and what that makes its AUC mean.

Sources pooled into one training set rarely share a class balance, and a model
can then score by recognising the source - its scanner, its screening interval,
how many priors its women own - instead of reading the finding. The
`source-label` balance removes that shortcut: every epoch draws each source's
classes in equal number, afresh, so no patient is thrown away and within an
epoch the source says nothing about the label.

The held-out sets are not balanced, so they are scored within each source and
the AUC is the mean of the per-source AUCs: a pooled AUC would reward knowing
the source.
"""

import numpy as np

NONE = "none"
SOURCE_LABEL = "source-label"


class Balance:
    """Every patient once per epoch, and one AUC over the whole set."""

    name = NONE
    resamples = False

    def weights(self, samples: list) -> np.ndarray:
        """How often each patient is drawn per epoch, on average."""
        return np.ones(len(samples))

    def draw(self, samples: list, generator: np.random.Generator) -> np.ndarray:
        return generator.permutation(len(samples))

    def groups(self, keys: list):
        """What an AUC is taken within and then averaged over; None for one AUC."""
        return None


class SourceLabelBalance(Balance):
    """Each source's two classes drawn equal, every epoch anew."""

    name = SOURCE_LABEL
    resamples = True

    @staticmethod
    def _cells(samples: list) -> tuple:
        cells = np.array([f"{sample.source}|{sample.label}" for sample in samples])
        sources = np.array([sample.source for sample in samples])
        size = {cell: int((cells == cell).sum()) for cell in np.unique(cells)}
        quota = {source: min(count for cell, count in size.items()
                             if cell.split("|")[0] == source)
                 for source in np.unique(sources)}

        return cells, sources, size, quota

    def weights(self, samples):
        cells, sources, size, quota = self._cells(samples)
        return np.array([quota[source] / size[cell] for cell, source in zip(cells, sources)])

    def draw(self, samples, generator):
        cells, sources, _, quota = self._cells(samples)
        chosen = [generator.choice(np.flatnonzero(cells == cell),
                                   quota[cell.split("|")[0]], replace=False)
                  for cell in np.unique(cells)]

        return generator.permutation(np.concatenate(chosen))

    def groups(self, keys):
        return np.array([key.split("/", 1)[0] for key in keys])


BALANCES = {balance.name: balance for balance in (Balance(), SourceLabelBalance())}


def balance_named(name: str) -> Balance:
    if name not in BALANCES:
        raise KeyError(f"no balance named {name!r}; choose one of {sorted(BALANCES)}")

    return BALANCES[name]
