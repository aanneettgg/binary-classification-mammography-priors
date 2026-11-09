"""Which patients train, which validate and which test - written down once, then frozen.

A split is a file, not a function. One regenerated at run time from a ratio and
a seed stops referring to the experiment that produced a result the moment the
cohort gains a patient, so the assignment is written out with a digest of itself,
every run records that digest, and an existing split file is never overwritten
with a different assignment.

Patients are ordered by a hash of their id and the seed, so the split does not
depend on the row order of any table, and they are stratified by label and by
how many priors they own: an arm that sees the priors must not be compared on
sets that hold different histories.
"""

import hashlib
import json
import os

import numpy as np

from mammo_priors.data.balance import NONE, SOURCE_LABEL

TRAIN, VALIDATION, TEST = "train", "validation", "test"
RATIOS = {TRAIN: 0.6, VALIDATION: 0.2, TEST: 0.2}
# Each fold rotates the sets one fifth along the hashed order, so across the
# five folds every patient is tested once.
FOLDS = 5
# 1, 2 and 3 or more priors.
PRIOR_BUCKETS = 3


class SplitFile:
    """A frozen split as read back: who is where, and how it is to be trained."""

    def __init__(self, path: str):
        with open(path) as split_file:
            self.record = json.load(split_file)
        self.path = path

    @property
    def name(self) -> str:
        return self.record["name"]

    @property
    def digest(self) -> str:
        return self.record["digest"]

    @property
    def balance(self) -> str:
        return self.record["balance"]

    @property
    def external(self) -> list:
        return self.record["external"]

    def keys_of(self, name: str) -> set:
        return set(self.record["patients"][name])

    def samples_of(self, cohort, name: str) -> list:
        keys = self.keys_of(name)
        return [sample for sample in cohort.samples if sample.key in keys]


class Split:
    """Train, validation and test over `sources`, scored afterwards on `external`.

    `pooled` stratifies by source as well, so each source is divided as its own
    split would divide it, and trains with the source-label balance.
    """

    def __init__(self, name: str, sources: list, external: list = (), seed: int = 0,
                 fold: int = 1, pooled: bool = False):
        overlap = set(sources) & set(external)
        if overlap:
            raise ValueError(f"split {name!r} would train and test on {sorted(overlap)}")
        if fold not in range(1, FOLDS + 1):
            raise ValueError(f"fold must be 1 to {FOLDS}, not {fold}")

        self.name = name
        self.sources = sorted(sources)
        self.external = sorted(external)
        self.seed = seed
        self.fold = fold
        self.pooled = pooled

    def _stratum(self, sample) -> str:
        stratum = f"{sample.label}/{min(sample.priors, PRIOR_BUCKETS)}"
        return f"{sample.source}/{stratum}" if self.pooled else stratum

    def _order(self, sample) -> str:
        return hashlib.sha256(f"{self.seed}/{sample.key}".encode()).hexdigest()

    @staticmethod
    def _slots(size: int) -> list:
        """A stratum of `size` cut into five fold-sized slots named by their fold-1 set."""
        counts = {TRAIN: round(size * RATIOS[TRAIN]),
                  VALIDATION: round(size * RATIOS[VALIDATION])}
        counts[TEST] = size - counts[TRAIN] - counts[VALIDATION]

        slots = []
        for name, count in counts.items():
            fifths = round(RATIOS[name] * FOLDS)
            edges = [round(count * at / fifths) for at in range(fifths + 1)]
            slots += [(name, end - start) for start, end in zip(edges, edges[1:])]

        return slots

    def assign(self, samples: list) -> dict:
        """Set name -> sorted patient keys, for the patients of this split's sources."""
        strata = {}
        for sample in samples:
            if sample.source in self.sources:
                strata.setdefault(self._stratum(sample), []).append(sample)

        sets = {TRAIN: [], VALIDATION: [], TEST: []}
        shift = self.fold - 1
        for members in strata.values():
            ordered = sorted(members, key=self._order)
            slots = self._slots(len(ordered))
            names = [name for name, _ in slots]
            rotated = names[shift:] + names[:shift]
            at = 0
            for name, (_, count) in zip(rotated, slots):
                sets[name] += [sample.key for sample in ordered[at:at + count]]
                at += count

        return {name: sorted(keys) for name, keys in sets.items()}

    def save(self, cohort, path: str) -> str:
        """Write the split; an existing file must hold the same assignment."""
        patients = self.assign(cohort.samples)
        digest = hashlib.sha256(json.dumps(patients, sort_keys=True).encode()).hexdigest()
        if os.path.exists(path):
            if SplitFile(path).digest == digest:
                print(f"{path} already holds this split")
                return path
            raise FileExistsError(
                f"{path} holds a different split, and a frozen split is never "
                f"rewritten: results that named it refer to the one on disk. "
                f"Choose another name, or delete it if nothing was reported from it.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as split_file:
            json.dump({
                "name": self.name, "sources": self.sources, "external": self.external,
                "seed": self.seed, "fold": self.fold, "ratios": RATIOS,
                "balance": SOURCE_LABEL if self.pooled else NONE,
                "digest": digest, "patients": patients,
            }, split_file, indent=1, sort_keys=True)

        self.describe(cohort, patients)
        return path

    def describe(self, cohort, patients: dict):
        by_key = {sample.key: sample for sample in cohort.samples}
        print(f"split {self.name!r}, fold {self.fold}, sources {self.sources}, "
              f"external {self.external}")
        for name, keys in patients.items():
            labels = np.array([by_key[key].label for key in keys], dtype=int)
            print(f"  {name:10s} {len(keys):5d} patients  "
                  f"{int((labels == 0).sum())} benign : {int((labels == 1).sum())} malignant")
