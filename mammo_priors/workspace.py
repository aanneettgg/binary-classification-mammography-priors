"""Where every step reads and writes, decided in one place.

Each command takes one `--work` folder instead of a path per artefact, so the
steps cannot disagree about where the cohort, a split, the cache or a run is.
"""

import os
import re

COHORT_JSON = "cohort.json"
METRICS_JSON = "metrics.json"
CONTROLS_JSON = "controls.json"
BEST_PT = "best.pt"


class Workspace:
    """The folder layout of one study: cohort, splits, patch cache and runs."""

    def __init__(self, root: str):
        self.root = root

    @property
    def cohort(self) -> str:
        return os.path.join(self.root, COHORT_JSON)

    def split(self, name: str, fold: int = 1) -> str:
        return os.path.join(self.root, "splits", f"fold{fold}", f"{name}.json")

    @property
    def cache(self) -> str:
        return os.path.join(self.root, "cache")

    @property
    def runs(self) -> str:
        return os.path.join(self.root, "runs")

    def run(self, digest: str) -> str:
        return os.path.join(self.runs, digest)

    @property
    def reports(self) -> str:
        return os.path.join(self.root, "reports")


def safe_name(text: str) -> str:
    """A file-system-safe rendering of an identifier, for one file per patient."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(text))
