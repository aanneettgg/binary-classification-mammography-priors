"""The whole ladder: every arm, encoder, aggregator and window size, run in turn.

The rules of the grid live here so that an arm added later is run the way the
earlier ones were:

- **A2 and A3 run every aggregator.** Over several exams the aggregators read the
  history differently, and that is what is being compared.
- **A0 runs the masked mean only**, since A1 - A0 asks about the second view.
- **A1 runs the masked mean by default** (`a1_aggregators` adds others). Over one
  exam every aggregator reads the same pooled vector, so the masked-mean A1 is
  a common baseline; but a GRU or transformer is a different head over that
  vector, so running A1 under each lets every with-prior arm be paired with an
  A1 of its own aggregator, separating the input's effect from the head's.

Finished runs are skipped, so an interrupted ladder resumes where it stopped.
"""

from dataclasses import dataclass, field

from mammo_priors.data.splits import SplitFile
from mammo_priors.models.aggregator import AGGREGATORS, MASKED_MEAN
from mammo_priors.patches.correspondence import SAME_LOCATION
from mammo_priors.training.experiment import run_config


@dataclass
class LadderGrid:
    arms: tuple = ("A0", "A1", "A2", "A3")
    encoders: tuple = ("resnet18-imagenet", "resnet34-imagenet", "resnet50-imagenet")
    aggregators: tuple = tuple(AGGREGATORS)
    a1_aggregators: tuple = (MASKED_MEAN,)
    sizes: tuple = (224, 512)
    correspondence: str = SAME_LOCATION
    # Any other RunConfig field held fixed across the grid (seed, view, ...).
    settings: dict = field(default_factory=dict)

    def aggregators_of(self, arm: str) -> tuple:
        if arm == "A0":
            return (MASKED_MEAN,)
        if arm == "A1":
            return self.a1_aggregators
        return self.aggregators

    def runs(self, workspace, split: SplitFile) -> list:
        return [run_config(workspace, split, arm, encoder, aggregator, size,
                           self.correspondence, **self.settings)
                for size in self.sizes
                for arm in self.arms
                for aggregator in self.aggregators_of(arm)
                for encoder in self.encoders]


def run_ladder(workspace, trainer, split_path: str, grid: LadderGrid):
    runs = grid.runs(workspace, SplitFile(split_path))
    print(f"ladder: {len(runs)} runs")
    for at, run in enumerate(runs, start=1):
        print(f"\n[{at}/{len(runs)}]")
        trainer.run(run, split_path)
