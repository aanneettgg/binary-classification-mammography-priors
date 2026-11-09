"""Everything one run reads - cohort, split, cache, arm - assembled from its config.

Training and the rescoring under negative controls both rebuild a run's sets
from here, so a rescore reads exactly the patients and patches the run did.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from mammo_priors.config import ComponentConfig, RunConfig
from mammo_priors.data.balance import balance_named
from mammo_priors.data.cohort import Cohort
from mammo_priors.data.splits import TEST, TRAIN, VALIDATION, SplitFile
from mammo_priors.datasets.arms import ARMS
from mammo_priors.datasets.grid_dataset import GridDataset, collate, sliced
from mammo_priors.datasets.preprocessing import Augment, GridPreprocess, dihedral_views
from mammo_priors.evaluation.metrics import scored_set
from mammo_priors.patches.cache import PatchCache
from mammo_priors.patches.extract import patch_config


def run_config(workspace, split: SplitFile, arm: str, encoder: str, aggregator: str,
               size_px: int, correspondence: str, **settings) -> RunConfig:
    """The config of one run on one split; `settings` are any other RunConfig fields."""
    cohort = Cohort.load(workspace.cohort)

    return RunConfig(arm=arm, encoder=ComponentConfig(encoder),
                     aggregator=ComponentConfig(aggregator),
                     patches=patch_config(cohort, correspondence), size_px=size_px,
                     split=split.name, split_digest=split.digest, balance=split.balance,
                     **settings)


class Experiment:
    """The sets of one run, built the same way for training and for rescoring."""

    def __init__(self, workspace, run: RunConfig, split_path: str):
        self.workspace = workspace
        self.run = run
        self.split = SplitFile(split_path)
        if self.split.digest != run.split_digest:
            raise ValueError(f"{split_path} is not the split the run recorded")
        self.cohort = Cohort.load(workspace.cohort)
        if self.cohort.revision() != run.patches.cohort_revision:
            raise ValueError("the cohort changed since this run's cache was cut; "
                             "re-run extraction and training")
        self.cache = PatchCache(workspace.cache, run.patches)
        self.arm = ARMS[run.arm].through(run.view)
        self.balance = balance_named(run.balance)
        self._check_margins()

    def _check_margins(self):
        slack = (self.run.patches.margin_px(prior=False) - self.run.size_px) // 2
        if not 0 <= self.run.augment.shift_px <= slack:
            raise ValueError(f"a {self.run.size_px} px window has {slack} px of margin "
                             f"a side; shift_px {self.run.augment.shift_px} does not fit")

    def dataset(self, samples: list, generator: torch.Generator = None,
                control=None) -> GridDataset:
        """Augmented when given a generator, which only the training set is."""
        augment = None if generator is None else Augment(self.run.augment, generator)
        return GridDataset(samples, self.cache, self.arm, self.run.size_px, augment=augment,
                           preprocess=GridPreprocess(self.run.preprocess),
                           interval=self.run.interval, control=control)

    def samples(self, name: str) -> list:
        return self.split.samples_of(self.cohort, name)

    def external(self, source: str) -> list:
        """One external source's patients, every one of them, scored on its own."""
        return self.cohort.of_sources([source])

    def check_independent(self):
        sets = {name: {sample.key for sample in self.samples(name)}
                for name in (TRAIN, VALIDATION, TEST)}
        for first, second in ((TRAIN, VALIDATION), (TRAIN, TEST), (VALIDATION, TEST)):
            shared = sets[first] & sets[second]
            if shared:
                raise AssertionError(f"{len(shared)} patients are in both {first} and "
                                     f"{second}; every number would be a leak")


class EpochSampler(Sampler):
    """The patients of each epoch as the split's balance draws them, afresh each epoch."""

    def __init__(self, samples: list, balance, seed: int):
        self.samples = samples
        self.balance = balance
        self.generator = np.random.default_rng(seed)
        self.size = len(balance.draw(samples, np.random.default_rng(seed)))

    def __iter__(self):
        return iter(self.balance.draw(self.samples, self.generator).tolist())

    def __len__(self):
        return self.size


def _seed_worker(worker_id: int):
    """Re-seed each worker's copy of the augmentation generator.

    A generator is copied into every worker, so without this they would all
    draw the same flips in the same order.
    """
    info = torch.utils.data.get_worker_info()
    augment = getattr(info.dataset, "augment", None) if info else None
    if augment is not None:
        augment.generator.manual_seed(info.seed % (2 ** 63))


def loader(dataset: GridDataset, batch: int, workers: int, shuffle: bool = False,
           sampler: Sampler = None) -> DataLoader:
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=workers, collate_fn=collate,
                      persistent_workers=workers > 0, worker_init_fn=_seed_worker)


def forward(model, batch: dict, device) -> torch.Tensor:
    return model(batch["patches"].to(device), batch["present"].to(device),
                 batch["gap_months"].to(device))


def score(model, data: DataLoader, run: RunConfig, device, balance, seed: int = 0) -> dict:
    """One set scored with the model as it stands: AUC, interval, every patient.

    With test-time augmentation a patient's logit is the mean over the eight
    orientations of her grid. Sliced like training, so a batch that fits there
    fits here; in eval mode BatchNorm reads running statistics, so slicing does
    not change a logit.
    """
    model.eval()
    logits, labels, keys = [], [], []
    with torch.no_grad():
        for batch in data:
            for part in sliced(batch, run.max_cells):
                if run.test_time_augment:
                    views = [forward(model, {**part, "patches": view}, device)
                             for view in dihedral_views(part["patches"])]
                    logit = torch.stack(views).mean(dim=0)
                else:
                    logit = forward(model, part, device)
                logits.append(logit.float().cpu().numpy())
                labels.append(part["label"].numpy())
                keys += part["key"]

    return scored_set(np.concatenate(logits), np.concatenate(labels).astype(int), keys,
                      groups=balance.groups(keys), seed=seed)
