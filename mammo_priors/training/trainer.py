"""Train one arm and write down what it was and what it scored.

Every arm runs this same loop. The run folder is named by the config's digest,
so a finished run (one with `metrics.json`) is recognised and skipped, and two
runs that differ in any setting never overwrite each other.
"""

import json
import os
import time

import torch
from torch import nn

from mammo_priors.data.splits import TEST, TRAIN, VALIDATION
from mammo_priors.datasets.grid_dataset import sliced
from mammo_priors.models.model import build
from mammo_priors.training.experiment import (EpochSampler, Experiment, forward, loader,
                                              score)
from mammo_priors.workspace import BEST_PT, METRICS_JSON


def finished(workspace, run) -> bool:
    return os.path.exists(os.path.join(workspace.run(run.digest()), METRICS_JSON))


class Trainer:
    """Fits one run, selects its epoch on validation AUC, scores test and externals."""

    def __init__(self, workspace, device: str = "cuda", workers: int = 4,
                 keep_weights: bool = False):
        self.workspace = workspace
        self.device = torch.device(device)
        self.workers = workers
        self.keep_weights = keep_weights

    def _criterion(self, experiment: Experiment, samples: list) -> nn.Module:
        """Binary cross-entropy, the positive class weighted by negatives over positives.

        Counted as training draws the patients, so a split balanced by its draw
        is not balanced a second time by the loss.
        """
        weights = experiment.balance.weights(samples)
        labels = torch.tensor([sample.label for sample in samples], dtype=torch.float64)
        drawn = torch.from_numpy(weights)
        positives, negatives = float((drawn * labels).sum()), float((drawn * (1 - labels)).sum())
        if not positives or not negatives:
            raise ValueError("the training set holds only one class")

        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives,
                                                            device=self.device))

    def train(self, experiment: Experiment) -> dict:
        run, schedule = experiment.run, experiment.run.training
        experiment.check_independent()
        generator = torch.Generator().manual_seed(run.seed)
        training = experiment.dataset(experiment.samples(TRAIN), generator=generator)
        validation = experiment.dataset(experiment.samples(VALIDATION))
        sampler = (EpochSampler(training.samples, experiment.balance, run.seed)
                   if experiment.balance.resamples else None)
        batches = loader(training, schedule.batch, self.workers, shuffle=True, sampler=sampler)
        print(f"  train {len(training)}, validation {len(validation)} patients")

        torch.manual_seed(run.seed)
        model = build(run).to(self.device)
        criterion = self._criterion(experiment, training.samples)
        optimiser = torch.optim.AdamW(model.groups(schedule.backbone_lr, schedule.head_lr),
                                      weight_decay=schedule.weight_decay)
        steps = max(1, len(batches))
        cycle = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=[schedule.backbone_lr, schedule.head_lr],
            total_steps=schedule.epochs * steps,
            pct_start=min(0.99, schedule.warmup_epochs / schedule.epochs), anneal_strategy="cos")

        best, best_epoch, best_state, history = None, -1, None, []
        for epoch in range(schedule.epochs):
            model.train()
            started, total, seen = time.time(), 0.0, 0
            for batch in batches:
                optimiser.zero_grad(set_to_none=True)
                patients = len(batch["label"])
                for part in sliced(batch, run.max_cells):
                    labels = part["label"].to(self.device)
                    loss = criterion(forward(model, part, self.device), labels)
                    # Weighted by its share, so the slices' gradients sum to the
                    # gradient of the whole batch's mean loss.
                    (loss * len(labels) / patients).backward()
                    total += float(loss.item()) * len(labels)
                    seen += len(labels)
                optimiser.step()
                cycle.step()

            scored = score(model, loader(validation, schedule.batch, self.workers), run,
                           self.device, experiment.balance, run.seed)
            history.append({"epoch": epoch, "loss": total / max(1, seen),
                            **{key: value for key, value in scored.items() if key != "scored"}})
            print(f"  epoch {epoch:2d}  loss {history[-1]['loss']:.4f}  "
                  f"validation AUC {scored['auc']:.4f}  ({time.time() - started:.0f} s)")

            # Selected on validation AUC, not loss: the loss is class-weighted
            # and the result is reported as a ranking.
            if best is None or scored["auc"] > best["auc"]:
                best, best_epoch = scored, epoch
                best_state = {name: tensor.detach().to("cpu", copy=True)
                              for name, tensor in model.state_dict().items()}
            elif epoch - best_epoch >= schedule.patience:
                print(f"  no improvement in {schedule.patience} epochs, stopping")
                break

        model.load_state_dict(best_state)
        folder = self.workspace.run(run.digest())
        if self.keep_weights:
            torch.save(best_state, os.path.join(folder, BEST_PT))

        results = {"best_epoch": best_epoch, "validation": best, "history": history,
                   "test": score(model, loader(experiment.dataset(experiment.samples(TEST)),
                                               schedule.batch, self.workers),
                                 run, self.device, experiment.balance, run.seed),
                   "external": {}}
        for source in experiment.split.external:
            data = experiment.dataset(experiment.external(source))
            results["external"][source] = score(
                model, loader(data, schedule.batch, self.workers), run, self.device,
                experiment.balance, run.seed)

        return results

    def run(self, run, split_path: str) -> dict:
        """Train `run` unless it already finished; returns its metrics."""
        folder = self.workspace.run(run.digest())
        path = os.path.join(folder, METRICS_JSON)
        if os.path.exists(path):
            print(f"  {run} already finished in {folder}")
            with open(path) as metrics_file:
                return json.load(metrics_file)

        print(f"=== {run}  ->  {folder}")
        experiment = Experiment(self.workspace, run, split_path)
        run.save(folder)
        results = self.train(experiment)
        with open(path, "w") as metrics_file:
            json.dump(results, metrics_file, indent=1)

        test = results["test"]
        print(f"  test AUC {test['auc']:.4f} [{test['auc_low']:.3f}, {test['auc_high']:.3f}] "
              f"n={test['patients']}")
        for source, scored in results["external"].items():
            print(f"  external {source} AUC {scored['auc']:.4f} n={scored['patients']}")

        return results
