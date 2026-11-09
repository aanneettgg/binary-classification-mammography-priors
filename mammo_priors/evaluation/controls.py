"""Score finished runs again under the negative controls, from the weights they kept.

A control asks what a trained model reads, so it must be the model the ladder
reported: its own weights (`--keep-weights` at training), cache and patients.
Each run is first rescored plainly, and that score must reproduce the test AUC
it recorded; otherwise something differs - model, cache or patients - and no
control measured against it would mean anything.
"""

import json
import os

import torch

from mammo_priors.config import ComponentConfig, RunConfig
from mammo_priors.data.splits import TEST
from mammo_priors.datasets.controls import CONTROLS
from mammo_priors.evaluation.metrics import paired_runs
from mammo_priors.models.model import build
from mammo_priors.training.experiment import Experiment, loader, score
from mammo_priors.workspace import BEST_PT, CONTROLS_JSON, METRICS_JSON

REPRODUCED_WITHIN = 1e-3


def score_controls(workspace, folder: str, split_path: str, device: str = "cuda",
                   workers: int = 4) -> dict:
    run = RunConfig.read(folder)
    experiment = Experiment(workspace, run, split_path)
    if not experiment.arm.sees_priors:
        raise ValueError(f"{run.arm} sees no prior; the controls change nothing in it")
    weights = os.path.join(folder, BEST_PT)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"{weights} is missing; train with --keep-weights")

    model = build(run).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    with open(os.path.join(folder, METRICS_JSON)) as metrics_file:
        recorded = json.load(metrics_file)

    sets = {TEST: (experiment.samples(TEST), experiment.balance)}
    for source in experiment.split.external:
        sets[source] = (experiment.external(source), experiment.balance)

    def scored(samples, balance, control=None):
        data = loader(experiment.dataset(samples, control=control),
                      run.training.batch, workers)
        return score(model, data, run, device, balance, run.seed), data.dataset

    plain, _ = scored(*sets[TEST])
    if abs(plain["auc"] - recorded["test"]["auc"]) > REPRODUCED_WITHIN:
        raise ValueError(f"rescored test AUC {plain['auc']:.4f} does not reproduce the "
                         f"recorded {recorded['test']['auc']:.4f}")

    results = {}
    for name, (samples, balance) in sets.items():
        reference, _ = scored(samples, balance) if name != TEST else (plain, None)
        for control_name in CONTROLS:
            control = CONTROLS.build(ComponentConfig(control_name))
            controlled, dataset = scored(samples, balance, control)
            results.setdefault(control_name, {})[name] = {
                "plain_auc": reference["auc"], "control_auc": controlled["auc"],
                "changed": control.changed(dataset), "patients": len(dataset),
                **paired_runs(controlled, reference, balance.groups, run.seed),
            }

    with open(os.path.join(folder, CONTROLS_JSON), "w") as controls_file:
        json.dump(results, controls_file, indent=1)

    return results
