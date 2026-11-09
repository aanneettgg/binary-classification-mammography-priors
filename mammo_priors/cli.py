"""The command line: one subcommand per step, all reading and writing one `--work` folder.

    mammo-priors cohort   --work W --images images.csv --findings findings.csv
    mammo-priors split    --work W --name main [--external SOURCE ...] [--pooled] [--fold k]
    mammo-priors extract  --work W --image-root DIR [--correspondence same-location]
    mammo-priors probe    --work W --split main
    mammo-priors train    --work W --split main --arm A3 --encoder resnet18-imagenet ...
    mammo-priors ladder   --work W --split main
    mammo-priors controls --work W --split main
    mammo-priors report   --work W
"""

import argparse
import glob
import os

import pandas as pd

from mammo_priors.config import AugmentConfig, RunConfig, TrainingConfig
from mammo_priors.data.cohort import Cohort, CohortBuilder
from mammo_priors.data.splits import FOLDS, Split, SplitFile
from mammo_priors.data.tables import VIEWS, read_findings, read_images
from mammo_priors.datasets.arms import ARMS
from mammo_priors.datasets.gaps import INTERVALS, ROUNDS
from mammo_priors.models.aggregator import AGGREGATORS, MASKED_MEAN
from mammo_priors.models.encoder import ENCODERS
from mammo_priors.patches.correspondence import CORRESPONDENCES, SAME_LOCATION
from mammo_priors.workspace import Workspace


def _cohort(arguments):
    builder = CohortBuilder(read_images(arguments.images), read_findings(arguments.findings),
                            min_priors=arguments.min_priors)
    cohort = builder.build()
    os.makedirs(arguments.work, exist_ok=True)
    cohort.save(Workspace(arguments.work).cohort)
    print(cohort.describe())


def _split(arguments):
    workspace = Workspace(arguments.work)
    cohort = Cohort.load(workspace.cohort)
    sources = arguments.sources or [source for source in cohort.sources
                                    if source not in arguments.external]
    Split(arguments.name, sources, arguments.external, seed=arguments.seed,
          fold=arguments.fold, pooled=arguments.pooled).save(
        cohort, workspace.split(arguments.name, arguments.fold))


def _extract(arguments):
    from mammo_priors.patches.extract import extract

    workspace = Workspace(arguments.work)
    extract(Cohort.load(workspace.cohort), workspace.cache, arguments.image_root,
            arguments.correspondence)


def _probe(arguments):
    from mammo_priors.evaluation.probe import probe

    workspace = Workspace(arguments.work)
    cohort = Cohort.load(workspace.cohort)
    split = SplitFile(workspace.split(arguments.split, arguments.fold))
    rows = [row for arm in arguments.arms
            for row in probe(cohort, split, ARMS[arm].through(arguments.view), arguments.interval)]
    print("Availability probe: patient AUC from which exams exist and their gaps alone.")
    print("Chance is 0.5; above it, the arm can read the label without a pixel.\n")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def _settings(arguments) -> dict:
    """The RunConfig fields the training options set, shared by `train` and `ladder`."""
    return {
        "seed": arguments.seed, "view": arguments.view, "interval": arguments.interval,
        "max_cells": arguments.max_cells, "checkpoint_oversized": arguments.checkpoint_oversized,
        "test_time_augment": arguments.test_time_augment,
        "augment": AugmentConfig(shift_px=arguments.shift_px),
        "training": TrainingConfig(epochs=arguments.epochs, patience=arguments.patience,
                                   batch=arguments.batch),
    }


def _trainer(arguments):
    from mammo_priors.training.trainer import Trainer

    return Trainer(Workspace(arguments.work), device=arguments.device,
                   workers=arguments.workers, keep_weights=arguments.keep_weights)


def _train(arguments):
    from mammo_priors.training.experiment import run_config

    workspace = Workspace(arguments.work)
    split_path = workspace.split(arguments.split, arguments.fold)
    run = run_config(workspace, SplitFile(split_path), arguments.arm, arguments.encoder,
                     arguments.aggregator, arguments.size, arguments.correspondence,
                     **_settings(arguments))
    _trainer(arguments).run(run, split_path)


def _ladder(arguments):
    from mammo_priors.training.ladder import LadderGrid, run_ladder

    workspace = Workspace(arguments.work)
    grid = LadderGrid(arms=tuple(arguments.arms), encoders=tuple(arguments.encoders),
                      aggregators=tuple(arguments.aggregators),
                      a1_aggregators=tuple(arguments.a1_aggregators),
                      sizes=tuple(arguments.sizes), correspondence=arguments.correspondence,
                      settings=_settings(arguments))
    run_ladder(workspace, _trainer(arguments),
               workspace.split(arguments.split, arguments.fold), grid)


def _controls(arguments):
    from mammo_priors.evaluation.controls import score_controls

    workspace = Workspace(arguments.work)
    split_path = workspace.split(arguments.split, arguments.fold)
    digest = SplitFile(split_path).digest
    for folder in sorted(glob.glob(os.path.join(workspace.runs, "*"))):
        if not os.path.exists(os.path.join(folder, "best.pt")):
            continue
        run = RunConfig.read(folder)
        if run.split_digest != digest or not ARMS[run.arm].sees_priors:
            continue
        print(f"=== controls for {run}")
        for control, sets in score_controls(workspace, folder, split_path, arguments.device,
                                            arguments.workers).items():
            for name, result in sets.items():
                print(f"  {control:17s} {name:10s} plain {result['plain_auc']:.4f} "
                      f"control {result['control_auc']:.4f} delta {result['delta']:+.4f} "
                      f"changed {result['changed']}/{result['patients']}")


def _report(arguments):
    from mammo_priors.evaluation.report import write_report

    workspace = Workspace(arguments.work)
    print(f"report -> {write_report(workspace.runs, arguments.out or workspace.reports)}")


def _training_options(parser):
    parser.add_argument("--split", required=True, help="the split's name")
    parser.add_argument("--fold", type=int, default=1, choices=range(1, FOLDS + 1))
    parser.add_argument("--correspondence", default=SAME_LOCATION,
                        choices=sorted(CORRESPONDENCES.classes),
                        help="which cache to read: how its priors were placed")
    parser.add_argument("--view", choices=VIEWS, default=None,
                        help="open only this view of every exam (default: both)")
    parser.add_argument("--interval", choices=INTERVALS, default=ROUNDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    parser.add_argument("--batch", type=int, default=TrainingConfig.batch,
                        help="patients per batch")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="most patches per forward pass; larger batches run in "
                             "slices with gradients accumulated")
    parser.add_argument("--checkpoint-oversized", action="store_true",
                        help="encode a patient holding more than --max-cells patches in "
                             "checkpointed chunks (for grids that do not fit even alone)")
    parser.add_argument("--shift-px", type=int, default=0,
                        help="training-time jitter of the window, in pixels")
    parser.add_argument("--test-time-augment", action="store_true",
                        help="score each patient as the mean over the 8 flips/quarter turns")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep-weights", action="store_true",
                        help="write best.pt, needed to score the negative controls later")


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(prog="mammo-priors", description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = main.add_subparsers(dest="command", required=True)

    def command(name, handler, help_text):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--work", required=True, help="the study's working folder")
        sub.set_defaults(handler=handler)
        return sub

    sub = command("cohort", _cohort, "build the cohort from images.csv and findings.csv")
    sub.add_argument("--images", required=True)
    sub.add_argument("--findings", required=True)
    sub.add_argument("--min-priors", type=int, default=1)

    sub = command("split", _split, "freeze a train/validation/test split")
    sub.add_argument("--name", required=True)
    sub.add_argument("--sources", nargs="*", default=None,
                     help="sources to divide (default: all but the external ones)")
    sub.add_argument("--external", nargs="*", default=[],
                     help="sources never trained on, scored once after selection")
    sub.add_argument("--pooled", action="store_true",
                     help="stratify by source and train with each source's classes "
                          "drawn equal every epoch")
    sub.add_argument("--fold", type=int, default=1, choices=range(1, FOLDS + 1))
    sub.add_argument("--seed", type=int, default=0)

    sub = command("extract", _extract, "cut the patch cache")
    sub.add_argument("--image-root", default="", help="folder the image paths are relative to")
    sub.add_argument("--correspondence", default=SAME_LOCATION,
                     choices=sorted(CORRESPONDENCES.classes))

    sub = command("probe", _probe, "availability probe: the label from mask and gaps alone")
    sub.add_argument("--split", required=True)
    sub.add_argument("--fold", type=int, default=1, choices=range(1, FOLDS + 1))
    sub.add_argument("--arms", nargs="+", default=["A2", "A3"], choices=sorted(ARMS))
    sub.add_argument("--view", choices=VIEWS, default=None)
    sub.add_argument("--interval", choices=INTERVALS, default=ROUNDS)

    sub = command("train", _train, "train one run")
    _training_options(sub)
    sub.add_argument("--arm", required=True, choices=sorted(ARMS))
    sub.add_argument("--encoder", default="resnet18-imagenet", choices=sorted(ENCODERS.classes))
    sub.add_argument("--aggregator", default=MASKED_MEAN, choices=sorted(AGGREGATORS.classes))
    sub.add_argument("--size", type=int, default=512, help="model window, pixels (<= 512)")

    sub = command("ladder", _ladder, "train every arm x encoder x aggregator x size")
    _training_options(sub)
    sub.add_argument("--arms", nargs="+", default=list(ARMS), choices=sorted(ARMS))
    sub.add_argument("--encoders", nargs="+", choices=sorted(ENCODERS.classes),
                     default=["resnet18-imagenet", "resnet34-imagenet", "resnet50-imagenet"])
    sub.add_argument("--aggregators", nargs="+", choices=sorted(AGGREGATORS.classes),
                     default=list(AGGREGATORS), help="for A2 and A3")
    sub.add_argument("--a1-aggregators", nargs="+", choices=sorted(AGGREGATORS.classes),
                     default=[MASKED_MEAN])
    sub.add_argument("--sizes", nargs="+", type=int, default=[224, 512])

    sub = command("controls", _controls,
                  "score the negative controls on every with-prior run that kept its weights")
    sub.add_argument("--split", required=True)
    sub.add_argument("--fold", type=int, default=1, choices=range(1, FOLDS + 1))
    sub.add_argument("--device", default="cuda")
    sub.add_argument("--workers", type=int, default=4)

    sub = command("report", _report, "tables of every run and every paired arm difference")
    sub.add_argument("--out", default=None, help="default: <work>/reports")

    return main


def main(argv=None):
    arguments = parser().parse_args(argv)
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
