"""Every finished run, and every arm difference the ladder asks, as tables.

Each with-prior arm is paired with its reference arm of the same configuration -
split, seed, encoder, window size, view, augmentation, and the same aggregator
when that reference was run with it, otherwise the masked-mean one (the column
`reference_aggregator` says which). Every difference is a paired bootstrap over
the patients both runs scored, on the test set and on each external source.

No configuration is singled out: the effect of the priors may depend on the
encoder, the window or the aggregator, so the whole grid is reported and summed
up by how many pairs lie above zero and how many intervals exclude it.
"""

import glob
import json
import os

import pandas as pd

from mammo_priors.config import CONFIG_JSON
from mammo_priors.data.balance import balance_named
from mammo_priors.datasets.arms import ARMS, COMPARISONS
from mammo_priors.evaluation.metrics import paired_runs
from mammo_priors.models.aggregator import MASKED_MEAN
from mammo_priors.workspace import CONTROLS_JSON, METRICS_JSON

NO_PRIOR = "-"


class Run:
    """One finished run folder: its config, its metrics, its controls if scored."""

    def __init__(self, folder: str):
        self.folder = folder
        with open(os.path.join(folder, CONFIG_JSON)) as config_file:
            self.config = json.load(config_file)
        with open(os.path.join(folder, METRICS_JSON)) as metrics_file:
            self.metrics = json.load(metrics_file)
        controls = os.path.join(folder, CONTROLS_JSON)
        self.controls = None
        if os.path.exists(controls):
            with open(controls) as controls_file:
                self.controls = json.load(controls_file)

    @property
    def arm(self) -> str:
        return self.config["arm"]

    @property
    def aggregator(self) -> str:
        return self.config["aggregator"]["name"]

    def configuration(self) -> dict:
        """Everything a pair is matched on: all but the arm and the aggregator.

        An arm that sees no prior reads the same index cells from every cache,
        so its placement method is recorded as "-" and it pairs with the
        with-prior arms of every method.
        """
        config = self.config
        sees_priors = ARMS[self.arm].sees_priors
        return {
            "split": config["split"], "split_digest": config["split_digest"][:12],
            "seed": config["seed"], "encoder": config["encoder"]["name"],
            "size_px": config["size_px"], "view": config["view"] or "both",
            "interval": config["interval"], "shift_px": config["augment"]["shift_px"],
            "tta": config["test_time_augment"],
            "correspondence": (config["patches"]["correspondence"]["name"]
                               if sees_priors else NO_PRIOR),
            "settings": json.dumps([config["preprocess"], config["training"]], sort_keys=True),
        }

    def sets(self) -> dict:
        """Name -> scored set: the test set, then each external source."""
        return {"test": self.metrics["test"], **self.metrics.get("external", {})}


def load_runs(runs_folder: str) -> list:
    return [Run(os.path.dirname(path))
            for path in sorted(glob.glob(os.path.join(runs_folder, "*", METRICS_JSON)))]


def _key(run: Run, aggregator: str = None) -> tuple:
    return (run.arm, aggregator or run.aggregator,
            tuple(sorted(run.configuration().items())))


def runs_table(runs: list) -> pd.DataFrame:
    rows = []
    for run in runs:
        for name, scored in run.sets().items():
            rows.append({"arm": run.arm, "aggregator": run.aggregator,
                         **{key: value for key, value in run.configuration().items()
                            if key != "settings"},
                         "set": name, "auc": scored["auc"], "low": scored["auc_low"],
                         "high": scored["auc_high"], "patients": scored["patients"],
                         "run": os.path.basename(run.folder)})

    return pd.DataFrame(rows)


def paired_table(runs: list, seed: int = 0) -> pd.DataFrame:
    by_key = {_key(run): run for run in runs}
    rows = []
    for arm, against in COMPARISONS:
        for run in (run for run in runs if run.arm == arm):
            configuration = run.configuration()
            if not ARMS[against].sees_priors:
                configuration["correspondence"] = NO_PRIOR
            reference = None
            for aggregator in (run.aggregator, MASKED_MEAN):
                reference = by_key.get((against, aggregator,
                                        tuple(sorted(configuration.items()))))
                if reference is not None:
                    break
            if reference is None:
                continue

            balance = balance_named(run.config["balance"])
            references = reference.sets()
            for name, scored in run.sets().items():
                if name not in references:
                    continue
                delta = paired_runs(scored, references[name], balance.groups, seed)
                if delta is None:
                    continue
                rows.append({"comparison": f"{arm} - {against}", "aggregator": run.aggregator,
                             "reference_aggregator": reference.aggregator,
                             **{key: value for key, value in run.configuration().items()
                                if key not in ("settings",)},
                             "set": name, **delta})

    return pd.DataFrame(rows)


def summary(paired: pd.DataFrame, by: list) -> pd.DataFrame:
    """Mean difference per group, and how many pairs (and intervals) point each way."""
    frame = paired.assign(above_zero=paired["delta"] > 0,
                          ci_up=paired["excludes_zero"] & (paired["delta"] > 0),
                          ci_down=paired["excludes_zero"] & (paired["delta"] < 0))

    return frame.groupby(by, sort=False).agg(
        pairs=("delta", "size"), mean_delta=("delta", "mean"),
        above_zero=("above_zero", "sum"), ci_up=("ci_up", "sum"),
        ci_down=("ci_down", "sum")).reset_index()


def controls_table(runs: list) -> pd.DataFrame:
    rows = []
    for run in runs:
        for control, sets in (run.controls or {}).items():
            for name, result in sets.items():
                rows.append({"control": control, "arm": run.arm, "aggregator": run.aggregator,
                             "encoder": run.config["encoder"]["name"],
                             "size_px": run.config["size_px"], "set": name, **result})

    return pd.DataFrame(rows)


def write_report(runs_folder: str, out_folder: str) -> str:
    runs = load_runs(runs_folder)
    if not runs:
        raise FileNotFoundError(f"no finished run under {runs_folder}")
    os.makedirs(out_folder, exist_ok=True)

    tables = {"runs": runs_table(runs), "paired": paired_table(runs),
              "controls": controls_table(runs)}
    for name, table in tables.items():
        if len(table):
            table.to_csv(os.path.join(out_folder, f"{name}.csv"), index=False)

    lines = [f"{len(runs)} finished runs", "",
             "PATIENT AUC PER RUN AND SET", tables["runs"].to_string(index=False), ""]
    paired = tables["paired"]
    if len(paired):
        lines += ["ARM DIFFERENCES, PAIRED (delta = AUC of the arm minus its reference)",
                  paired.drop(columns=["split_digest"]).to_string(index=False), "",
                  "SUMMARY BY COMPARISON AND SET",
                  summary(paired, ["comparison", "set"]).to_string(index=False), "",
                  "SUMMARY BY COMPARISON, AGGREGATOR AND SET",
                  summary(paired, ["comparison", "aggregator", "set"]).to_string(index=False), ""]
    if len(tables["controls"]):
        lines += ["NEGATIVE CONTROLS (delta = AUC under the control minus plain)",
                  tables["controls"].to_string(index=False), ""]

    path = os.path.join(out_folder, "report.txt")
    with open(path, "w", encoding="utf-8") as report:
        report.write("\n".join(lines))

    return path
