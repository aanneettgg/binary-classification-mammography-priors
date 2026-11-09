"""The whole pipeline on a few synthetic patients, on the CPU, in about a minute.

Random-weight encoders and one epoch: this checks that every step runs and hands
the next what it expects, not that anything is learnt.
"""

import os
import subprocess
import sys

import pandas as pd

from mammo_priors.cli import main

EXAMPLE = os.path.join(os.path.dirname(__file__), os.pardir, "examples", "make_synthetic.py")
FAST = ["--epochs", "1", "--batch", "4", "--device", "cpu", "--workers", "0"]


def test_end_to_end(tmp_path):
    data, work = tmp_path / "data", tmp_path / "work"
    subprocess.run([sys.executable, EXAMPLE, "--out", str(data), "--patients", "16",
                    "--external-patients", "6"], check=True)

    main(["cohort", "--work", str(work), "--images", str(data / "images.csv"),
          "--findings", str(data / "findings.csv")])
    main(["split", "--work", str(work), "--name", "main", "--external", "siteB"])
    main(["extract", "--work", str(work), "--image-root", str(data)])
    main(["probe", "--work", str(work), "--split", "main"])
    main(["ladder", "--work", str(work), "--split", "main", "--encoders", "resnet18",
          "--aggregators", "masked-mean", "gru", "--sizes", "224", "--keep-weights", *FAST])
    main(["controls", "--work", str(work), "--split", "main", "--device", "cpu",
          "--workers", "0"])
    main(["report", "--work", str(work)])

    paired = pd.read_csv(work / "reports" / "paired.csv")
    assert set(paired["comparison"]) == {"A1 - A0", "A2 - A1", "A3 - A1", "A3 - A2"}
    assert set(paired["set"]) == {"test", "siteB"}
    controls = pd.read_csv(work / "reports" / "controls.csv")
    assert set(controls["control"]) == {"temporal-shuffle", "prior-swap"}

    # A finished run is skipped, not trained again.
    runs = len(os.listdir(work / "runs"))
    main(["train", "--work", str(work), "--split", "main", "--arm", "A1",
          "--encoder", "resnet18", "--size", "224", *FAST])
    assert len(os.listdir(work / "runs")) == runs
