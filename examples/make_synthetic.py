"""Write a small synthetic dataset in the input format, to try the pipeline end to end.

    python examples/make_synthetic.py --out example_data --patients 80

Every patient has a finding on her latest exam, drawn on both views. On the
index exam a malignant and a benign finding look alike (same size range), so a
model without priors should sit near chance. What differs is the history: a
malignant finding was smaller on each earlier exam, while a benign one was
already there at the same size. The data are built so that only the priors
carry the label; what the ladder makes of it depends on the model and the
number of patients. Real data will not be this tidy.

Two sources are written, `siteA` and `siteB`, so an external test set can be
tried as well. The images are 16-bit PNG.
"""

import argparse
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
from PIL import Image

SIDE = 800
VIEWS = ("CC", "MLO")


def breast(generator: np.random.Generator, laterality: str) -> np.ndarray:
    """A half-ellipse of smooth 'tissue' against the chest-wall edge, background dark."""
    rows, columns = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
    depth = generator.uniform(0.55, 0.75) * SIDE
    inside = (columns / depth) ** 2 + ((rows - SIDE / 2) / (0.45 * SIDE)) ** 2 < 1
    texture = 1200 + 300 * np.sin(rows / generator.uniform(40, 90)) * np.cos(
        columns / generator.uniform(40, 90))
    image = np.where(inside, texture, 100.0)

    return image[:, ::-1] if laterality == "R" else image


def with_lesion(image: np.ndarray, centre: tuple, radius: float) -> np.ndarray:
    if radius <= 0:
        return image
    rows, columns = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
    blob = np.exp(-((rows - centre[0]) ** 2 + (columns - centre[1]) ** 2) / (2 * radius ** 2))
    return image + 1500 * blob


def patient(generator, number: int, source: str, out: str) -> tuple:
    patient_id = f"P{number:04d}"
    label = int(generator.random() < 0.5)
    laterality = "L" if generator.random() < 0.5 else "R"
    exams = int(generator.integers(2, 5))
    radius = generator.uniform(14, 28)
    index_day = date(2024, 1, 1) + timedelta(days=int(generator.integers(0, 365)))
    centres = {view: (generator.uniform(250, 550), generator.uniform(120, 300))
               for view in VIEWS}
    if laterality == "R":
        centres = {view: (y, SIDE - x) for view, (y, x) in centres.items()}

    images, findings = [], []
    for t in range(exams):
        day = index_day - timedelta(days=int(730 * t + generator.integers(-60, 60)))
        # Malignant findings grow towards the index exam; benign ones do not.
        size = radius * (0.5 ** t) if label else radius
        shift = generator.integers(-15, 16, size=2)
        for side in ("L", "R"):
            for view in VIEWS:
                image = breast(generator, side)
                if side == laterality:
                    y, x = centres[view]
                    image = with_lesion(image, (y + shift[0], x + shift[1]), size)
                path = f"{source}/{patient_id}/{day.isoformat()}_{side}_{view}.png"
                os.makedirs(os.path.join(out, os.path.dirname(path)), exist_ok=True)
                Image.fromarray(np.clip(image, 0, 65535).astype(np.uint16)).save(
                    os.path.join(out, path))
                images.append({"source": source, "patient_id": patient_id,
                               "exam_date": day.isoformat(), "laterality": side,
                               "view": view, "path": path})
                if t == 0 and side == laterality:
                    y, x = centres[view]
                    half = 2.5 * radius
                    findings.append({"source": source, "patient_id": patient_id,
                                     "exam_date": day.isoformat(), "laterality": side,
                                     "view": view, "x_min": x - half, "y_min": y - half,
                                     "x_max": x + half, "y_max": y + half,
                                     "label": "malignant" if label else "benign"})

    return images, findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="example_data")
    parser.add_argument("--patients", type=int, default=80)
    parser.add_argument("--external-patients", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    generator = np.random.default_rng(arguments.seed)
    images, findings = [], []
    for source, count in (("siteA", arguments.patients),
                          ("siteB", arguments.external_patients)):
        for number in range(count):
            new_images, new_findings = patient(generator, number, source, arguments.out)
            images += new_images
            findings += new_findings

    pd.DataFrame(images).to_csv(os.path.join(arguments.out, "images.csv"), index=False)
    pd.DataFrame(findings).to_csv(os.path.join(arguments.out, "findings.csv"), index=False)
    print(f"wrote {len(images)} images and {len(findings)} findings to {arguments.out}")


if __name__ == "__main__":
    main()
