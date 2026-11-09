"""The two input tables, read and checked before anything is derived from them.

`images.csv` lists every image: which woman, which exam, which breast and view,
and where the file is. `findings.csv` lists the biopsy-proven findings drawn on
those images, with a box and a label. Everything else - the index exam, the
priors, the patient label, the patches - is derived from these two, so a dataset
is brought to the ladder by writing them and nothing else.
"""

import pandas as pd

BENIGN, MALIGNANT = 0, 1
LABELS = {"benign": BENIGN, "malignant": MALIGNANT, "0": BENIGN, "1": MALIGNANT}

# The view axis of every grid, in this order. CC first, so a one-view arm that
# keeps the first view a patient has keeps CC when both exist.
VIEWS = ("CC", "MLO")
LEFT, RIGHT = "L", "R"
LATERALITIES = (LEFT, RIGHT)

# A source that is not named is one source, so a single dataset needs no column.
DEFAULT_SOURCE = "data"

IMAGE_COLUMNS = ("patient_id", "exam_date", "laterality", "view", "path")
FINDING_COLUMNS = ("patient_id", "exam_date", "laterality", "view",
                   "x_min", "y_min", "x_max", "y_max", "label")
# What addresses one image of one breast at one exam.
IMAGE_KEYS = ["source", "patient_id", "exam_date", "laterality", "view"]


class TableError(ValueError):
    """An input table that cannot be used as it stands, and why."""


def _read(path: str, required: tuple) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"patient_id": str, "source": str})
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise TableError(f"{path} lacks the columns {missing}; "
                         f"it needs {list(required)} (and optionally 'source')")
    if "source" not in table.columns:
        table["source"] = DEFAULT_SOURCE
    table["source"] = table["source"].fillna(DEFAULT_SOURCE).astype(str)
    # A patient is named "source/patient_id" downstream, so a source must not
    # hold the separator.
    if table["source"].str.contains(r"[/|]").any():
        raise TableError(f"{path}: a source name may not contain '/' or '|'")

    return _normalised(table, path)


def _normalised(table: pd.DataFrame, path: str) -> pd.DataFrame:
    """Dates as ISO strings, laterality and view upper-cased and checked.

    Dates are compared and sorted as strings everywhere after this, which is
    only correct for one format; parsing once here is what makes that safe.
    """
    table = table.copy()
    parsed = pd.to_datetime(table["exam_date"], errors="coerce")
    if parsed.isna().any():
        bad = table.loc[parsed.isna(), "exam_date"].head(3).tolist()
        raise TableError(f"{path}: unreadable exam_date values, e.g. {bad}")
    table["exam_date"] = parsed.dt.strftime("%Y-%m-%d")

    table["laterality"] = table["laterality"].astype(str).str.upper().str[0]
    table["view"] = table["view"].astype(str).str.upper()
    for column, allowed in (("laterality", LATERALITIES), ("view", VIEWS)):
        wrong = sorted(set(table[column]) - set(allowed))
        if wrong:
            raise TableError(f"{path}: {column} must be one of {allowed}, found {wrong}")

    return table


def read_images(path: str) -> pd.DataFrame:
    """The image table; a repeated (exam, breast, view) keeps the first path by name.

    Nothing in the table prefers one repeat acquisition over another, so the
    rule is chosen for being reproducible.
    """
    images = _read(path, IMAGE_COLUMNS)

    return images.sort_values("path").drop_duplicates(IMAGE_KEYS).reset_index(drop=True)


def read_findings(path: str) -> pd.DataFrame:
    """The proven findings, labels as 0 (benign) and 1 (malignant).

    A row without a label is a finding nobody biopsied: it can neither date an
    index exam nor carry a label, so it is dropped here and counted.
    """
    findings = _read(path, FINDING_COLUMNS)
    labelled = findings["label"].notna()
    if not labelled.all():
        print(f"{path}: {int((~labelled).sum())} findings without a label are ignored")
    findings = findings[labelled].copy()

    text = findings["label"].astype(str).str.strip().str.lower().str.replace(".0", "", regex=False)
    findings["label"] = text.map(LABELS)
    if findings["label"].isna().any():
        bad = sorted(set(text[findings["label"].isna()]))
        raise TableError(f"{path}: label must be benign/malignant or 0/1, found {bad}")
    findings["label"] = findings["label"].astype(int)

    for column in ("x_min", "y_min", "x_max", "y_max"):
        findings[column] = findings[column].astype(float).round().astype(int)
    inverted = (findings["x_max"] <= findings["x_min"]) | (findings["y_max"] <= findings["y_min"])
    if inverted.any():
        raise TableError(f"{path}: {int(inverted.sum())} boxes have max <= min")

    return findings.reset_index(drop=True)
