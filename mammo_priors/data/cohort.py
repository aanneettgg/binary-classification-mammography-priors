"""What a sample is: one patient, her index exam, the exams before it, her label.

The unit of analysis is a patient, not an image or a finding, so one sample is
one prediction and the resampling unit of every confidence interval.

The rules are research decisions and are applied here and nowhere else:

- The **index exam** is the latest exam holding a proven finding. Exams after it
  are discarded: the finding had already been resolved by then.
- The **label** is the worst finding on the index exam.
- A **prior** is an earlier exam that holds an image of a breast with a finding.
- A patient needs at least `min_priors` priors (one by default). Whether a
  prior *exists* can predict the label on its own - patients referred at their
  first screen differ from those referred later - and a model shown that fact
  would learn it without reading a pixel. Requiring a prior removes it, and puts
  every arm on exactly the same women.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from mammo_priors.data.tables import IMAGE_KEYS, VIEWS


@dataclass(frozen=True)
class Finding:
    """One proven box on the index exam, in the pixels of the image as stored."""

    laterality: str
    view: str
    box_yx: tuple          # (y_min, x_min, y_max, x_max)
    label: int


@dataclass
class Exam:
    """One visit: its date and the image of each breast and view it holds."""

    date: str
    images: dict = field(default_factory=dict)    # "L/CC" -> path

    def image_of(self, laterality: str, view: str) -> str:
        """The path of one image, or None: a missing view is a hole in the grid."""
        return self.images.get(f"{laterality}/{view}")


@dataclass
class Sample:
    """One patient. Exams newest first, so t = 0 is the index exam.

    Newest first because an arm is a slice of the time axis: "index only" is a
    prefix and "index plus every prior" is the whole axis.
    """

    patient_id: str
    source: str
    label: int
    exams: list
    findings: list

    @property
    def key(self) -> str:
        """The patient's name across sources, which may reuse the same ids."""
        return f"{self.source}/{self.patient_id}"

    @property
    def dates(self) -> list:
        return [exam.date for exam in self.exams]

    @property
    def priors(self) -> int:
        return len(self.exams) - 1

    @classmethod
    def from_record(cls, record: dict) -> "Sample":
        return cls(
            patient_id=record["patient_id"], source=record["source"],
            label=int(record["label"]),
            exams=[Exam(**exam) for exam in record["exams"]],
            findings=[Finding(finding["laterality"], finding["view"],
                              tuple(finding["box_yx"]), int(finding["label"]))
                      for finding in record["findings"]],
        )


class Cohort:
    """Every sample, written to one JSON file that extraction and training read.

    Self-contained - it holds the image paths - so later steps never reopen the
    input tables, and its revision (a digest of its content) names the patch
    cache cut from it: a corrected box can never land in a cache cut before it.
    """

    def __init__(self, samples: list):
        self.samples = samples

    def revision(self) -> str:
        return hashlib.sha256(json.dumps(
            [asdict(sample) for sample in self.samples], sort_keys=True
        ).encode()).hexdigest()[:16]

    def save(self, path: str):
        with open(path, "w") as cohort_file:
            json.dump({"revision": self.revision(),
                       "samples": [asdict(sample) for sample in self.samples]},
                      cohort_file, indent=1)

    @classmethod
    def load(cls, path: str) -> "Cohort":
        with open(path) as cohort_file:
            return cls([Sample.from_record(record)
                        for record in json.load(cohort_file)["samples"]])

    def of_sources(self, sources) -> list:
        return [sample for sample in self.samples if sample.source in sources]

    @property
    def sources(self) -> list:
        return sorted({sample.source for sample in self.samples})

    def describe(self) -> str:
        frame = pd.DataFrame([{"source": sample.source, "label": sample.label,
                               "priors": sample.priors} for sample in self.samples])
        lines = [f"{len(frame)} patients",
                 pd.crosstab(frame["source"], frame["label"].map(
                     {0: "benign", 1: "malignant"})).to_string(),
                 "priors per patient:",
                 frame["priors"].value_counts().sort_index().to_string()]

        return "\n".join(lines)


class CohortBuilder:
    """The samples the two input tables describe, under the rules of the module."""

    def __init__(self, images: pd.DataFrame, findings: pd.DataFrame,
                 min_priors: int = 1):
        self.images = images
        self.findings = findings
        self.min_priors = min_priors
        self.excluded = {}

    def _index_dates(self) -> pd.Series:
        return self.findings.groupby(["source", "patient_id"])["exam_date"].max()

    def _on_index(self, index: pd.Series) -> pd.DataFrame:
        """The findings on the index exam that sit on an image the table lists.

        A finding whose image is missing cannot be cut, and keeping its label
        while dropping its pixels would label a patient by something no arm sees.
        """
        dated = self.findings.join(index.rename("index_date"), on=["source", "patient_id"])
        on_index = dated[dated["exam_date"] == dated["index_date"]]
        listed = on_index.merge(self.images[IMAGE_KEYS], on=IMAGE_KEYS, how="inner")
        self.excluded["finding without a listed image"] = len(on_index) - len(listed)

        return listed

    def _exams(self, patient: pd.DataFrame, index_date: str, breasts: set) -> list:
        """The index exam and every earlier one holding a finding breast, newest first."""
        mine = patient[(patient["exam_date"] <= index_date)
                       & patient["laterality"].isin(breasts)]
        exams = []
        for date, visit in sorted(mine.groupby("exam_date"), reverse=True):
            exams.append(Exam(date, {f"{row.laterality}/{row.view}": row.path
                                     for row in visit.itertuples()}))

        return exams

    def build(self) -> Cohort:
        index = self._index_dates()
        on_index = self._on_index(index)
        images = {key: group for key, group in self.images.groupby(["source", "patient_id"])}

        samples, few_priors = [], 0
        for (source, patient_id), found in on_index.groupby(["source", "patient_id"]):
            exams = self._exams(images[(source, patient_id)], index[(source, patient_id)],
                                set(found["laterality"]))
            if len(exams) - 1 < self.min_priors:
                few_priors += 1
                continue

            findings = [Finding(row.laterality, row.view,
                                tuple(int(value) for value in
                                      (row.y_min, row.x_min, row.y_max, row.x_max)),
                                int(row.label))
                        for row in found.sort_values(
                            ["laterality", "view", "y_min", "x_min"]).itertuples()]
            samples.append(Sample(patient_id, source, int(found["label"].max()),
                                  exams, findings))

        self.excluded[f"fewer than {self.min_priors} prior(s)"] = few_priors
        print("excluded: " + ", ".join(f"{count} {reason}"
                                       for reason, count in self.excluded.items()))

        return Cohort(sorted(samples, key=lambda sample: sample.key))


def views_of(sample: Sample) -> list:
    """The findings of a sample grouped by view position, in the order of VIEWS."""
    return [[finding for finding in sample.findings if finding.view == view]
            for view in VIEWS]
