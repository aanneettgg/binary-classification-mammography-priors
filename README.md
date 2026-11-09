# binary-classification-mammography-priors

An **ablation ladder** that measures what prior mammograms add to classifying a
biopsied finding as benign or malignant.

A patient's input is a grid of **exams × views × tiles**: the index exam (the one
whose finding went to biopsy), every earlier exam of the same breast, the CC and
MLO views, and the patches covering the finding. Each arm of the ladder sees a
different part of that grid:

| Arm | Input | Question |
|---|---|---|
| A0 | index exam, one view | the floor |
| A1 | index exam, CC + MLO | baseline without history |
| A2 | index exam + the most recent prior | what one prior adds |
| A3 | index exam + every prior | what the whole history adds |

The report gives the paired differences **A1 − A0**, **A2 − A1**, **A3 − A1**
(the main comparison) and **A3 − A2**.

## Design: what makes an arm difference mean something

- **An arm is a mask, not a model.** Every arm runs the same encoder, aggregator,
  loss, schedule, split and patients; only the visible cells change. A difference
  between two arms is therefore a difference in input alone.
- **Only visible patches reach the encoder**, so a masked patch cannot affect a
  prediction, even through a bug.
- **Every patient owns at least one prior.** Whether a prior *exists* can predict
  the label on its own (for example, women referred at their first screen differ
  from those referred later). Requiring one removes that signal and puts every
  arm on the same women.
- **Time is given as order, not duration** by default (`--interval rounds`). A
  measured screening interval can differ between the classes or identify the
  screening programme. `--interval months` is available; check it with the
  probe first.
- **One intensity window per patient** (1st–99th percentile of all her patches).
  This keeps any change between exams and removes absolute intensity, which
  identifies the scanner.
- **Augmentation is drawn once per patient**: flips, quarter turns, brightness and
  contrast change the index exam and its priors together.
- **Native resolution.** Windows are cropped, never resampled. A 512 px window
  is tiled over findings larger than that; a 224 px window is cut from its centre.
- **Paired bootstrap.** Two arms score the same patients, so each resample
  recomputes both AUCs on the same draw (2,000 resamples, patient level).
- **Availability probe.** A logistic regression that reads only which exams an
  arm shows and their gaps. A with-prior gain counts as coming from the
  priors' pixels only where this probe is at chance.
- **Negative controls** on trained models, with no retraining:
  - *Temporal shuffle* puts the exams in a wrong order. A sequence model should
    lose AUC; the order-blind masked mean must not change.
  - *Prior swap* replaces a woman's priors with those of another woman from the
    same source with the same exam layout. If the model reads the priors as her
    own anatomy, the AUC falls towards A1.
- **Frozen splits**: patient-level, 60/20/20, stratified by label and number of
  priors, ordered by a hash of the patient id, written to a file with a digest
  and never silently rewritten. Five folds rotate the test set.
- **Every run is named by the digest of its full configuration.** Finished runs
  are skipped, and two different settings can never overwrite each other.

## Installation

```bash
pip install -e .                    # core
pip install -e ".[registration]"    # + SimpleITK, for registered prior placement
pip install -e ".[dicom-codecs]"    # + decoders for compressed DICOM
pip install -e ".[test]"            # + pytest
```

Install the PyTorch build that matches your CUDA version from
[pytorch.org](https://pytorch.org) first if you train on a GPU. Python ≥ 3.10.

## Input: two CSV files

**`images.csv`**: one row per image.

| column | meaning |
|---|---|
| `patient_id` | patient identifier |
| `exam_date` | date of the exam (any format pandas parses; ISO recommended) |
| `laterality` | `L` or `R` |
| `view` | `CC` or `MLO` |
| `path` | image file, relative to `--image-root` (DICOM, or 16-bit PNG/TIFF) |
| `source` | *optional*: site or dataset name, for pooled training and external test sets |

**`findings.csv`**: one row per biopsy-proven finding, drawn on the image of
that patient, exam, breast and view.

| column | meaning |
|---|---|
| `patient_id`, `exam_date`, `laterality`, `view`, (`source`) | which image |
| `x_min`, `y_min`, `x_max`, `y_max` | bounding box in pixels of the image **as stored** |
| `label` | `benign` / `malignant` (or `0` / `1`); rows without a label are ignored |

From these two tables the pipeline derives:

- the **index exam**: each patient's latest exam with a finding; later exams are
  discarded;
- the **label**: the worst finding on the index exam;
- the **priors**: earlier exams holding an image of a breast with a finding.

Images must be at least 768 px on each side, because each prior crop is
stored with that margin. MONOCHROME1 DICOM is inverted automatically. Right
breasts are mirrored so the chest wall is always on the same side. A prior
stored as the mirror image of its index exam is detected from its pixels and
corrected.

## Quick start on synthetic data

```bash
python examples/make_synthetic.py --out example_data --patients 100
mammo-priors cohort  --work work --images example_data/images.csv --findings example_data/findings.csv
mammo-priors split   --work work --name main --external siteB
mammo-priors extract --work work --image-root example_data
mammo-priors probe   --work work --split main
mammo-priors ladder  --work work --split main --encoders resnet18-imagenet --sizes 224 --epochs 10
mammo-priors report  --work work
```

In the synthetic data, a malignant finding was smaller on earlier exams, and a
benign one was already there at the same size. On the index exam alone the two
look alike, so only the priors carry the label.

## Running the ladder on your data

```bash
mammo-priors cohort  --work W --images images.csv --findings findings.csv
mammo-priors split   --work W --name main [--external SITE_B ...]
mammo-priors extract --work W --image-root /path/to/images
mammo-priors probe   --work W --split main            # check before reading any with-prior result
mammo-priors ladder  --work W --split main --keep-weights
mammo-priors controls --work W --split main          # needs the weights kept above
mammo-priors report  --work W                        # W/reports/report.txt + CSV tables
```

By default the ladder runs **A0–A3 × ResNet-18/34/50 (ImageNet) × two window
sizes (224, 512 px)**:

- A2 and A3 run under all three aggregators (masked mean, GRU, transformer).
- A0 and A1 run under the masked mean only.

Add `--a1-aggregators masked-mean gru transformer` to also train A1 under each
aggregator. Then every with-prior run is paired with an A1 that has the same
head, which separates the effect of the priors from the effect of the
aggregator. Use `mammo-priors train ...` for a single run.

Training: AdamW (encoder 1e-4, aggregator 1e-3, weight decay 1e-4), one-cycle
schedule with 3 warm-up epochs, up to 40 epochs, early stopping after 8 epochs
without improvement in validation AUC, 8 patients per batch, positive class
weighted by negatives / positives.

### Options

| option | effect |
|---|---|
| `split --external SITE` | sources never trained on; scored once with the selected weights |
| `split --pooled` | several training sources: stratified by source; every epoch draws each source's classes equally; AUC is the mean of the per-source AUCs |
| `split --fold k` | fold *k* of 5; across the folds every patient is tested once |
| `extract/ladder --correspondence centroid\|affine\|deformable` | priors placed by registration instead of the same pixel coordinates (needs `[registration]`) |
| `--view CC\|MLO` | the ladder with one view only |
| `--shift-px 32 --test-time-augment` | window jitter in training; mean over 8 orientations at test |
| `--max-cells N` | at most N patches per forward pass (gradients accumulated) |
| `--checkpoint-oversized` | encode a patient larger than `--max-cells` in checkpointed chunks |

To compare a registration method with same-location placement, cut its cache
and run only the arms that see priors:
`mammo-priors extract --work W --image-root ... --correspondence deformable`,
then `mammo-priors ladder ... --correspondence deformable --arms A2 A3`.
Arms without priors read the same index patches from every cache, so the report
pairs them with the with-prior arms of every method.

## Working folder

```
W/cohort.json                    the derived cohort (self-contained: holds the image paths)
W/splits/fold<k>/<name>.json     frozen splits
W/cache/<digest>/...             patch cache, one .npz per patient
W/runs/<digest>/config.json      what the run was
W/runs/<digest>/metrics.json     validation / test / external AUC + every patient's logit
W/runs/<digest>/controls.json    negative controls, if scored
W/reports/                       report.txt, runs.csv, paired.csv, controls.csv
```

## Extending

- **Encoder**: subclass `models.encoder.Encoder`, add it to `ENCODERS`.
- **Aggregator**: subclass `models.aggregator.Aggregator`, add it to `AGGREGATORS`.
- **Arm**: one `Arm(...)` line in `datasets/arms.py`. For example,
  `Arm("A4", slice(1, None))` gives the priors without the index exam.
- **Correspondence method**: subclass `patches.correspondence.Correspondence`.
- **Negative control**: subclass `datasets.controls.Control`, add it to `CONTROLS`.

## Tests

```bash
pytest
```

The unit tests check tiling, arm masks, gaps, aggregator masking, metrics and
splits. `tests/test_pipeline.py` runs every command on a small synthetic dataset
on the CPU (about a minute and a half).

## License

MIT, see [LICENSE](LICENSE).
