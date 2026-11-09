"""What a cache and a run are, written down so that every number can be traced back.

A run is named by the digest of everything that could change its result, so two
runs that differ in any setting can never overwrite each other, and a finished
run is recognised and skipped rather than trained again.
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass

CONFIG_JSON = "config.json"

# The side of the window a finding is tiled by, in pixels of the image it is cut
# from. One number for every arm and timepoint: a window that meant different
# things in different arms would make an arm difference a comparison of two
# different inputs. Smaller model windows are cut from the centre of this one.
PATCH_PX = 512

# How much wider than the window each cached crop is. The margin is what
# training-time jitter is drawn from. The prior's is wider because its window is
# placed by a method rather than by an annotation, so more misplacement has to
# fit inside it.
INDEX_MARGIN_PX = 640
PRIOR_MARGIN_PX = 768


class Registry:
    """Name to class, so a component is chosen by a config and never imported.

    An experiment that names its encoder can swap it without being edited, and a
    run config then says exactly which component produced its numbers.
    """

    def __init__(self, kind: str, classes: dict):
        self.kind = kind
        self.classes = classes

    def build(self, config: "ComponentConfig", **arguments):
        if config.name not in self.classes:
            raise KeyError(f"no {self.kind} named {config.name!r}; "
                           f"choose one of {sorted(self.classes)}")

        return self.classes[config.name](**config.options, **arguments)

    def __iter__(self):
        return iter(self.classes)

    def __contains__(self, name):
        return name in self.classes


def digest_of(value) -> str:
    """A stable short hash of anything JSON can hold, with sorted keys."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


@dataclass(frozen=True)
class ComponentConfig:
    """A registry name plus the options the component is built with."""

    name: str
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PatchConfig:
    """Everything that decides what is in the patch cache, and nothing else.

    Its digest names the cache folder, so a change to any of these cuts a new
    cache instead of mixing two settings in one. The encoder is absent on
    purpose: the same patches serve every encoder.
    """

    correspondence: ComponentConfig
    cohort_revision: str
    tile_px: int = PATCH_PX
    index_margin_px: int = INDEX_MARGIN_PX
    prior_margin_px: int = PRIOR_MARGIN_PX
    # Right breasts are mirrored so the chest wall is always on the same side.
    canonicalise_laterality: bool = True

    def margin_px(self, prior: bool) -> int:
        return self.prior_margin_px if prior else self.index_margin_px

    def digest(self) -> str:
        return digest_of(asdict(self))


@dataclass(frozen=True)
class TrainingConfig:
    """The optimisation schedule, identical for every arm.

    The encoder carries pretrained weights worth keeping and the aggregator
    starts at random, so the head learns ten times faster than the backbone.
    """

    epochs: int = 40
    patience: int = 8
    batch: int = 8
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3


@dataclass(frozen=True)
class AugmentConfig:
    """What training may vary. One draw covers a patient's whole grid."""

    flip_probability: float = 0.5
    quarter_turns: bool = True
    brightness: float = 0.10
    contrast: float = 0.10
    # The most pixels the window may move off the finding's centre per axis.
    shift_px: int = 0


@dataclass(frozen=True)
class PreprocessConfig:
    """One intensity window per patient: percentiles of her present cells.

    Per patient, so an exam and its priors are divided by the same two numbers
    and a change between them survives; percentiles, so absolute intensity -
    which identifies the scanner - does not.
    """

    window_percentiles: tuple = (1.0, 99.0)
    mean: float = 0.5
    std: float = 0.25


@dataclass(frozen=True)
class RunConfig:
    """One training run, as the thing that produced a row of the results."""

    arm: str
    encoder: ComponentConfig
    aggregator: ComponentConfig
    patches: PatchConfig
    size_px: int
    split: str
    split_digest: str
    seed: int = 0
    view: str = None
    interval: str = "rounds"
    balance: str = "none"
    max_cells: int = None
    checkpoint_oversized: bool = False
    test_time_augment: bool = False
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def digest(self) -> str:
        return digest_of(asdict(self))

    def save(self, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, CONFIG_JSON)
        with open(path, "w") as config_file:
            json.dump(asdict(self), config_file, indent=2, sort_keys=True, default=str)

        return path

    @classmethod
    def read(cls, folder: str) -> "RunConfig":
        with open(os.path.join(folder, CONFIG_JSON)) as config_file:
            return _rebuilt(cls, json.load(config_file))

    def __str__(self):
        view = f" {self.view}" if self.view else ""
        return (f"{self.arm}{view} {self.encoder.name}+{self.aggregator.name} "
                f"{self.size_px}px {self.split}")


def _rebuilt(cls, recorded: dict):
    """A (nested) frozen dataclass back from the dictionary `asdict` wrote."""
    values = {}
    for item in fields(cls):
        if item.name not in recorded:
            continue
        value = recorded[item.name]
        kind = item.type if isinstance(item.type, type) else _NESTED.get(item.name)
        if kind is not None and is_dataclass(kind):
            value = _rebuilt(kind, value)
        elif isinstance(value, list):
            value = tuple(value)
        values[item.name] = value

    return cls(**values)


# The nested configs by field name, for `_rebuilt` when annotations are strings.
_NESTED = {
    "encoder": ComponentConfig, "aggregator": ComponentConfig,
    "correspondence": ComponentConfig, "patches": PatchConfig,
    "preprocess": PreprocessConfig, "augment": AugmentConfig,
    "training": TrainingConfig,
}
