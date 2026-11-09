"""The feature grid of one patient in, one logit out.

Takes a presence mask rather than a length, which is what makes the ladder
honest: the arms differ in what is masked and every arm runs this same code.

The tile axis is a grid axis, never a batch axis: a patient whose finding needed
four windows is still one logit, and folding tiles into the batch would inflate
n and narrow every confidence interval.

To add an aggregator, subclass `Aggregator` and add it to `AGGREGATORS`.
"""

import math
from abc import ABC, abstractmethod

import torch
from torch import nn

from mammo_priors.config import Registry

# The range of intervals the time encoding has to tell apart, in months.
SHORTEST_PERIOD_MONTHS = 1.0
LONGEST_PERIOD_MONTHS = 360.0


class Aggregator(nn.Module, ABC):
    @abstractmethod
    def forward(self, features: torch.Tensor, present: torch.Tensor,
                gap_months: torch.Tensor) -> torch.Tensor:
        """[B, T, V, K, D] features, [B, T, V, K] mask, [B, T] gaps -> [B] logits.

        `gap_months` is the interval from each present exam to the next older
        present one (`mammo_priors.datasets.gaps`).
        """


class MaskedMean(Aggregator):
    """The mean of the present cells, then a linear layer.

    Reads no order and no interval. It is the control that says how much of a
    gain comes from having more patches to average rather than from reading the
    history as a sequence.
    """

    def __init__(self, dimension: int):
        super().__init__()
        self.head = nn.Linear(dimension, 1)

    def forward(self, features, present, gap_months):
        weights = present.unsqueeze(-1).to(features.dtype)
        # Clamped: a padded row of a batch can be all False, and 0/0 would
        # poison the whole batch.
        pooled = (features * weights).sum(dim=(1, 2, 3)) / weights.sum(dim=(1, 2, 3)).clamp(min=1.0)

        return self.head(pooled).squeeze(-1)


def exam_embeddings(features: torch.Tensor, present: torch.Tensor):
    """One vector per exam (the mean of its present cells), and which exams exist.

    Each exam counts once whatever its tile count. At one timepoint this equals
    MaskedMean's pooling, so A1 is a shared anchor across aggregators.
    """
    weights = present.unsqueeze(-1).to(features.dtype)
    per_exam = (features * weights).sum(dim=(2, 3)) / weights.sum(dim=(2, 3)).clamp(min=1.0)

    return per_exam, present.any(dim=(2, 3))


def present_gaps(gap_months: torch.Tensor, timepoint: torch.Tensor) -> torch.Tensor:
    """The gaps with every absent exam's forced to zero, so it cannot shift the others."""
    return gap_months * timepoint.to(gap_months.dtype)


def months_before(gap_months: torch.Tensor, timepoint: torch.Tensor) -> torch.Tensor:
    """Months from the newest present exam back to each exam (newest = 0)."""
    gaps = present_gaps(gap_months, timepoint)
    return torch.cumsum(gaps, dim=1) - gaps


def _compact(steps: torch.Tensor, present: torch.Tensor):
    """Present timepoints of each row moved to the front, in their order.

    So a recurrence never consumes an absent exam sitting between two present ones.
    """
    order = torch.argsort(present.to(torch.int8), dim=1, descending=True, stable=True)
    lengths = present.sum(dim=1).clamp(min=1)
    gathered = torch.gather(steps, 1, order.unsqueeze(-1).expand(-1, -1, steps.shape[-1]))

    return gathered, lengths


def _reverse_prefix(steps: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse the first `lengths[i]` steps of row i: newest-first becomes oldest-first."""
    batch, timepoints = steps.shape[:2]
    position = torch.arange(timepoints, device=steps.device).expand(batch, timepoints)
    span = lengths.unsqueeze(1)
    source = torch.where(position < span, span - 1 - position, position)

    return torch.gather(steps, 1, source.unsqueeze(-1).expand(-1, -1, steps.shape[-1]))


class TemporalGRU(Aggregator):
    """A GRU over the exam vectors, oldest to newest, with an embedding of each gap.

    Read at the newest present exam, where the label lives. Light on purpose:
    histories are short and cohorts small. At one timepoint it is one GRU step
    from a zero state - a small MLP over what MaskedMean sees.
    """

    def __init__(self, dimension: int, hidden: int = 256, gap_features: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        # log1p of the months: intervals span a month to a decade.
        self.gap = nn.Sequential(nn.Linear(1, gap_features), nn.Tanh())
        self.rnn = nn.GRU(dimension + gap_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, features, present, gap_months):
        per_exam, timepoint = exam_embeddings(features, present)
        months = present_gaps(gap_months, timepoint).clamp(min=0)
        steps = torch.cat([per_exam, self.gap(torch.log1p(months).unsqueeze(-1))], dim=-1)

        steps, lengths = _compact(steps, timepoint)
        outputs, _ = self.rnn(_reverse_prefix(steps, lengths))
        newest = outputs[torch.arange(len(lengths), device=outputs.device),
                         (lengths - 1).clamp(min=0)]

        return self.head(self.dropout(newest)).squeeze(-1)


class _ContinuousTime(nn.Module):
    """A learned projection of sinusoids over a real-valued time in months.

    A positional index cannot say that one prior is six months old and another
    six years. The periods span the range a screening history covers, not the
    thousands of positions the classic encoding was built for.
    """

    def __init__(self, dimension: int):
        super().__init__()
        if dimension % 2:
            raise ValueError("the continuous-time encoding needs an even dimension")
        periods = torch.logspace(math.log10(SHORTEST_PERIOD_MONTHS),
                                 math.log10(LONGEST_PERIOD_MONTHS), dimension // 2)
        self.register_buffer("frequency", 2 * math.pi / periods)
        self.project = nn.Linear(dimension, dimension)

    def forward(self, months: torch.Tensor) -> torch.Tensor:
        angles = months.unsqueeze(-1) * self.frequency
        return self.project(torch.cat([angles.sin(), angles.cos()], dim=-1))


class TemporalTransformer(Aggregator):
    """One pre-norm transformer layer over the exams, read out at a query token.

    Exam vectors are projected to a fixed width, so the layer is the same size
    behind every encoder. Time enters as a continuous encoding of months before
    the newest exam; absent exams are removed by the attention mask, so the same
    weights serve a woman with one prior and a woman with many.
    """

    def __init__(self, dimension: int, width: int = 256, heads: int = 4,
                 feedforward: int = None, dropout: float = 0.1):
        super().__init__()
        self.project = nn.Linear(dimension, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feedforward or 2 * width,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.time = _ContinuousTime(width)
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.query, std=0.02)
        self.head = nn.Linear(width, 1)

    def forward(self, features, present, gap_months):
        per_exam, timepoint = exam_embeddings(features, present)
        tokens = self.project(per_exam) + self.time(months_before(gap_months, timepoint))

        batch = tokens.shape[0]
        now = self.time(torch.zeros(batch, 1, device=tokens.device))
        sequence = torch.cat([self.query + now, tokens], dim=1)
        # True marks what attention ignores: never the query, every absent exam.
        ignored = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device),
                             ~timepoint], dim=1)

        return self.head(self.encoder(sequence, src_key_padding_mask=ignored)[:, 0]).squeeze(-1)


MASKED_MEAN = "masked-mean"

AGGREGATORS = Registry("aggregator", {
    MASKED_MEAN: MaskedMean,
    "gru": TemporalGRU,
    "transformer": TemporalTransformer,
})
