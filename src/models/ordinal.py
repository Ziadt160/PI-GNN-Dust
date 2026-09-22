"""An ordinal head for dust severity, replacing three independent binary heads.

Why the binary heads had to go
------------------------------
The station model predicts three nested events -- visibility below 5 km, below
1 km, below 0.4 km -- with three independent sigmoid outputs and three
independent BCE losses.  Two things go wrong with that, and we measured both.

**1. Nothing enforces the nesting.**  Every storm hour is also a dusty hour, so
``P(vis < 0.4) <= P(vis < 1) <= P(vis < 5)`` must hold.  Independent heads can and
do violate it, which makes the three numbers unusable as one forecast.

**2. The rarest head starves.**  At a 0.070% base rate the ``severe`` head sees
856 positives in 1.2 million observations and scores TS 0.000 at 24 h -- it never
fires.  Treated as separate problems, the 37,562 ``dusty`` events tell it nothing.
They should: they are the same physical variable further down the same scale.

Both are fixed by predicting the **grade** rather than three memberships::

    0  clear        vis >= 5 km
    1  dusty        1   <= vis < 5 km
    2  dust_storm   0.4 <= vis < 1 km
    3  severe             vis < 0.4 km

One softmax over four ordered grades.  The three threshold probabilities come
back out as tail sums (:func:`cumulative_probabilities`), monotone by
construction, so the metric pipeline and every published comparison are unchanged.

The loss
--------
SORD -- soft ordinal regression (Diaz & Marathe 2019), the loss Sarafian et al.
2023 use for exactly this task in the Middle East.  Ordinary cross-entropy treats
"predicted clear, was severe" and "predicted storm, was severe" as equally wrong.
SORD does not: it replaces the one-hot target with a soft distribution that decays
with distance in grade,

    delta(y)_i = softmax_i( -phi(i, y) ),    phi = |i - y|^2 by default

and then applies the usual cross-entropy against it.  Being one grade out costs
little, being three grades out costs a lot.  That is the correct shape for a
warning system, where under-forecasting a severe storm as clear is the failure
that matters.

Censoring is not an issue here, unlike for the continuous ``log_beta`` target.
METAR visibility is right-censored at the 9.5 km ceiling, but every grade boundary
sits far below it, so a censored observation is unambiguously grade 0.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def grades_from_labels(labels, thresholds):
    """Nested threshold memberships ``[..., K-1]`` -> ordinal grade ``[...]``.

    The thresholds are nested, so the grade is simply how many of them the
    observation crosses.  Order of the threshold columns does not matter.

    Parameters
    ----------
    labels : bool array ``[..., n_thresholds]`` as produced by
        :meth:`src.data.station_target.DustTarget.align`.
    thresholds : the matching threshold values, unused except as a shape check.
    """
    labels = np.asarray(labels)
    if labels.shape[-1] != len(thresholds):
        raise ValueError(f"labels have {labels.shape[-1]} threshold columns but "
                         f"{len(thresholds)} thresholds were given")
    return labels.astype(np.int64).sum(axis=-1)


def cumulative_index(name, thresholds):
    """Which tail sum reproduces the binary event ``name``.

    ``P(vis < t)`` is ``P(grade >= k)`` where ``k`` counts the thresholds at or
    above ``t``: crossing a 1 km threshold means also having crossed 5 km, so the
    1 km event is grade 2 or worse.
    """
    t = thresholds[name]
    return sum(1 for other in thresholds.values() if other >= t)


def cumulative_probabilities(logits, thresholds, names):
    """Grade logits ``[..., K]`` -> per-threshold event probabilities ``[..., len(names)]``.

    Tail sums of one softmax, so ``P`` is monotone in the threshold by
    construction -- the guarantee the independent heads could not give.
    """
    probs = torch.softmax(logits, dim=-1)
    # Reverse cumulative sum: tail[..., k] = P(grade >= k).
    tail = probs.flip(-1).cumsum(-1).flip(-1)
    cols = [cumulative_index(n, thresholds) for n in names]
    return torch.stack([tail[..., k] for k in cols], dim=-1)


def sord_targets(y, n_grades, power=2.0):
    """Soft ordinal targets ``delta(y)`` -- SORD's replacement for the one-hot vector.

    ``delta(y)_i = softmax_i(-|i - y|^power)``: mass concentrated on the true
    grade, decaying with ordinal distance.
    """
    idx = torch.arange(n_grades, device=y.device, dtype=torch.float32)
    dist = (idx.view(*([1] * y.dim()), n_grades) - y.unsqueeze(-1).float()).abs()
    return torch.softmax(-dist.pow(power), dim=-1)


class SORDLoss(nn.Module):
    """Soft ordinal regression loss with per-grade sample weighting.

    ``class_weight`` scales each sample by the weight of its OBSERVED grade.  At
    this base rate it is not optional: grade 0 is 96.9% of observations, and an
    unweighted SORD collapses to predicting clear forever.  It plays the same role
    as ``pos_weight`` did for the binary heads.
    """

    def __init__(self, n_grades, class_weight=None, power=2.0):
        super().__init__()
        self.n_grades = int(n_grades)
        self.power = float(power)
        if class_weight is None:
            class_weight = torch.ones(self.n_grades)
        self.register_buffer("class_weight", torch.as_tensor(
            class_weight, dtype=torch.float32))

    def forward(self, logits, y, mask=None):
        """``logits [..., K]``, ``y [...]`` int64 grades, ``mask [...]`` bool."""
        target = sord_targets(y, self.n_grades, self.power)
        per_sample = -(target * torch.log_softmax(logits, dim=-1)).sum(-1)
        w = self.class_weight[y.clamp(0, self.n_grades - 1)]
        per_sample = per_sample * w
        if mask is None:
            return per_sample.mean()
        m = mask.to(per_sample.dtype)
        return (per_sample * m).sum() / m.sum().clamp(min=1)


def inverse_frequency_weights(y, n_grades, cap=200.0):
    """Per-grade weights relative to the MOST COMMON grade, capped.

    ``w_k = min(n_max / n_k, cap)``, so the majority grade weighs exactly 1 and a
    grade ``r`` times rarer weighs ``r``.  This is the multi-class form of the
    ``pos_weight = n_neg / n_pos`` the binary heads used, capped at the same 200,
    which keeps the two paths on one scale and makes the ablation a comparison of
    heads rather than of loss magnitudes.

    The cap matters: ``severe`` is 0.070% of observations, so an uncapped ratio is
    ~1,400 and a single mislabelled hour would dominate the gradient.

    Normalising to mean 1 instead -- the obvious alternative, and what this
    function did first -- is wrong here.  With 96.9% of observations in grade 0,
    mean-normalisation after capping drives the majority weight to 0.004, so the
    model receives almost no gradient for correctly saying "clear" and
    over-forecasts dust everywhere.  Anchoring on the majority grade avoids that.
    """
    y = np.asarray(y).ravel()
    counts = np.bincount(y, minlength=n_grades).astype("float64")
    counts = np.maximum(counts, 1.0)
    return np.minimum(counts.max() / counts, cap).astype("float32")
