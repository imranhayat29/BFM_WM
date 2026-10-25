"""
Evaluation metrics of the BFM wall-model predictions (a-priori, against dataset targets).

Percent errors use a low-value blend: relative to |pi_out,true| where it is above a
small threshold (default 2 +- 1), relative to the laminar-law reference sqrt(|pi_1|)
below it, so that points with a vanishing wall shear stress (separation, reattachment,
laminar regions) do not dominate the statistics. True targets are used here only, never
in predictions; ``predicted_percent`` is the inference-only counterpart for predicted
quantities (normalised by the baseline prediction).
"""
import json
import os
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src_err.gating import blend_diagnostics, lowvalue_percent
from src_err.utils import to_jsonable

# Pipeline output key -> name in the summary (only the keys present are summarised)
DEFAULT_ERROR_FIELDS = {
    "baseline": "e_base_pct",
    "corrected": "e_final_pct",
    "always_corrected": "e_corr_pct",
    "eqwm": "e_eqwm_pct",
}


def percent_error(y_pred: np.ndarray,
                  y_true: np.ndarray,
                  pi1_abs: np.ndarray,
                  thr_low: float = 2.0,
                  delta_low: float = 1.0) -> np.ndarray:
    """Signed percent error of a pi_out prediction, with the low-value blend on the truth.

    ``e = y_true - y_pred``, ``w = ramp_above(|y_true|; thr_low, delta_low)`` and
    ``e% = w 100 e / max(|y_true|, 1e-12) + (1 - w) 100 e / max(sqrt|pi_1|, 1e-12)``.
    Both arguments are pi_out magnitudes.

    Args:
        y_pred: Predicted pi_out.
        y_true: Dataset pi_out (evaluation only).
        pi1_abs: |pi_1|.
        thr_low: Ramp threshold on |y_true| ([metrics.LowValue].Threshold).
        delta_low: Ramp half-width ([metrics.LowValue].Delta).

    Returns:
        Signed percent errors (float64).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    e = y_true - np.asarray(y_pred, dtype=np.float64)
    return lowvalue_percent(e, y_true, pi1_abs, thr_low, delta_low)


def predicted_percent(q: np.ndarray,
                      ref_abs: np.ndarray,
                      pi1_abs: np.ndarray,
                      thr_low: float = 2.0,
                      delta_low: float = 1.0) -> np.ndarray:
    """Percent of a predicted quantity (error mean or sigma), inference-only.

    Same blend as ``percent_error`` but normalised by the baseline prediction
    ``ref_abs = |pi_out,baseline|`` ([gates.LowValue]); used for the mu% and sigma% maps.

    Args:
        q: Predicted quantity in pi_out units (signed values are kept).
        ref_abs: |pi_out,baseline|.
        pi1_abs: |pi_1|.
        thr_low: Ramp threshold on |ref|.
        delta_low: Ramp half-width.

    Returns:
        Percent values (float64).
    """
    return lowvalue_percent(q, ref_abs, pi1_abs, thr_low, delta_low)


def error_stats(e_pct: np.ndarray) -> Dict[str, float]:
    """Summary statistics of percent errors (non-finite values are ignored).

    Args:
        e_pct: Signed percent errors.

    Returns:
        Dict with ``mean_abs``, ``median_abs``, ``p90_abs``, ``frac_le_10``, ``frac_le_20``
        (fractions in [0, 1] of |e%| <= 10 and <= 20) and ``count`` (finite points).
    """
    a = np.abs(np.asarray(e_pct, dtype=np.float64))
    a = a[np.isfinite(a)]
    if a.size == 0:
        nan = float("nan")
        return {"mean_abs": nan, "median_abs": nan, "p90_abs": nan,
                "frac_le_10": nan, "frac_le_20": nan, "count": 0}
    return {
        "mean_abs": float(np.mean(a)),
        "median_abs": float(np.median(a)),
        "p90_abs": float(np.percentile(a, 90.0)),
        "frac_le_10": float(np.mean(a <= 10.0)),
        "frac_le_20": float(np.mean(a <= 20.0)),
        "count": int(a.size),
    }


def _value_stats(v: np.ndarray) -> Dict[str, float]:
    """Mean/median/min/max/p10/p90 of the finite values of ``v``."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"count": 0}
    return {"mean": float(np.mean(v)), "median": float(np.median(v)),
            "min": float(np.min(v)), "max": float(np.max(v)),
            "p10": float(np.percentile(v, 10.0)), "p90": float(np.percentile(v, 90.0)),
            "count": int(v.size)}


def _subset_stats(out: Mapping[str, np.ndarray],
                  mask: np.ndarray,
                  error_fields: Mapping[str, str]) -> Dict[str, Any]:
    """Error and confidence statistics on the points selected by ``mask``."""
    res: Dict[str, Any] = {"count": int(np.sum(mask)),
                           "errors": {name: error_stats(np.asarray(out[key])[mask])
                                      for name, key in error_fields.items() if key in out}}
    if "confidence" in out:
        res["confidence"] = _value_stats(np.asarray(out["confidence"])[mask])
    return res


def summarize(out: Mapping[str, np.ndarray],
              dataset_key: Optional[np.ndarray] = None,
              class_names: Optional[Sequence[str]] = None,
              label_key: str = "labels_posthoc",
              error_fields: Mapping[str, str] = DEFAULT_ERROR_FIELDS,
              extra: Optional[Mapping[str, Any]] = None,
              verbose: bool = True) -> Dict[str, Any]:
    """Summary statistics of a prediction run (the core of ``summary.json``).

    Args:
        out: Per-point pipeline outputs (``pipeline.predict_turb``/``predict_lamturb``).
            Uses the ``error_fields`` keys, ``confidence``, ``sigma_pct``, ``mu_pct``,
            ``alpha_nw``/``alpha_c``/``alpha_a``, ``label_key`` and ``laminar`` when present.
        dataset_key: Per-point dataset key, for the per-dataset statistics.
        class_names: Names indexed by the labels in ``out[label_key]``.
        label_key: Label field to count.
        error_fields: Summary name -> output key of the percent errors.
        extra: Entries merged at the top level (models, config, gates, versions...).
        verbose: Print the blend diagnostics and an error table.

    Returns:
        JSON-serialisable dict: ``num_samples``, ``errors``, ``per_dataset``,
        ``confidence``, ``gate_metric``, ``blend``, ``class_counts`` and, when any point
        is laminar, ``switch`` (lam/turb counts and per-branch statistics).
    """
    keys = [key for key in error_fields.values() if key in out]
    n = int(len(np.asarray(out[keys[0]]))) if keys else int(len(np.asarray(out["y_final"])))
    everything = np.ones(n, dtype=bool)
    overall = _subset_stats(out, everything, error_fields)
    summary: Dict[str, Any] = {"num_samples": n, "errors": overall["errors"]}

    if dataset_key is not None:
        dk = np.asarray(dataset_key).astype(str)
        summary["per_dataset"] = {str(k): _subset_stats(out, dk == k, error_fields)
                                  for k in np.unique(dk)}
    if "confidence" in out:
        summary["confidence"] = overall["confidence"]
    summary["gate_metric"] = {name: _value_stats(out[name])
                              for name in ("sigma_pct", "mu_pct") if name in out}
    if all(k in out for k in ("alpha_nw", "alpha_c", "alpha_a")):
        summary["blend"] = blend_diagnostics(out["alpha_nw"], out["alpha_c"], out["alpha_a"],
                                             verbose=verbose)
    if label_key in out:
        labels = np.asarray(out[label_key]).astype(np.int64)
        uniq, counts = np.unique(labels, return_counts=True)
        names = list(class_names) if class_names is not None else []
        summary["class_counts"] = {
            (names[u] if 0 <= u < len(names) else f"label_{u}"): int(c)
            for u, c in zip(uniq, counts)}
    if "laminar" in out:
        lam = np.asarray(out["laminar"], dtype=bool)
        if np.any(lam):
            summary["switch"] = {
                "n_laminar": int(np.sum(lam)), "n_turbulent": int(np.sum(~lam)),
                "frac_laminar": float(np.mean(lam)),
                "laminar": _subset_stats(out, lam, error_fields),
                "turbulent": _subset_stats(out, ~lam, error_fields),
            }
    if extra:
        summary.update(extra)

    if verbose:
        print("=" * 80)
        print(f"Percent-error statistics (N={n:,}):")
        print(f"  {'field':<18}{'mean|e|':>10}{'median|e|':>11}{'p90|e|':>10}"
              f"{'<=10%':>8}{'<=20%':>8}")
        for name, st in summary["errors"].items():
            print(f"  {name:<18}{st['mean_abs']:>10.3f}{st['median_abs']:>11.3f}"
                  f"{st['p90_abs']:>10.3f}{st['frac_le_10']:>8.3f}{st['frac_le_20']:>8.3f}")
    return summary


def write_summary(path: str, summary: Mapping[str, Any]) -> str:
    """Write ``summary`` as indented JSON (numpy types converted, NaN -> null).

    Args:
        path: Output file (its directory is created).
        summary: Summary dict.

    Returns:
        The path written.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2)
        handle.write("\n")
    print(f"Saved: {path}")
    return path
