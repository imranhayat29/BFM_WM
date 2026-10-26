"""
Laminar-turbulent switching helpers for the lam-turb BFM wall model.

A point is laminar when its Kt/Km sensor (fluctuating / mean kinetic energy) is at or
below a threshold, the CharLES BFMv2 convention (``kt_km <= KT_KM_THRESHOLD``). Both
branches (baseline, error model, SNGP) are evaluated on every point and the results are
hard-switched per point. Model loading (one bundle per branch) lives in
``src_err/pipeline.py``.
"""
from typing import Sequence, Tuple

import numpy as np

DEFAULT_KTKM_THRESHOLD = 0.005  # CharLES BFMv2 lam-turb inputs; LT-NB used 0.002


def laminar_mask(kt_km: np.ndarray, threshold: float = DEFAULT_KTKM_THRESHOLD) -> np.ndarray:
    """Laminar mask ``kt_km <= threshold`` (CharLES convention; LT-NB mixed ``<`` and ``<=``).

    Args:
        kt_km: Per-point Kt/Km values (column or name inference, see ``data.ktkm_values``).
        threshold: Kt/Km threshold.

    Returns:
        Boolean mask, True on laminar points. NaN/inf values are turbulent (False).
    """
    k = np.asarray(kt_km, dtype=np.float64)
    return np.isfinite(k) & (k <= threshold)


def hard_switch(lam_vals: np.ndarray, turb_vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-point hard switch: ``lam_vals`` where ``mask`` else ``turb_vals``.

    Args:
        lam_vals: Laminar-branch values, shape (N, ...).
        turb_vals: Turbulent-branch values, same shape.
        mask: Laminar mask, shape (N,); broadcast over trailing axes.

    Returns:
        The switched array.
    """
    lam_vals = np.asarray(lam_vals)
    turb_vals = np.asarray(turb_vals)
    if lam_vals.shape != turb_vals.shape:
        raise ValueError(f"hard_switch: shape mismatch {lam_vals.shape} vs {turb_vals.shape}")
    m = np.asarray(mask, dtype=bool)
    if m.shape[0] != lam_vals.shape[0]:
        raise ValueError(f"hard_switch: mask length {m.shape[0]} != {lam_vals.shape[0]}")
    m = m.reshape(m.shape + (1,) * (lam_vals.ndim - m.ndim))
    return np.where(m, lam_vals, turb_vals)


def composite_labels(lam_labels: np.ndarray,
                     turb_labels: np.ndarray,
                     mask: np.ndarray,
                     turb_names: Sequence[str],
                     lam_names: Sequence[str]) -> Tuple[np.ndarray, list]:
    """Composite class labels and names of the two branch SNGPs.

    Turbulent classes come first (labels unchanged), then the laminar classes offset by
    ``len(turb_names)`` (LT-NB cell 65). Labels must already be restricted to each
    branch's trained (or post-hoc merged) classes, so the two ranges cannot collide; the
    OOD class, if any, is appended by the caller at index ``len(names)``.

    Args:
        lam_labels: Laminar-branch labels (indices into ``lam_names``).
        turb_labels: Turbulent-branch labels (indices into ``turb_names``).
        mask: Laminar mask.
        turb_names: Turbulent class names.
        lam_names: Laminar class names.

    Returns:
        ``(labels, names)`` with ``names = turb_names + lam_names``.
    """
    lam = np.asarray(lam_labels, dtype=np.int64)
    turb = np.asarray(turb_labels, dtype=np.int64)
    n_turb, n_lam = len(turb_names), len(lam_names)
    m = np.asarray(mask, dtype=bool)
    # Out-of-range labels would silently alias the other branch's classes
    if np.any(m & ((lam < 0) | (lam >= n_lam))):
        raise ValueError(f"composite_labels: laminar labels outside [0, {n_lam})")
    if np.any(~m & ((turb < 0) | (turb >= n_turb))):
        raise ValueError(f"composite_labels: turbulent labels outside [0, {n_turb})")
    labels = hard_switch(lam + n_turb, turb, m)
    return labels, list(turb_names) + list(lam_names)
