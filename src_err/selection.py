"""
Selection of one wall-normal station per x location, for line plots (C_f vs x).

Many samples share the same x and differ in the wall-normal distance n1; the line plots
use one representative point per x. Ported from ``BFM_WM/src_err/select_by_x_wallnormal.py``
(same results, vectorised per x group). Modes:
    - 'raw':        the point whose n1 is closest to a target n1;
    - 'normalized': target = min(n1) + p (max(n1) - min(n1)) per x, p in [0, 1];
    - 'yplus':      the point whose y+ = n1 |u_tau| / nu is closest to a target y+;
    - piecewise raw: different raw targets upstream/downstream of an x transition,
      with an optional linear ramp.

The 'yplus' mode uses the dataset u_tau (column 3 of U). This only picks the plotting
stations, never a prediction, so it does not leak truth into the predictions.
"""
from typing import Optional, Tuple

import numpy as np

SELECTION_MODES = ("raw", "normalized", "yplus")


def _select_closest(x: np.ndarray,
                    wall_normal: np.ndarray,
                    target: Optional[float],
                    p: Optional[float],
                    round_decimals: int,
                    target_of_x=None) -> Tuple[np.ndarray, np.ndarray]:
    """Pick, per rounded x, the point whose wall-normal value is closest to the target.

    Exactly one of ``target`` (absolute), ``p`` (normalised) or ``target_of_x`` (callable
    of the rounded x values) is used. Ties break to the first occurrence in the input
    order, as ``np.argmin`` in the original per-group loop.

    Returns:
        ``(x_unique, selected_indices)`` sorted by x; indices refer to the input arrays.
    """
    x = np.asarray(x, dtype=np.float64)
    wn = np.asarray(wall_normal, dtype=np.float64)
    if x.shape != wn.shape:
        raise ValueError("x and the wall-normal coordinate must have the same shape")
    finite = np.isfinite(x) & np.isfinite(wn)
    if not np.any(finite):
        raise ValueError("No finite (x, wall-normal) pairs found")
    orig = np.flatnonzero(finite)
    xk = np.round(x[finite], int(round_decimals))
    wn = wn[finite]

    x_unique, group = np.unique(xk, return_inverse=True)
    if target_of_x is not None:
        tgt = np.asarray(target_of_x(x_unique), dtype=np.float64)[group]
    elif p is not None:
        n_groups = x_unique.size
        wn_min = np.full(n_groups, np.inf)
        wn_max = np.full(n_groups, -np.inf)
        np.minimum.at(wn_min, group, wn)
        np.maximum.at(wn_max, group, wn)
        tgt = (wn_min + float(p) * (wn_max - wn_min))[group]
    else:
        tgt = np.full(wn.shape, float(target))

    dist = np.abs(wn - tgt)
    # Sort by (group, distance, input position): the first row of each group is argmin
    order = np.lexsort((orig, dist, group))
    first = np.ones(order.size, dtype=bool)
    first[1:] = group[order[1:]] != group[order[:-1]]
    return x_unique, orig[order[first]]


def select_n1_per_x(x: np.ndarray,
                    U: np.ndarray,
                    mode: str = "raw",
                    value: float = 0.001,
                    round_decimals: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one n1 station per x.

    Args:
        x: x (or x/c) per sample.
        U: (N, >=4) array ``[n1, u1, nu, u_tau, ...]`` (the first ``unnormalized_inputs``
            columns).
        mode: 'raw' (value = target n1), 'normalized' (value = p in [0, 1]) or 'yplus'
            (value = target y+ = n1 |u_tau| / nu, dataset u_tau: plotting only).
        value: Target as described for ``mode``.
        round_decimals: x values are grouped by ``np.round(x, round_decimals)``.

    Returns:
        ``(x_sel, n1_sel, idx_sel)``: sorted unique x, the selected n1 and the indices of
        the selected points in the input arrays.
    """
    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError("U must be a 2D array")
    if mode == "raw":
        x_sel, idx = _select_closest(x, U[:, 0], float(value), None, round_decimals)
    elif mode == "normalized":
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("value must be within [0, 1] in 'normalized' mode")
        x_sel, idx = _select_closest(x, U[:, 0], None, float(value), round_decimals)
    elif mode == "yplus":
        # Magnitude: the dataset u_tau is signed in some files (reversed flow)
        yplus = U[:, 0] * np.abs(U[:, 3]) / U[:, 2]
        x_sel, idx = _select_closest(x, yplus, float(value), None, round_decimals)
    else:
        raise ValueError(f"mode must be one of {SELECTION_MODES}, got {mode!r}")
    return x_sel, U[idx, 0], idx


def select_n1_per_x_piecewise(x: np.ndarray,
                              U: np.ndarray,
                              upstream: float,
                              downstream: float,
                              x_transition: float,
                              ramp_width: float = 0.0,
                              round_decimals: int = 12
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select one n1 station per x with an x-dependent raw n1 target.

    Upstream of ``x_transition`` the target is ``upstream``, downstream it is
    ``downstream``; with ``ramp_width > 0`` the target is interpolated linearly over
    ``[x_transition - ramp_width/2, x_transition + ramp_width/2]``, otherwise it switches
    abruptly (``x < x_transition`` is upstream).

    Args:
        x: x (or x/c) per sample.
        U: (N, >=1) array whose column 0 is n1.
        upstream: Target n1 upstream.
        downstream: Target n1 downstream.
        x_transition: Transition x.
        ramp_width: Width of the linear ramp in x (>= 0).
        round_decimals: x grouping rounding.

    Returns:
        ``(x_sel, n1_sel, idx_sel)`` as ``select_n1_per_x``.
    """
    U = np.asarray(U)
    if U.ndim != 2:
        raise ValueError("U must be a 2D array")
    if ramp_width < 0.0:
        raise ValueError("ramp_width must be >= 0")
    up, down, xt, w = float(upstream), float(downstream), float(x_transition), float(ramp_width)

    def target_of_x(xu: np.ndarray) -> np.ndarray:
        if w == 0.0:
            return np.where(xu < xt, up, down)
        x0, x1 = xt - 0.5 * w, xt + 0.5 * w
        alpha = np.clip((xu - x0) / (x1 - x0), 0.0, 1.0)
        return np.where(xu <= x0, up, np.where(xu >= x1, down, up + alpha * (down - up)))

    x_sel, idx = _select_closest(x, U[:, 0], None, None, round_decimals, target_of_x)
    return x_sel, U[idx, 0], idx


def select_stations(x: np.ndarray, U: np.ndarray, opts: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch on a ``[plots.cf_lines]`` table.

    Args:
        x: x per sample.
        U: Unnormalised inputs ``[n1, u1, nu, u_tau, ...]``.
        opts: ``Selection`` ('raw' | 'normalized' | 'yplus' | 'raw_piecewise') with
            ``Value``, or ``Upstream``/``Downstream``/``XTransition``/``RampWidth``.

    Returns:
        ``(x_sel, n1_sel, idx_sel)``.
    """
    sel = opts.get("Selection", "raw")
    if sel == "raw_piecewise":
        return select_n1_per_x_piecewise(x, U, opts["Upstream"], opts["Downstream"],
                                         opts["XTransition"], opts.get("RampWidth", 0.0))
    return select_n1_per_x(x, U, mode=sel, value=opts.get("Value", 0.001))
