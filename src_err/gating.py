"""
Inference-time quantities and the blended (3-switch) correction of the BFM wall model.

Everything here works on numpy arrays in pi_out units (pi_out = u_tau * n1 / nu), as
magnitudes. Two identities are used throughout:
    - pi_1 = u1 * n1 / nu = u+ * y+ (the first network input) and pi_out = y+, so every
      algebraic wall law u+ = f(y+) becomes the scalar equation |pi_1| = y+ * f(y+);
    - y+ computed with a u_tau estimate equals the pi_out of that estimate.

The ramps, the EQWM law, the corrected candidate, the low-value blend of the aleatoric
percent and the nesting of the three switches mirror CharLES BFMv2
(src/charles/cti_solver_ig_bcs_kernels.hpp, src/charles/wm/AlgebraicWM_spalding.hpp).
The network inputs still differ on reversed-flow points: the datasets use a global-x sign
frame, CharLES a local u2-positive frame (README, To do). No function here may use
true-target information.
"""
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np

# Wall-law constants (shared by CharLES and the parent src/baseline_models.py)
KAPPA = 0.41
B = 5.2
# CharLES AlgebraicWM_spalding: y* solves y+ = ln(y+)/kappa + B (two-layer law)
CHARLES_YSTAR = 11.0623
# Parent src/baseline_models.py: matching point and near-wall coefficient
PARENT_YSTAR = 23.0
PARENT_A = 0.5 * (1.0 / (KAPPA * PARENT_YSTAR) - 1.0) / PARENT_YSTAR

EQWM_LAWS = ("charles", "parent")
ALEATORIC_MODES = ("percentage", "ratio")

# Floor of every percent denominator (same value as CharLES and the notebooks)
_EPS_DENOM = 1.0e-12


# =============================================================================
# Ramps (CharLES smoothBlendAboveThreshold / smoothBlendBelowThreshold)
# =============================================================================

def ramp_above(metric: np.ndarray, threshold: float, delta: float) -> np.ndarray:
    """Linear ramp from 0 (metric <= threshold - delta) to 1 (metric >= threshold + delta).

    Identical to CharLES ``smoothBlendAboveThreshold``: ``delta <= 0`` gives the hard
    switch ``metric > threshold``; NaN metrics give NaN (ramp) or 0 (hard switch), as in
    CharLES.

    Args:
        metric: Per-point metric.
        threshold: Centre of the ramp.
        delta: Half-width of the ramp.

    Returns:
        Weights in [0, 1] (float64, same shape as ``metric``).
    """
    m = np.asarray(metric, dtype=np.float64)
    if delta <= 0.0:
        return (m > threshold).astype(np.float64)
    return np.clip((m - (threshold - delta)) / (2.0 * delta), 0.0, 1.0)


def ramp_below(metric: np.ndarray, threshold: float, delta: float) -> np.ndarray:
    """Linear ramp from 1 (metric <= threshold - delta) to 0 (metric >= threshold + delta).

    Identical to CharLES ``smoothBlendBelowThreshold``: ``delta <= 0`` gives the hard
    switch ``metric < threshold``.

    Args:
        metric: Per-point metric.
        threshold: Centre of the ramp.
        delta: Half-width of the ramp.

    Returns:
        Weights in [0, 1] (float64, same shape as ``metric``).
    """
    m = np.asarray(metric, dtype=np.float64)
    if delta <= 0.0:
        return (m < threshold).astype(np.float64)
    return np.clip((threshold + delta - m) / (2.0 * delta), 0.0, 1.0)


# =============================================================================
# Equilibrium wall model in pi-space
# =============================================================================

def _newton_increasing(func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
                       target: np.ndarray,
                       lo: np.ndarray,
                       hi: np.ndarray,
                       x0: np.ndarray,
                       max_iter: int = 50,
                       rtol: float = 1.0e-10) -> np.ndarray:
    """Vectorised safeguarded Newton solve of ``g(x) = target`` for increasing ``g``.

    The bracket [lo, hi] (with g(lo) <= target <= g(hi)) is shrunk every iteration and a
    bisection step replaces any Newton step that leaves it, so convergence is guaranteed
    even where the Newton iteration alone would overshoot.

    Args:
        func: Returns ``(g(x), g'(x))`` for an array ``x``.
        target: Right-hand sides.
        lo: Lower bracket.
        hi: Upper bracket.
        x0: Initial guesses.
        max_iter: Maximum number of iterations.
        rtol: Relative step tolerance.

    Returns:
        The roots (float64).
    """
    lo = lo.astype(np.float64).copy()
    hi = hi.astype(np.float64).copy()
    x = np.clip(x0.astype(np.float64), lo, hi)
    active = np.ones(x.shape, dtype=bool)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        xa = x[idx]
        g, dg = func(xa)
        r = g - target[idx]
        lo[idx] = np.where(r < 0.0, xa, lo[idx])
        hi[idx] = np.where(r > 0.0, xa, hi[idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            step = r / dg
        x_newton = xa - step
        # Converged Newton steps are accepted even if they touch the bracket ends
        done = (r == 0.0) | (np.isfinite(step) & (np.abs(step) <= rtol * np.abs(xa)))
        bad = ~done & (~np.isfinite(x_newton) | (x_newton <= lo[idx]) | (x_newton >= hi[idx]))
        x_new = np.where(bad, 0.5 * (lo[idx] + hi[idx]), x_newton)
        x[idx] = np.where(r == 0.0, xa, x_new)
        active[idx[done]] = False
    return x


def _log_law_yplus(pi1: np.ndarray, max_iter: int, rtol: float) -> np.ndarray:
    """Root y+ of |pi_1| = y+ (ln(y+)/kappa + B) for |pi_1| > 0."""
    def func(y):
        ly = np.log(y) / KAPPA
        return y * (ly + B), ly + 1.0 / KAPPA + B

    # g(exp(-kappa B)) = 0 and g is increasing beyond exp(-1 - kappa B); g(max(1, pi1/B)) >= pi1
    lo = np.full(pi1.shape, np.exp(-KAPPA * B))
    hi = np.maximum(1.0, pi1 / B)
    return _newton_increasing(func, pi1, lo, hi, np.sqrt(pi1), max_iter, rtol)


def _parent_nearwall_yplus(pi1: np.ndarray, max_iter: int, rtol: float) -> np.ndarray:
    """Root y+ of |pi_1| = y+^2 + A y+^3 (parent near-wall layer, A < 0)."""
    # The cubic increases up to y+ = -2/(3A) (~34.3); the parent's fsolve, started at
    # sqrt(pi_1), converges to this first root.
    y_max = -2.0 / (3.0 * PARENT_A)
    g_max = y_max ** 2 + PARENT_A * y_max ** 3

    def func(y):
        return y * y + PARENT_A * y ** 3, 2.0 * y + 3.0 * PARENT_A * y * y

    target = np.minimum(pi1, g_max)
    lo = np.zeros(pi1.shape)
    hi = np.full(pi1.shape, y_max)
    return _newton_increasing(func, target, lo, hi, np.sqrt(pi1), max_iter, rtol)


def eqwm_piout(pi1_abs: np.ndarray,
               law: str = "charles",
               max_iter: int = 50,
               rtol: float = 1.0e-10) -> np.ndarray:
    """Equilibrium wall-model pi_out (= its y+) from |pi_1|, vectorised.

    ``law='charles'`` mirrors CharLES BFMv2 ``AlgebraicWM_spaldingf::solve_utau`` (the
    EQWM fallback of the blended correction): the linear branch y+ = sqrt(|pi_1|) is
    taken when sqrt(|pi_1|) < y* = 11.0623, else the log law |pi_1| = y+ (ln y+/kappa + B)
    is solved (kappa 0.41, B 5.2). CharLES iterates in float32 with a relaxed Newton
    (factor 0.4, residual tolerance 1e-5 in u+ units); here the same root is solved to
    ``rtol`` in float64.

    ``law='parent'`` mirrors ``src/baseline_models.log_law_solve`` (used by the
    notebooks' EQWM): solve the log law; where the resulting y+ < 23, re-solve the
    near-wall layer |pi_1| = y+^2 + A y+^3 with A = 0.5 (1/(23 kappa) - 1)/23.

    Args:
        pi1_abs: |pi_1| = |u1| n1 / nu (the absolute value is taken anyway).
        law: 'charles' or 'parent'.
        max_iter: Maximum Newton iterations.
        rtol: Relative tolerance on the y+ step.

    Returns:
        pi_out of the EQWM (float64, >= 0). |pi_1| = 0 gives 0; non-finite inputs give NaN.
    """
    if law not in EQWM_LAWS:
        raise ValueError(f"Unknown EQWM law {law!r}; expected one of {EQWM_LAWS}")
    pi1 = np.abs(np.asarray(pi1_abs, dtype=np.float64))
    out = np.full(pi1.shape, np.nan)
    finite = np.isfinite(pi1)
    out[finite & (pi1 == 0.0)] = 0.0
    pos = finite & (pi1 > 0.0)
    if not np.any(pos):
        return out
    p = pi1[pos]

    if law == "charles":
        y = np.sqrt(p)
        log_branch = y >= CHARLES_YSTAR  # CharLES: linear branch iff yp < ystar
        if np.any(log_branch):
            y[log_branch] = _log_law_yplus(p[log_branch], max_iter, rtol)
    else:
        y = _log_law_yplus(p, max_iter, rtol)
        near = y < PARENT_YSTAR
        if np.any(near):
            y[near] = _parent_nearwall_yplus(p[near], max_iter, rtol)

    out[pos] = y
    return out


def lamlaw_piout(pi1_abs: np.ndarray) -> np.ndarray:
    """Laminar-law pi_out (= its own y+): tau_w = mu |u1| / y1, so pi_out = sqrt(|pi_1|).

    No pressure-gradient term, as the CharLES BFMv2 laminar fallback.

    Args:
        pi1_abs: |pi_1|.

    Returns:
        sqrt(|pi_1|) (float64).
    """
    return np.sqrt(np.abs(np.asarray(pi1_abs, dtype=np.float64)))


def corrected_piout(y_base_raw: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Always-corrected candidate |y_base_raw + mu| (CharLES: |u_tau,b + eps_err|).

    The error model is trained on eps = y - y_base_raw with the raw baseline output, so
    y_base_raw + mu estimates y. The raw output is not always positive (linear output
    layer), and the notebooks' | |y_base_raw| + mu | added 2|y_base_raw| where it is negative.

    Args:
        y_base_raw: Raw baseline network output (pi_out units).
        mu: Predicted error mean (pi_out units).

    Returns:
        Corrected pi_out magnitude.
    """
    return np.abs(np.asarray(y_base_raw, dtype=np.float64) + np.asarray(mu, dtype=np.float64))


# =============================================================================
# Aleatoric gate metric
# =============================================================================

def lowvalue_percent(q: np.ndarray,
                     ref_abs: np.ndarray,
                     pi1_abs: np.ndarray,
                     thr_low: float,
                     delta_low: float) -> np.ndarray:
    """Percent of ``q`` relative to ``ref_abs``, blended with sqrt(|pi_1|) at low values.

    ``w = ramp_above(|ref|; thr_low, delta_low)`` and
    ``q% = w 100 q / max(|ref|, 1e-12) + (1 - w) 100 q / max(sqrt|pi_1|, 1e-12)``.
    A tiny reference (near separation/reattachment) must not inflate the percent, so
    below the ramp the laminar-law pi_out sqrt(|pi_1|) is the reference instead.
    The two percents (not the denominators) are blended, as in CharLES.

    Args:
        q: Quantity in pi_out units (signed values are kept).
        ref_abs: Reference pi_out magnitude.
        pi1_abs: |pi_1|.
        thr_low: Ramp threshold on |ref|.
        delta_low: Ramp half-width.

    Returns:
        Blended percent (float64).
    """
    q = np.asarray(q, dtype=np.float64)
    ref = np.abs(np.asarray(ref_abs, dtype=np.float64))
    w = ramp_above(ref, thr_low, delta_low)
    lin = np.sqrt(np.abs(np.asarray(pi1_abs, dtype=np.float64)))
    return (w * (100.0 * q / np.maximum(ref, _EPS_DENOM))
            + (1.0 - w) * (100.0 * q / np.maximum(lin, _EPS_DENOM)))


def aleatoric_metric(sigma: np.ndarray,
                     ref_abs: np.ndarray,
                     pi1_abs: np.ndarray,
                     mode: str = "percentage",
                     thr_low: float = 2.0,
                     delta_low: float = 1.0,
                     mu: Optional[np.ndarray] = None) -> np.ndarray:
    """Metric of the aleatoric switch.

    - ``'percentage'`` (default, = CharLES BFMv2): ``lowvalue_percent(|sigma|, |pi_out,b|)``,
      where ``ref_abs`` must be the baseline prediction (inference-time).
    - ``'ratio'`` (notebooks): ``|sigma| / (|mu| + 1e-12)``.

    Args:
        sigma: Predicted error standard deviation (pi_out units).
        ref_abs: |pi_out,baseline|.
        pi1_abs: |pi_1|.
        mode: 'percentage' or 'ratio'.
        thr_low: Low-value ramp threshold (percentage mode).
        delta_low: Low-value ramp half-width (percentage mode).
        mu: Predicted error mean (required for 'ratio').

    Returns:
        Per-point metric (float64).
    """
    s = np.abs(np.asarray(sigma, dtype=np.float64))
    if mode == "percentage":
        return lowvalue_percent(s, ref_abs, pi1_abs, thr_low, delta_low)
    if mode == "ratio":
        if mu is None:
            raise ValueError("aleatoric_metric(mode='ratio') needs mu")
        return s / (np.abs(np.asarray(mu, dtype=np.float64)) + _EPS_DENOM)
    raise ValueError(f"Unknown aleatoric metric mode {mode!r}; expected one of {ALEATORIC_MODES}")


# =============================================================================
# Fallbacks and the blended correction
# =============================================================================

def fallbacks(pi1_abs: np.ndarray,
              laminar_mask: Optional[np.ndarray] = None,
              eqwm_law: str = "charles",
              y_eqwm: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback predictions and the y+ of the near-wall switch (CharLES BFMv2 convention).

    Turbulent points use the EQWM for both the confidence and the near-wall fallback, with
    y+_nw = pi_out,EQWM. Laminar points use the laminar law sqrt(|pi_1|) for both (the EQWM
    is a turbulent log law above y+ 11.06), with its own y+ = sqrt(|pi_1|).

    Args:
        pi1_abs: |pi_1|.
        laminar_mask: True on laminar points (None = all turbulent).
        eqwm_law: EQWM law ('charles' or 'parent'), used if ``y_eqwm`` is not given.
        y_eqwm: Precomputed EQWM pi_out (avoids solving twice).

    Returns:
        ``(y_conf_fb, y_nw_fb, yplus_nw)``.
    """
    pi1 = np.abs(np.asarray(pi1_abs, dtype=np.float64))
    if y_eqwm is None:
        y_eqwm = eqwm_piout(pi1, law=eqwm_law)
    y_eqwm = np.asarray(y_eqwm, dtype=np.float64)
    if laminar_mask is None:
        return y_eqwm.copy(), y_eqwm.copy(), y_eqwm.copy()
    lam = np.asarray(laminar_mask, dtype=bool)
    y_fb = np.where(lam, lamlaw_piout(pi1), y_eqwm)
    return y_fb, y_fb.copy(), y_fb.copy()


def blend(y_base: np.ndarray,
          y_corr: np.ndarray,
          y_conf_fb: np.ndarray,
          y_nw_fb: np.ndarray,
          yplus_nw: np.ndarray,
          confidence: np.ndarray,
          metric: np.ndarray,
          gates: Mapping[str, Mapping[str, float]]
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three-switch blended correction with the CharLES BFMv2 nesting.

    ``inner  = a_a y_corr + (1 - a_a) y_base`` (aleatoric switch),
    ``middle = a_c inner  + (1 - a_c) y_conf_fb`` (SNGP-confidence switch),
    ``final  = a_nw middle + (1 - a_nw) y_nw_fb`` (near-wall y+ switch).

    Args:
        y_base: Baseline pi_out magnitude |y_b|.
        y_corr: Always-corrected candidate (``corrected_piout``).
        y_conf_fb: Low-confidence fallback.
        y_nw_fb: Near-wall fallback.
        yplus_nw: y+ of the near-wall switch (inference-time, ``fallbacks``).
        confidence: SNGP confidence C.
        metric: Aleatoric metric (``aleatoric_metric``).
        gates: The test-TOML ``[gates]`` table: ``Aleatoric``, ``Confidence`` and ``Yplus``,
            each with ``Threshold`` and ``Delta``.

    Returns:
        ``(y_final, alpha_nw, alpha_c, alpha_a)``.
    """
    ga, gc, gy = gates["Aleatoric"], gates["Confidence"], gates["Yplus"]
    alpha_a = ramp_below(metric, float(ga["Threshold"]), float(ga["Delta"]))
    alpha_c = ramp_above(confidence, float(gc["Threshold"]), float(gc["Delta"]))
    alpha_nw = ramp_above(np.abs(np.asarray(yplus_nw, dtype=np.float64)),
                          float(gy["Threshold"]), float(gy["Delta"]))
    inner = alpha_a * y_corr + (1.0 - alpha_a) * y_base
    middle = alpha_c * inner + (1.0 - alpha_c) * y_conf_fb
    y_final = alpha_nw * middle + (1.0 - alpha_nw) * y_nw_fb
    return y_final, alpha_nw, alpha_c, alpha_a


def blend_diagnostics(alpha_nw: np.ndarray,
                      alpha_c: np.ndarray,
                      alpha_a: np.ndarray,
                      verbose: bool = True,
                      tol: float = 1.0e-12) -> Dict[str, object]:
    """Counts and fractions of full / blended / fallback points of the three switches.

    Args:
        alpha_nw: Near-wall weights.
        alpha_c: Confidence weights.
        alpha_a: Aleatoric weights.
        verbose: Print a short table.
        tol: Tolerance for "exactly 0" / "exactly 1".

    Returns:
        Dict with one entry per switch (``full``, ``blended``, ``zero`` counts and fractions,
        ``mean``), plus ``n``, ``frac_fully_corrected`` (all three weights 1: y_final = y_corr)
        and ``frac_full_fallback`` (alpha_nw = 0 or alpha_c = 0: y_final is a fallback).
    """
    a_nw = np.asarray(alpha_nw, dtype=np.float64)
    a_c = np.asarray(alpha_c, dtype=np.float64)
    a_a = np.asarray(alpha_a, dtype=np.float64)
    n = int(a_nw.size)
    denom = max(n, 1)

    def _one(a):
        full = int(np.sum(a >= 1.0 - tol))
        zero = int(np.sum(a <= tol))
        blended = int(np.sum((a > tol) & (a < 1.0 - tol)))
        return {"full": full, "blended": blended, "zero": zero,
                "frac_full": full / denom, "frac_blended": blended / denom,
                "frac_zero": zero / denom,
                "mean": float(np.nanmean(a)) if n else float("nan")}

    diag = {
        "n": n,
        "near_wall": _one(a_nw),
        "confidence": _one(a_c),
        "aleatoric": _one(a_a),
        "frac_fully_corrected": float(np.sum((a_nw >= 1.0 - tol) & (a_c >= 1.0 - tol)
                                             & (a_a >= 1.0 - tol)) / denom),
        "frac_full_fallback": float(np.sum((a_nw <= tol) | (a_c <= tol)) / denom),
    }
    if verbose:
        nw, c, a = diag["near_wall"], diag["confidence"], diag["aleatoric"]
        print(f"3-switch blended correction (N={n:,}):")
        print(f"  Near-wall  : pass={nw['full']:,}, blended={nw['blended']:,}, "
              f"fallback={nw['zero']:,} (mean alpha {nw['mean']:.3f})")
        print(f"  Confidence : pass={c['full']:,}, blended={c['blended']:,}, "
              f"fallback={c['zero']:,} (mean alpha {c['mean']:.3f})")
        print(f"  Aleatoric  : corrected={a['full']:,}, blended={a['blended']:,}, "
              f"baseline={a['zero']:,} (mean alpha {a['mean']:.3f})")
        print(f"  Fully corrected: {100 * diag['frac_fully_corrected']:.2f} %, "
              f"full fallback: {100 * diag['frac_full_fallback']:.2f} %")
    return diag
