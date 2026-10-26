"""
Shared a-priori prediction pipeline of the BFM wall model with error model and SNGP confidence

Used by test_turb.py, test_lamturb.py and test_turb_allcases.py. One pass gives, per point:

    baseline   y_b = |NN_b(pi)|                       (raw output kept as y_base_raw)
    error      (mu, sigma) = error model(pi)          (pi_out units)
    corrected  y_corr = |y_b,raw + mu|                (always-corrected candidate)
    confidence C = 1 - H(q)/ln(K - K_tr + 1)          (un-merged SNGP probabilities)
    fallbacks  EQWM (turbulent points) / laminar law sqrt(|pi_1|) (laminar points)
    blend      y_final = 3-switch blend (aleatoric, confidence, near-wall y+), CharLES nesting

No true-target information enters any prediction (user principle): the predictions are
computed from the network inputs X, |pi_1| and (lam-turb) the Kt/Km sensor only. The
dataset targets (`arrays.y`) are read only in `evaluate`, which adds the percent errors
e_*_pct (evaluation, truth-normalised blend); the dataset u_tau (`arrays.U[:, 3]`) is not
read at all.
All quantities are magnitudes (dataset targets are |u_tau| n1/nu >= 0).
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from src_err import classes as cls
from src_err import gating
from src_err.data import DatasetArrays, baseline_output, ktkm_values, load_baseline_model
from src_err.evaluation import percent_error, predicted_percent
from src_err.lamturb import composite_labels, hard_switch, laminar_mask

OOD_NAME = "OOD"


# ---------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------


@dataclass
class BranchModels:
    """
    The three networks of one wall-model branch (turbulent, or laminar in lam-turb)

    Attributes:
        baseline: WallModel (parent src/wall_model.py) in eval mode
        error: ErrorBundle (src_err/error_model.py)
        sngp: SngpBundle (src_err/sngp.py)
        posthoc_groups: {group: [trained class names]} for labels/plots (never the confidence)
        display_names: {class or group name: display name}
        paths: resolved model paths {'baseline', 'sngp', 'error'}
        branch: 'turb' or 'lam' (log messages and summary keys)
    """
    baseline: Any
    error: Any
    sngp: Any
    posthoc_groups: Dict[str, List[str]] = field(default_factory=dict)
    display_names: Dict[str, str] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    branch: str = "turb"


def load_branch_models(models_cfg: Mapping[str, str], classes_cfg: Optional[Mapping[str, Any]],
                       device: Any, branch: str = "turb") -> BranchModels:
    """
    Load the baseline, error model and SNGP of one branch

    Post-hoc groups and display names come from the test TOML `[classes]` when set there,
    otherwise from the SNGP metadata (`bfmwm_v2` block; empty for the shipped legacy models).

    Args:
        models_cfg: `[models]` table (Baseline, Sngp, Error; names under ./models or paths)
        classes_cfg: `[classes]` table (PosthocGroups, DisplayNames; None entries = metadata)
        device: torch device of the baseline
        branch: 'turb' or 'lam'

    Returns:
        BranchModels
    """
    from src_err.error_model import load_error_model
    from src_err.sngp import load_sngp
    from src_err.utils import resolve_model_path

    print("=" * 80)
    print(f"Loading the {branch} branch models")
    baseline = load_baseline_model(models_cfg["Baseline"], device)
    error = load_error_model(models_cfg["Error"])
    sngp = load_sngp(models_cfg["Sngp"])
    classes_cfg = classes_cfg or {}
    groups = classes_cfg.get("PosthocGroups")
    names = classes_cfg.get("DisplayNames")
    groups_src = "TOML" if groups is not None else "SNGP metadata"
    names_src = "TOML" if names is not None else "SNGP metadata"
    groups = dict(sngp.posthoc_groups if groups is None else groups)
    names = dict(sngp.display_names if names is None else names)
    # Validate the group members against the trained classes now (clear error before inference)
    cls.posthoc_merge(np.zeros((1, sngp.num_classes)), sngp.class_map_trained, groups)
    models = BranchModels(baseline=baseline, error=error, sngp=sngp, posthoc_groups=groups,
                          display_names=names, branch=branch,
                          paths={"baseline": resolve_model_path(models_cfg["Baseline"]),
                                 "sngp": sngp.path, "error": error.path})
    print(f"  post-hoc groups ({groups_src}): {groups or 'none'}")
    print(f"  display names   ({names_src}): {names or 'none'}")
    print(f"  plot classes    : {posthoc_class_names(models)}")
    return models


def _posthoc_mapping(models: BranchModels) -> Tuple[np.ndarray, cls.ClassMap]:
    """Trained label -> post-hoc label and the merged class map (by name, cls.posthoc_merge)."""
    sngp = models.sngp
    _, merged_map, mapping = cls.posthoc_merge(np.zeros((1, sngp.num_classes)), sngp.class_map_trained,
                                               models.posthoc_groups)
    return mapping, merged_map


def posthoc_class_names(models: BranchModels) -> List[str]:
    """
    Display names of the post-hoc merged classes (index = `labels_posthoc`)

    Args:
        models: BranchModels

    Returns:
        Names in label order (groups first, then the ungrouped trained classes)
    """
    _, merged_map = _posthoc_mapping(models)
    return cls.display_names(merged_map, models.display_names) if models.display_names else list(merged_map.names)


def class_names_for_plot(turb: BranchModels, lam: Optional[BranchModels] = None) -> List[str]:
    """
    Class names of the label maps: post-hoc merged display names (+ laminar ones) + 'OOD'

    The label index of a point is `labels_posthoc` (turbulent-only) or `labels_composite`
    (lam-turb: turbulent classes first, then laminar); OOD points (C below the confidence
    threshold) get the last index.

    Args:
        turb: Turbulent models
        lam: Laminar models (lam-turb), or None

    Returns:
        Names in label order, ending with 'OOD'
    """
    names = posthoc_class_names(turb)
    if lam is not None:
        names = names + posthoc_class_names(lam)
    return names + [OOD_NAME]


# ---------------------------------------------------------------------------------------
# Inference building blocks
# ---------------------------------------------------------------------------------------


def pi1_abs_from(arrays: DatasetArrays, model_config: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    """
    |pi_1| = |u1| n1 / nu of every point, from the unnormalised inputs U = [n1, u1, nu, ...]

    This is what CharLES computes from the LES velocity at the matching point; it is an
    inference input (never u_tau). It is not taken from the stored network input X[:, 0]
    because some local data files store inconsistent `inputs` tables (see
    `input_consistency`); the networks still receive the stored X they were trained on.

    Args:
        arrays: DatasetArrays
        model_config: Unused (kept for API symmetry with the input-column selection)

    Returns:
        (N,) float64
    """
    U = np.asarray(arrays.U, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(U[:, 1] * U[:, 0] / U[:, 2])


def input_consistency(arrays: DatasetArrays, model_config: Mapping[str, Any], rtol: float = 1e-4,
                      verbose: bool = True) -> Dict[str, float]:
    """
    Fraction of points per dataset whose stored network input pi_1 = X[:, 0] differs from u1 n1/nu

    Only checked for InputScaling 2 without near-wall mode (X[:, 0] = u1_y_over_nu). A
    non-zero fraction means the h5 `inputs` table is inconsistent with its
    `unnormalized_inputs` (the networks then see different inputs than CharLES would).

    Args:
        arrays: DatasetArrays
        model_config: Baseline checkpoint config
        rtol: Relative tolerance
        verbose: Print a WARNING per inconsistent dataset

    Returns:
        {dataset key: inconsistent fraction} (empty if not applicable)
    """
    scaling = model_config.get("model", {}).get("inputs", {}).get("InputScaling", 1)
    near_wall = float(model_config.get("data", {}).get("near_wall_threshold", 0.0) or 0.0)
    if scaling != 2 or near_wall > 0.0:
        return {}
    U = np.asarray(arrays.U, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ref = U[:, 1] * U[:, 0] / U[:, 2]
    x0 = np.asarray(arrays.X[:, 0], dtype=np.float64)
    bad = ~np.isclose(x0, ref, rtol=rtol, atol=1e-12)
    res = {}
    for key in arrays.keys():
        m = arrays.dataset_key == key
        res[key] = float(np.mean(bad[m]))
        if verbose and res[key] > 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.nanmedian(x0[m & bad] / ref[m & bad])
            print(f"WARNING: dataset {key}: stored input u1_y_over_nu != u1*n1/nu on {100 * res[key]:.1f} % "
                  f"of the points (median ratio {ratio:.3g}); the networks get the stored inputs, "
                  f"|pi_1| for the EQWM/laminar law/low-value blends uses u1*n1/nu")
    return res


def branch_predictions(models: BranchModels, X: np.ndarray, cfg: Mapping[str, Any],
                       verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Baseline, error-model and SNGP predictions of one branch on all points

    Args:
        models: BranchModels
        X: (N, D) network inputs
        cfg: Test config (`[confidence].Covariance`)
        verbose: Print timings

    Returns:
        {'y_base_raw', 'y_base', 'mu', 'sigma', 'y_corr', 'confidence', 'labels',
         'labels_posthoc'} (+ 'gp_variance' in GP-covariance mode)
    """
    from src_err.error_model import predict_error
    from src_err.sngp import predict as sngp_predict

    t0 = time.time()
    out: Dict[str, np.ndarray] = {}
    out["y_base_raw"] = baseline_output(models.baseline, X)
    out["y_base"] = np.abs(out["y_base_raw"])
    t1 = time.time()
    out["mu"], out["sigma"] = predict_error(models.error, X)
    out["y_corr"] = gating.corrected_piout(out["y_base_raw"], out["mu"])
    t2 = time.time()
    covariance = cfg.get("confidence", {}).get("Covariance", "none")
    sp = sngp_predict(models.sngp, X, covariance=covariance)
    out["confidence"] = np.asarray(sp["confidence"], dtype=np.float64)
    out["labels"] = np.asarray(sp["labels"], dtype=np.int64)
    if "variance" in sp:
        out["gp_variance"] = np.asarray(sp["variance"], dtype=np.float64)
    mapping, _ = _posthoc_mapping(models)
    out["labels_posthoc"] = mapping[out["labels"]]
    if verbose:
        print(f"[{models.branch}] baseline {t1 - t0:.1f} s, error model {t2 - t1:.1f} s, "
              f"SNGP ({covariance}) {time.time() - t2:.1f} s on {len(X):,} points")
    return out


def _gate_and_blend(out: Dict[str, np.ndarray], pi1: np.ndarray, cfg: Mapping[str, Any]) -> None:
    """sigma%/mu% (baseline-normalised), the aleatoric gate metric and the 3-switch blend (in place)."""
    gates = cfg["gates"]
    low = gates["LowValue"]
    ale = gates["Aleatoric"]
    out["sigma_pct"] = predicted_percent(out["sigma"], out["y_base"], pi1, low["Threshold"], low["Delta"])
    out["mu_pct"] = predicted_percent(out["mu"], out["y_base"], pi1, low["Threshold"], low["Delta"])
    if ale.get("Type", "percentage") == "percentage":
        out["gate_metric"] = out["sigma_pct"].copy()
    else:
        out["gate_metric"] = gating.aleatoric_metric(out["sigma"], out["y_base"], pi1, mode=ale["Type"],
                                                     mu=out["mu"])
    y_final, a_nw, a_c, a_a = gating.blend(out["y_base"], out["y_corr"], out["y_conf_fb"], out["y_nw_fb"],
                                           out["yplus_nw"], out["confidence"], out["gate_metric"], gates)
    out.update(y_final=y_final, alpha_nw=a_nw, alpha_c=a_c, alpha_a=a_a)


def evaluate(out: Dict[str, np.ndarray], y_true: np.ndarray, pi1: np.ndarray,
             cfg: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """
    Add the evaluation fields (truth used here only)

    Percent errors with the `[metrics.LowValue]` blend on |y_true|: e_base_pct (|y_b|),
    e_final_pct (blended y_final), e_corr_pct (always corrected) and e_eqwm_pct (EQWM).

    Args:
        out: Prediction dict (modified in place)
        y_true: Dataset pi_out
        pi1: |pi_1|
        cfg: Test config

    Returns:
        `out`
    """
    low = cfg["metrics"]["LowValue"]
    thr, dlt = low["Threshold"], low["Delta"]
    y_true = np.abs(np.asarray(y_true, dtype=np.float64))
    out["y_true"] = y_true
    out["e_base_pct"] = percent_error(out["y_base"], y_true, pi1, thr, dlt)
    out["e_final_pct"] = percent_error(out["y_final"], y_true, pi1, thr, dlt)
    out["e_corr_pct"] = percent_error(out["y_corr"], y_true, pi1, thr, dlt)
    out["e_eqwm_pct"] = percent_error(out["y_eqwm"], y_true, pi1, thr, dlt)
    return out


# ---------------------------------------------------------------------------------------
# Full pipelines
# ---------------------------------------------------------------------------------------


def predict_turb(models: BranchModels, arrays: DatasetArrays, cfg: Mapping[str, Any],
                 verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Turbulent-only pipeline (test_turb.py, test_turb_allcases.py); all points are turbulent

    Args:
        models: BranchModels
        arrays: DatasetArrays (X, U, y, ...)
        cfg: Test config ([eqwm], [confidence], [gates], [metrics])
        verbose: Print timings and blend diagnostics

    Returns:
        Per-point dict: y_true (evaluation only), y_base, y_base_raw, mu, sigma, y_corr,
        y_eqwm, yplus_nw, y_conf_fb, y_nw_fb, confidence, labels, labels_posthoc,
        sigma_pct, mu_pct, gate_metric, alpha_a, alpha_c, alpha_nw, y_final, laminar (all
        False), kt_km (NaN), e_base_pct, e_final_pct, e_corr_pct, e_eqwm_pct
    """
    pi1 = pi1_abs_from(arrays)
    input_consistency(arrays, models.baseline.config, verbose=verbose)
    out = branch_predictions(models, arrays.X, cfg, verbose=verbose)
    out["y_eqwm"] = gating.eqwm_piout(pi1, law=cfg["eqwm"]["Law"])
    out["y_conf_fb"], out["y_nw_fb"], out["yplus_nw"] = gating.fallbacks(pi1, None, y_eqwm=out["y_eqwm"])
    out["laminar"] = np.zeros(len(pi1), dtype=bool)
    out["kt_km"] = np.full(len(pi1), np.nan)
    _gate_and_blend(out, pi1, cfg)
    return evaluate(out, arrays.y, pi1, cfg)


def _check_same_inputs(lam: BranchModels, turb: BranchModels) -> None:
    """Both branches must read the same input columns (the arrays are loaded once)."""
    def key(m):
        c = m.baseline.config
        return (c.get("model", {}).get("inputs", {}).get("InputScaling", 1),
                float(c.get("data", {}).get("near_wall_threshold", 0.0) or 0.0))
    if key(lam) != key(turb):
        raise ValueError(f"Laminar and turbulent baselines use different inputs (InputScaling, "
                         f"near_wall_threshold) {key(lam)} vs {key(turb)}; not supported")


def predict_lamturb(lam: BranchModels, turb: BranchModels, arrays: DatasetArrays, cfg: Mapping[str, Any],
                    verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Laminar-turbulent pipeline (test_lamturb.py)

    Both branches (baseline, error model, SNGP, each with its own normalisation) run on all
    points and are hard-switched with the Kt/Km sensor (laminar iff kt_km <= [switch]
    KtKmThreshold; column or name inference). Laminar points use the laminar law
    sqrt(|pi_1|) (no dp/ds) for both fallbacks and as the y+ of the near-wall switch;
    turbulent points use the EQWM (CharLES BFMv2).

    Args:
        lam: Laminar models
        turb: Turbulent models
        arrays: DatasetArrays
        cfg: Test config (+ [switch])
        verbose: Print timings

    Returns:
        The predict_turb keys plus laminar (mask), kt_km, labels_composite and
        composite_names (str array of the composite class names, length K_turb' + K_lam').
        `labels` / `labels_posthoc` are also composite (turbulent first, laminar offset by
        the number of turbulent classes of that numbering).
    """
    _check_same_inputs(lam, turb)
    pi1 = pi1_abs_from(arrays)
    input_consistency(arrays, turb.baseline.config, verbose=verbose)
    kt = ktkm_values(arrays)
    mask = laminar_mask(kt, cfg["switch"]["KtKmThreshold"])
    if verbose:
        print(f"Kt/Km switch (kt_km <= {cfg['switch']['KtKmThreshold']}): {int(mask.sum()):,} laminar, "
              f"{int((~mask).sum()):,} turbulent points")
    bt = branch_predictions(turb, arrays.X, cfg, verbose=verbose)
    bl = branch_predictions(lam, arrays.X, cfg, verbose=verbose)
    out: Dict[str, np.ndarray] = {}
    for k in ("y_base_raw", "y_base", "mu", "sigma", "y_corr", "confidence"):
        out[k] = hard_switch(bl[k], bt[k], mask)
    if "gp_variance" in bt:
        out["gp_variance"] = hard_switch(bl["gp_variance"], bt["gp_variance"], mask)
    # Un-merged trained labels (turbulent K_tr first, then laminar) and the post-hoc composite
    out["labels"], _ = composite_labels(bl["labels"], bt["labels"], mask,
                                        turb.sngp.class_names, lam.sngp.class_names)
    names_t, names_l = posthoc_class_names(turb), posthoc_class_names(lam)
    out["labels_composite"], comp_names = composite_labels(bl["labels_posthoc"], bt["labels_posthoc"], mask,
                                                           names_t, names_l)
    out["labels_posthoc"] = out["labels_composite"]
    out["composite_names"] = np.asarray(comp_names, dtype=str)
    out["y_eqwm"] = gating.eqwm_piout(pi1, law=cfg["eqwm"]["Law"])
    out["y_conf_fb"], out["y_nw_fb"], out["yplus_nw"] = gating.fallbacks(pi1, mask, y_eqwm=out["y_eqwm"])
    out["laminar"] = mask
    out["kt_km"] = kt
    _gate_and_blend(out, pi1, cfg)
    return evaluate(out, arrays.y, pi1, cfg)


def ood_and_nearwall_masks(out: Mapping[str, np.ndarray], gates: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot masks of the fallback regions (no side effects on labels or class lists)

    Args:
        out: Pipeline output
        gates: `[gates]` table

    Returns:
        (ood = C < Confidence.Threshold, nearwall = yplus_nw < Yplus.Threshold)
    """
    ood = np.asarray(out["confidence"]) < float(gates["Confidence"]["Threshold"])
    nearwall = np.abs(np.asarray(out["yplus_nw"])) < float(gates["Yplus"]["Threshold"])
    return ood, nearwall


def per_point_arrays(out: Mapping[str, np.ndarray], n: int) -> Dict[str, np.ndarray]:
    """
    The per-point (length-n) numeric fields of a pipeline output (for arrays.npz)

    Args:
        out: Pipeline output
        n: Number of points

    Returns:
        {name: array}
    """
    return {k: np.asarray(v) for k, v in out.items()
            if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n and v.dtype.kind in "biuf"}
