"""
A-priori test of the turbulent BFM wall model (baseline + error model + SNGP + EQWM) on all cases

    python test_turb_allcases.py test/test_turb_allcases.toml [--gpu N] [--output-dir DIR]
           [--max-samples N] [--datasets k1 k2 ...] [--scale linear|log|both] [--dry-run]
    python test_turb_allcases.py [<toml>] --replot RUN_DIR [--scale linear|log|both] [--output-dir DIR]

Port of the Error_vs_baseline_plotting notebooks (EVB, its log-log variant and the AIAA
always-vs-blended PDF) onto the shared v2 pipeline (src_err/pipeline.predict_turb, the same
code as test_turb.py). One pass per dataset gives the baseline, the blended (3-switch,
inference-only) and the always-corrected predictions and the EQWM; the dataset truth is
used only to select evaluation points (domain filter, outlier and error-change filters) and
to compute the percent errors. Per case (dataset, or a [groups] pool of datasets) the mean,
median and P90 of |e%| are aggregated; one marker per case is plotted at
(mean |e_base%|, mean |e_final%|).

Outputs in the output directory: cases.csv (one row per case, group member and skipped
dataset), overall.json (resolved config, model paths, git commit, versions, overall
averages), points.npz (per-point errors/confidence/weights with dataset tag and role) and
the figures <RunName>_<figure>[_log].<fmt>. `--replot RUN_DIR` regenerates every figure from
RUN_DIR/cases.csv + points.npz without importing torch or TensorFlow (plot options from the
given TOML, else from RUN_DIR/overall.json). `--dry-run` validates the TOML and resolves
every dataset file without loading any model.

Run from the repository root (TOMLs under ./inputfiles/, models under ./models/, data under ./data/).
"""
import argparse
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

import numpy as np

# Per-point fields kept for points.npz and the pooled PDFs (float32): pipeline key -> stored name
POINT_FIELDS = {
    "e_base_pct": "e_base_pct", "e_final_pct": "e_final_pct", "e_corr_pct": "e_always_pct",
    "e_eqwm_pct": "e_eqwm_pct", "sigma_pct": "sigma_pct", "confidence": "confidence",
    "alpha_a": "alpha_a", "alpha_c": "alpha_c", "alpha_nw": "alpha_nw",
    "y_true": "y_true", "y_base": "y_base", "y_final": "y_final", "y_eqwm": "y_eqwm",
}
# Error families: cases.csv column prefix -> points.npz field
ERROR_FAMILIES = {"base": "e_base_pct", "final": "e_final_pct", "always": "e_always_pct", "eqwm": "e_eqwm_pct"}
ROLE_TRAINING, ROLE_UNTRAINED = "training", "untrained"
PLOTTED_KINDS = ("dataset", "group")


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default=None,
                        help="TOML under ./inputfiles/ (e.g. test/test_turb_allcases.toml) or a path; "
                             "optional with --replot (then the options stored in RUN_DIR/overall.json are used)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (-1 = CPU); overrides [general].GpuNum")
    parser.add_argument("--output-dir", default=None,
                        help="output directory; overrides [general].OutputDir (with --replot: where the figures go, "
                             "default RUN_DIR)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="per-dataset deterministic sample cap after the domain filter (0 = all); "
                             "overrides [data].MaxSamplesPerDataset")
    parser.add_argument("--datasets", nargs="+", default=None, metavar="KEY",
                        help="evaluate only these dataset keys (or [groups] names, expanded to their members)")
    parser.add_argument("--scale", choices=["linear", "log", "both"], default=None,
                        help="scatter axes; overrides [plots].Scale")
    parser.add_argument("--replot", default=None, metavar="RUN_DIR",
                        help="regenerate the figures from RUN_DIR/cases.csv + points.npz (no torch/TF, no inference)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the TOML and resolve every dataset file, then exit (no models loaded)")
    return parser.parse_args(argv)


def _check_cwd() -> None:
    if os.path.abspath(os.getcwd()) != REPO_ROOT:
        print(f"WARNING: run from the repository root ({REPO_ROOT}); ./inputfiles, ./models and ./data "
              f"are resolved relative to the current directory {os.getcwd()}")


# ---------------------------------------------------------------------------------------
# Dataset selection, groups and roles
# ---------------------------------------------------------------------------------------


def select_datasets(cfg: Mapping[str, Any], requested: Optional[Sequence[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Dataset keys to evaluate and the groups formed from them

    Args:
        cfg: Validated benchmark config
        requested: --datasets (keys or group names), or None for [data].Datasets

    Returns:
        (keys in evaluation order, {group: members that are evaluated}); groups without any
        evaluated member are dropped
    """
    groups_cfg = cfg["groups"]
    if requested is None:
        keys = list(cfg["data"]["Datasets"])
    else:
        keys = []
        for item in requested:
            members = groups_cfg.get(item, [item])
            for k in members:
                if k not in keys:
                    keys.append(k)
        extra = [k for k in keys if k not in cfg["data"]["Datasets"]]
        if extra:
            print(f"NOTE: --datasets {extra} are not in [data].Datasets of the TOML (evaluated anyway)")
    groups: Dict[str, List[str]] = {}
    for gname, members in groups_cfg.items():
        present = [m for m in members if m in keys]
        if present:
            groups[gname] = present
            missing = [m for m in members if m not in keys]
            if missing:
                print(f"NOTE: group '{gname}' is pooled from {present} only ({missing} not selected)")
    return keys, groups


def dataset_roles(keys: Sequence[str], groups: Mapping[str, Sequence[str]],
                  training: Sequence[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Role ('training' / 'untrained') of every dataset and of every group

    A dataset's own role is its [data].TrainingDatasets membership. A group is 'training' if
    its name is listed there or if any member is a training set (a mixed group is reported);
    group members take the role of their group (they are plotted only as the pooled case).

    Args:
        keys: Evaluated dataset keys
        groups: {group: evaluated members}
        training: [data].TrainingDatasets

    Returns:
        ({dataset key: role used for its points}, {group: role})
    """
    train = set(training)
    own = {k: ROLE_TRAINING if k in train else ROLE_UNTRAINED for k in keys}
    group_roles: Dict[str, str] = {}
    for gname, members in groups.items():
        member_roles = {own[m] for m in members}
        if gname in train or ROLE_TRAINING in member_roles:
            group_roles[gname] = ROLE_TRAINING
        else:
            group_roles[gname] = ROLE_UNTRAINED
        if len(member_roles) > 1 and gname not in train:
            print(f"NOTE: group '{gname}' mixes training and untrained members; plotted as training")
    roles = dict(own)
    for gname, members in groups.items():
        for m in members:
            roles[m] = group_roles[gname]
    return roles, group_roles


def group_of(groups: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    """{member key: group name}."""
    return {m: g for g, members in groups.items() for m in members}


# ---------------------------------------------------------------------------------------
# Evaluation filters (truth used only to select evaluation points)
# ---------------------------------------------------------------------------------------


def domain_mask(arrays: Any, ev: Mapping[str, Any]) -> Tuple[np.ndarray, str]:
    """
    Evaluation-domain filter ([evaluation].DomainFilter)

    'retau' keeps |u_tau delta / nu| > RetauMin (EVB Retau_thresh, dataset u_tau); 'piout'
    keeps |pi_out,true| >= PioutMin; 'none' or a zero minimum keeps everything. The truth
    only selects evaluation points; no prediction uses it.

    Args:
        arrays: DatasetArrays of one dataset
        ev: [evaluation] table

    Returns:
        (keep mask, description)
    """
    n = len(arrays)
    mode = ev["DomainFilter"]
    if mode == "retau" and ev["RetauMin"] > 0.0:
        U = np.asarray(arrays.U, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            retau = np.abs(U[:, 3] * np.asarray(arrays.delta, dtype=np.float64) / U[:, 2])
        return np.isfinite(retau) & (retau > ev["RetauMin"]), f"|u_tau*delta/nu| > {ev['RetauMin']:g}"
    if mode == "piout" and ev["PioutMin"] > 0.0:
        y = np.abs(np.asarray(arrays.y, dtype=np.float64))
        return np.isfinite(y) & (y >= ev["PioutMin"]), f"|pi_out,true| >= {ev['PioutMin']:g}"
    return np.ones(n, dtype=bool), "none"


def evaluation_mask(p: Mapping[str, np.ndarray], ev: Mapping[str, Any]) -> np.ndarray:
    """
    Points used in the statistics: joint outlier mask and the optional error-change filter

    Args:
        p: Per-point fields of a dataset (POINT_FIELDS names)
        ev: [evaluation] table

    Returns:
        Boolean mask
    """
    eb, ef, ee = (np.abs(np.asarray(p[k], dtype=np.float64)) for k in ("e_base_pct", "e_final_pct", "e_eqwm_pct"))
    used = np.isfinite(eb) & np.isfinite(ef) & np.isfinite(ee)
    thr = ev["OutlierPercent"]
    if thr > 0.0:
        used &= (eb <= thr) & (ef <= thr) & (ee <= thr)
    if ev["MinErrorChangePercent"] > 0.0:
        used &= np.abs(ef - eb) > ev["MinErrorChangePercent"]
    return used


# ---------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------


def _abs_stats(v: np.ndarray) -> Tuple[float, float, float]:
    a = np.abs(np.asarray(v, dtype=np.float64))
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    return float(np.mean(a)), float(np.median(a)), float(np.percentile(a, 90.0))


def case_statistics(p: Mapping[str, np.ndarray], ev: Mapping[str, Any], compare_always: bool) -> Dict[str, float]:
    """
    Statistics of one case over its used points

    Args:
        p: Per-point fields restricted to the used points
        ev: [evaluation] table (OutlierPercent for the max improvement)
        compare_always: Report the always-corrected statistics (else NaN)

    Returns:
        Dict of cases.csv statistic columns
    """
    row: Dict[str, float] = {}
    for fam, key in ERROR_FAMILIES.items():
        if fam == "always" and not compare_always:
            row.update({f"{fam}_mean": np.nan, f"{fam}_median": np.nan, f"{fam}_p90": np.nan})
            continue
        row[f"{fam}_mean"], row[f"{fam}_median"], row[f"{fam}_p90"] = _abs_stats(p[key])
    row["improvement"] = row["base_mean"] - row["final_mean"]
    row["improvement_always"] = row["base_mean"] - row["always_mean"]
    eb = np.abs(np.asarray(p["e_base_pct"], dtype=np.float64))
    gain = eb - np.abs(np.asarray(p["e_final_pct"], dtype=np.float64))
    thr = ev["OutlierPercent"]
    ok = eb <= thr if thr > 0.0 else np.ones(eb.size, dtype=bool)
    row["max_improvement"] = float(np.max(gain[ok])) if np.any(ok) else np.nan
    n = len(eb)
    for name in ("sigma_pct", "confidence", "alpha_a", "alpha_c", "alpha_nw"):
        v = np.asarray(p[name], dtype=np.float64)
        row[f"{name}_mean"] = float(np.nanmean(v)) if n else np.nan
    if n:
        a_nw, a_c, a_a = (np.asarray(p[k], dtype=np.float64) for k in ("alpha_nw", "alpha_c", "alpha_a"))
        tol = 1e-6  # float32 storage of the weights
        row["frac_fully_corrected"] = float(np.mean((a_nw >= 1 - tol) & (a_c >= 1 - tol) & (a_a >= 1 - tol)))
        row["frac_full_fallback"] = float(np.mean((a_nw <= tol) | (a_c <= tol)))
    else:
        row["frac_fully_corrected"] = row["frac_full_fallback"] = np.nan
    return row


def _subset(p: Mapping[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v)[mask] for k, v in p.items()}


def _concat(parts: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in POINT_FIELDS.values()}


def aggregate_cases(records: Mapping[str, Dict[str, Any]], keys: Sequence[str], groups: Mapping[str, Sequence[str]],
                    roles: Mapping[str, str], group_roles: Mapping[str, str], display: Mapping[str, str],
                    ev: Mapping[str, Any], compare_always: bool) -> List[Dict[str, Any]]:
    """
    One row per evaluated dataset (kind 'dataset' or 'member'), group ('group') and skipped dataset

    Args:
        records: Per-dataset results of run_dataset
        keys: Evaluated keys (order of the rows)
        groups: {group: evaluated members}
        roles: Role of the points of every dataset (groups' members take the group role)
        group_roles: {group: role}
        display: [display_names]
        ev: [evaluation] table
        compare_always: Report the always-corrected statistics

    Returns:
        List of row dicts (cases.csv), datasets first, then groups
    """
    member_of = group_of(groups)
    rows: List[Dict[str, Any]] = []
    for key in keys:
        rec = records[key]
        row: Dict[str, Any] = {
            "name": display.get(key, key), "kind": "member" if key in member_of else "dataset",
            "keys": key, "group": member_of.get(key, ""), "role": roles[key],
            "status": rec["status"], "reason": rec.get("reason", ""),
            "n_raw": rec.get("n_raw", 0), "n_domain": rec.get("n_domain", 0), "n_eval": rec.get("n_eval", 0),
            "n_used": int(np.sum(rec["used"])) if rec["status"] == "ok" else 0,
            "input_inconsistent_frac": rec.get("input_inconsistent_frac", np.nan),
        }
        if rec["status"] == "ok":
            row.update(case_statistics(_subset(rec["points"], rec["used"]), ev, compare_always))
        rows.append(row)
    for gname, members in groups.items():
        ok = [m for m in members if records[m]["status"] == "ok"]
        row = {"name": gname, "kind": "group", "keys": ";".join(members), "group": gname,
               "role": group_roles[gname], "status": "ok" if ok else "skipped",
               "reason": "" if ok else "no member evaluated",
               "n_raw": sum(records[m].get("n_raw", 0) for m in members),
               "n_domain": sum(records[m].get("n_domain", 0) for m in members),
               "n_eval": sum(records[m].get("n_eval", 0) for m in members),
               "n_used": sum(int(np.sum(records[m]["used"])) for m in ok),
               "input_inconsistent_frac": np.nan}
        skipped = [m for m in members if m not in ok]
        if skipped and ok:
            row["reason"] = f"members skipped: {';'.join(skipped)}"
        if ok:
            fracs = [records[m]["input_inconsistent_frac"] for m in ok]
            weights = [records[m]["n_eval"] for m in ok]
            if np.sum(weights) > 0 and np.all(np.isfinite(fracs)):
                row["input_inconsistent_frac"] = float(np.average(fracs, weights=weights))
            pooled = _concat([_subset(records[m]["points"], records[m]["used"]) for m in ok])
            if len(pooled["e_base_pct"]):
                row.update(case_statistics(pooled, ev, compare_always))
            else:
                row["status"], row["reason"] = "skipped", "no points left after the evaluation filters"
        rows.append(row)
    return rows


def plotted_cases(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Rows drawn as scatter markers (ungrouped datasets and groups with statistics)."""
    return [r for r in rows if r["kind"] in PLOTTED_KINDS and r["status"] == "ok"
            and np.isfinite(float(r.get("base_mean", np.nan)))]


def overall_statistics(rows: Sequence[Mapping[str, Any]], points: Mapping[str, np.ndarray],
                       compare_always: bool) -> Dict[str, Any]:
    """
    Overall averages: unweighted over cases (EVB headline), point-weighted, and per role

    Args:
        rows: cases.csv rows
        points: Pooled per-point arrays of all evaluated datasets (with 'used', 'training')
        compare_always: Include the always-corrected statistics

    Returns:
        {'unweighted': ..., 'point_weighted': ..., 'by_role': {role: {...}}}
    """
    fams = [f for f in ERROR_FAMILIES if compare_always or f != "always"]

    def unweighted(cases):
        if not cases:
            return {"n_cases": 0}
        out: Dict[str, Any] = {"n_cases": len(cases)}
        for f in fams:
            out[f"{f}_mean"] = float(np.nanmean([c[f"{f}_mean"] for c in cases]))
        out["improvement"] = out["base_mean"] - out["final_mean"]
        out["max_improvement"] = float(np.nanmean([c["max_improvement"] for c in cases]))
        for k in ("sigma_pct_mean", "confidence_mean", "alpha_a_mean", "alpha_c_mean", "alpha_nw_mean",
                  "frac_fully_corrected", "frac_full_fallback"):
            out[k] = float(np.nanmean([c[k] for c in cases]))
        return out

    def weighted(mask):
        n = int(np.sum(mask))
        if n == 0:
            return {"n_points": 0}
        out: Dict[str, Any] = {"n_points": n}
        for f in fams:
            out[f"{f}_mean"], out[f"{f}_median"], out[f"{f}_p90"] = _abs_stats(points[ERROR_FAMILIES[f]][mask])
        out["improvement"] = out["base_mean"] - out["final_mean"]
        out["sigma_pct_mean"] = float(np.nanmean(points["sigma_pct"][mask]))
        out["confidence_mean"] = float(np.nanmean(points["confidence"][mask]))
        return out

    cases = plotted_cases(rows)
    used = np.asarray(points["used"], dtype=bool) if len(points.get("used", [])) else np.zeros(0, dtype=bool)
    train = np.asarray(points["training"], dtype=bool) if used.size else used
    res: Dict[str, Any] = {"unweighted": unweighted(cases), "point_weighted": weighted(used), "by_role": {}}
    for role, pmask in ((ROLE_TRAINING, train), (ROLE_UNTRAINED, ~train)):
        res["by_role"][role] = {"unweighted": unweighted([c for c in cases if c["role"] == role]),
                                "point_weighted": weighted(used & pmask)}
    return res


# ---------------------------------------------------------------------------------------
# Console tables
# ---------------------------------------------------------------------------------------


def _f(v: Any, fmt: str = "10.2f") -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return f"{'-':>{int(fmt.split('.')[0])}}"
    return f"{v:{fmt}}" if np.isfinite(v) else f"{'nan':>{int(fmt.split('.')[0])}}"


def print_case_table(rows: Sequence[Mapping[str, Any]], overall: Mapping[str, Any], compare_always: bool) -> None:
    """EVB-style summary table (mean |e%| per case) and the overall averages."""
    width = 150
    print("\n" + "=" * width)
    print("ERROR COMPARISON SUMMARY (mean |e%| over the used points of each case; final = blended expert model)")
    print("=" * width)
    print(f"{'Case':<32s}{'Role':<10s}{'N used':>9s}{'Baseline':>10s}{'Final':>10s}{'Always':>10s}{'EQWM':>10s}"
          f"{'Improv':>10s}{'MaxImprov':>11s}{'Sigma%':>9s}{'Conf':>8s}{'FullCorr':>10s}{'FullFB':>8s}")
    print("-" * width)
    by_group: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        if r["kind"] == "member":
            by_group.setdefault(r["group"], []).append(r)

    def line(r, indent=""):
        name = (indent + str(r["name"]))[:31]
        if r["status"] != "ok" or not np.isfinite(float(r.get("base_mean", np.nan))):
            print(f"{name:<32s}{r['role']:<10s}  SKIPPED: {r['reason']}")
            return
        print(f"{name:<32s}{r['role']:<10s}{int(r['n_used']):>9d}{_f(r['base_mean'])}{_f(r['final_mean'])}"
              f"{_f(r['always_mean'])}{_f(r['eqwm_mean'])}{_f(r['improvement'])}{_f(r['max_improvement'], '11.2f')}"
              f"{_f(r['sigma_pct_mean'], '9.2f')}{_f(r['confidence_mean'], '8.4f')}"
              f"{_f(r['frac_fully_corrected'], '10.3f')}{_f(r['frac_full_fallback'], '8.3f')}")

    for r in rows:
        if r["kind"] == "dataset":
            line(r)
    for r in rows:
        if r["kind"] == "group":
            line(r)
            for m in by_group.get(r["name"], []):
                line(m, indent="  - ")
    print("=" * width)
    u = overall["unweighted"]
    if u.get("n_cases", 0) == 0:
        print("No case evaluated.")
        return
    print(f"\nOVERALL AVERAGE (unweighted mean over {u['n_cases']} cases):")
    print(f"  Baseline Error:      {u['base_mean']:.2f}%")
    print(f"  Corrected Error:     {u['final_mean']:.2f}%   (blended)")
    if compare_always:
        print(f"  Always Corrected:    {u['always_mean']:.2f}%")
    print(f"  EQWM Error:          {u['eqwm_mean']:.2f}%")
    print(f"  Avg Sigma (%):       {u['sigma_pct_mean']:.2f}%")
    print(f"  Avg Confidence:      {u['confidence_mean']:.4f}")
    print(f"  Average Improvement: {u['improvement']:.2f}%")
    print(f"  Average Max Improvement: {u['max_improvement']:.2f}%")
    pw = overall["point_weighted"]
    if pw.get("n_points", 0):
        alw = f", always {pw['always_mean']:.2f}" if compare_always else ""
        print(f"POINT-WEIGHTED ({pw['n_points']:,} points): baseline {pw['base_mean']:.2f}, final {pw['final_mean']:.2f}"
              f"{alw}, EQWM {pw['eqwm_mean']:.2f} % (medians {pw['base_median']:.2f} / {pw['final_median']:.2f} %)")
    for role, st in overall["by_role"].items():
        uw, w = st["unweighted"], st["point_weighted"]
        if uw.get("n_cases", 0) == 0:
            print(f"{role.upper():<10s}: no case")
            continue
        print(f"{role.upper():<10s}: {uw['n_cases']} cases, unweighted baseline {uw['base_mean']:.2f} -> final "
              f"{uw['final_mean']:.2f} % (EQWM {uw['eqwm_mean']:.2f}); point-weighted {w['base_mean']:.2f} -> "
              f"{w['final_mean']:.2f} % on {w['n_points']:,} points")


def print_domain_table(records: Mapping[str, Mapping[str, Any]], keys: Sequence[str], desc: str) -> None:
    """EVB Re_tau-filter table: points removed by the domain filter per dataset."""
    print("\n" + "=" * 80)
    print(f"DOMAIN FILTER SUMMARY ({desc})")
    print("=" * 80)
    rows = [(k, records[k]["n_raw"] - records[k]["n_domain"], records[k]["n_raw"]) for k in keys
            if records[k].get("n_raw", 0) > 0 and records[k]["n_raw"] > records[k].get("n_domain", 0)]
    if not rows:
        print("No points removed by the domain filter")
        print("=" * 80)
        return
    print(f"Number of datasets with points removed: {len(rows)}")
    print(f"\n{'Dataset':<40s} {'Filtered Out':>13s} {'Original':>12s} {'% Filtered':>11s}")
    print("-" * 80)
    for k, nf, tot in rows:
        print(f"{k:<40s} {nf:>13d} {tot:>12d} {100.0 * nf / tot:>10.2f}%")
    print("=" * 80)


# ---------------------------------------------------------------------------------------
# Inference pass
# ---------------------------------------------------------------------------------------


def run_dataset(key: str, models: Any, cfg: Mapping[str, Any], max_samples: int) -> Dict[str, Any]:
    """
    Load one dataset, apply the domain filter and the sample cap, run the shared pipeline

    Args:
        key: Dataset key
        models: BranchModels
        cfg: Benchmark config
        max_samples: Per-dataset cap (0 = all)

    Returns:
        Record: status ('ok' / 'skipped'), reason, n_raw/n_domain/n_eval, points (float32
        per-point fields), used (evaluation mask), input_inconsistent_frac, time
    """
    from src_err.data import load_datasets
    from src_err.pipeline import input_consistency, predict_turb

    d = cfg["data"]
    rec: Dict[str, Any] = {"status": "skipped", "reason": "", "n_raw": 0, "n_domain": 0, "n_eval": 0}
    t0 = time.time()
    try:
        arrays = load_datasets([key], models.baseline.config, d["upy"], d["downy"], files_override=d["Files"],
                               stride=d["Stride"], max_samples=0)
        rec["n_raw"] = len(arrays)
        if len(arrays) == 0:
            rec["reason"] = "no points in the n1/delta window"
            return rec
        keep, _ = domain_mask(arrays, cfg["evaluation"])
        arrays = arrays.subset(np.flatnonzero(keep))
        rec["n_domain"] = len(arrays)
        if len(arrays) == 0:
            rec["reason"] = "no points left after the domain filter"
            return rec
        if max_samples and len(arrays) > max_samples:
            arrays = arrays.subset(np.linspace(0, len(arrays) - 1, int(max_samples), dtype=np.int64))
        rec["n_eval"] = len(arrays)
        rec["input_inconsistent_frac"] = input_consistency(arrays, models.baseline.config, verbose=False).get(key, np.nan)
        out = predict_turb(models, arrays, cfg)
        points = {name: np.asarray(out[k], dtype=np.float32) for k, name in POINT_FIELDS.items()}
        rec["used"] = evaluation_mask(points, cfg["evaluation"])
        rec["points"] = points
        if not np.any(rec["used"]):
            rec["reason"] = "no points left after the outlier / error-change filters"
            return rec
        rec["status"] = "ok"
    except Exception as exc:  # one failing dataset must not stop the benchmark; it is reported as skipped
        rec["reason"] = f"{type(exc).__name__}: {exc}".replace("\n", " ")
        print(f"ERROR: dataset {key} failed: {rec['reason']}")
        traceback.print_exc(limit=3)
    finally:
        rec["time"] = time.time() - t0
    return rec


def pool_points(records: Mapping[str, Mapping[str, Any]], keys: Sequence[str],
                roles: Mapping[str, str], groups: Mapping[str, Sequence[str]],
                display: Mapping[str, str]) -> Dict[str, np.ndarray]:
    """
    Concatenate the per-point arrays of all evaluated datasets (points.npz content)

    Returns:
        POINT_FIELDS arrays + 'dataset' (int index into 'dataset_keys'), 'case' (index into
        'case_names': the group name for group members, else the display name), 'training'
        (bool, role of the plotted case), 'used' (evaluation mask)
    """
    member_of = group_of(groups)
    ok = [k for k in keys if "points" in records[k]]
    case_names: List[str] = []
    parts, ds_idx, case_idx, train, used = [], [], [], [], []
    for i, k in enumerate(ok):
        rec = records[k]
        n = len(rec["used"])
        cname = member_of.get(k, display.get(k, k))
        if cname not in case_names:
            case_names.append(cname)
        parts.append(rec["points"])
        ds_idx.append(np.full(n, i, dtype=np.int32))
        case_idx.append(np.full(n, case_names.index(cname), dtype=np.int32))
        train.append(np.full(n, roles[k] == ROLE_TRAINING))
        used.append(np.asarray(rec["used"], dtype=bool))
    if not parts:
        empty = {name: np.zeros(0, dtype=np.float32) for name in POINT_FIELDS.values()}
        empty.update(dataset=np.zeros(0, np.int32), case=np.zeros(0, np.int32), training=np.zeros(0, bool),
                     used=np.zeros(0, bool), dataset_keys=np.asarray([], dtype=str),
                     case_names=np.asarray([], dtype=str))
        return empty
    pts = _concat(parts)
    pts.update(dataset=np.concatenate(ds_idx), case=np.concatenate(case_idx), training=np.concatenate(train),
               used=np.concatenate(used), dataset_keys=np.asarray(ok, dtype=str),
               case_names=np.asarray(case_names, dtype=str))
    return pts


# ---------------------------------------------------------------------------------------
# Figures (shared by the full run and --replot; matplotlib only)
# ---------------------------------------------------------------------------------------


def make_figures(rows: Sequence[Mapping[str, Any]], points: Mapping[str, np.ndarray], cfg: Mapping[str, Any],
                 out_dir: str, scale: Optional[str]) -> List[str]:
    """
    Scatter figures (train_test, zoom, sigma, confidence; linear and/or log) and the PDFs

    Args:
        rows: cases.csv rows
        points: Pooled per-point arrays (points.npz)
        cfg: Config with [general], [plots], [display_names], [evaluation]
        out_dir: Output directory
        scale: --scale override

    Returns:
        Names of the figures written (without extension)
    """
    import matplotlib
    matplotlib.use("Agg")
    from src_err import plotting

    plots = dict(cfg["plots"])
    run_name = cfg["general"]["RunName"]
    formats = cfg["general"]["Formats"]
    compare_always = cfg["evaluation"]["CompareAlways"]
    written: List[str] = []
    cases = plotted_cases(rows)
    print("=" * 80)
    if cases:
        table = {"name": [c["name"] for c in cases],
                 "baseline": [float(c["base_mean"]) for c in cases],
                 "final": [float(c["final_mean"]) for c in cases],
                 "training": [c["role"] == ROLE_TRAINING for c in cases],
                 "sigma": [float(c["sigma_pct_mean"]) for c in cases],
                 "confidence": [float(c["confidence_mean"]) for c in cases]}
        figs = plotting.benchmark_scatter_figures(table, plots, run_name, out_dir=out_dir, formats=formats,
                                                  scale=scale, rename_mapping=cfg["display_names"])
        written += [f"{run_name}_{k}" for k in figs]
    else:
        print("WARNING: no case to plot; scatter figures skipped")
    used = np.asarray(points.get("used", []), dtype=bool)
    if used.any():
        pdf_points = {"baseline": points["e_base_pct"][used], "final": points["e_final_pct"][used],
                      "eqwm": points["e_eqwm_pct"][used], "confidence": points["confidence"][used],
                      "training": points["training"][used]}
        if compare_always:
            pdf_points["always"] = points["e_always_pct"][used]
        figs = plotting.benchmark_pdf_figures(pdf_points, plots, run_name, out_dir=out_dir, formats=formats)
        written += [f"{run_name}_{k}" for k in figs]
    else:
        print("WARNING: no used points; PDF figures skipped")
    print(f"Figures written to {out_dir}: {', '.join(written) or 'none'}")
    return written


# ---------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------

CASE_COLUMNS = ["name", "kind", "keys", "group", "role", "status", "reason", "n_raw", "n_domain", "n_eval",
                "n_used", "input_inconsistent_frac"]
CASE_COLUMNS += [f"{f}_{s}" for f in ERROR_FAMILIES for s in ("mean", "median", "p90")]
CASE_COLUMNS += ["improvement", "improvement_always", "max_improvement", "sigma_pct_mean", "confidence_mean",
                 "alpha_a_mean", "alpha_c_mean", "alpha_nw_mean", "frac_fully_corrected", "frac_full_fallback"]


def write_cases_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    import pandas as pd
    df = pd.DataFrame([{c: r.get(c, np.nan) for c in CASE_COLUMNS} for r in rows], columns=CASE_COLUMNS)
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"Saved: {path}")


def read_cases_csv(path: str) -> List[Dict[str, Any]]:
    import pandas as pd
    df = pd.read_csv(path, keep_default_na=False, na_values=["nan", "NaN", ""])
    for c in ("name", "kind", "keys", "group", "role", "status", "reason"):
        df[c] = df[c].fillna("").astype(str)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------------------


def print_config(cfg: Mapping[str, Any], keys: Sequence[str], groups: Mapping[str, Sequence[str]],
                 out_dir: str, gpu: int, max_samples: int, domain_desc: str) -> None:
    d, g, ev = cfg["data"], cfg["gates"], cfg["evaluation"]
    print("=" * 80)
    print(f"A-priori benchmark: {cfg.get('config_file')}")
    print(f"  models        : baseline {cfg['models']['Baseline']}")
    print(f"                  SNGP     {cfg['models']['Sngp']}")
    print(f"                  error    {cfg['models']['Error']}")
    print(f"  datasets      : {len(keys)} ({', '.join(keys[:6])}{', ...' if len(keys) > 6 else ''})")
    print(f"  groups        : {dict(groups) or 'none'}")
    print(f"  window        : upy {d['upy']} (>= 1: no upper cut), downy {d['downy']}; stride {d['Stride']}; "
          f"max samples per dataset {max_samples or 'all'}")
    print(f"  EQWM law      : {cfg['eqwm']['Law']};  confidence covariance: {cfg['confidence']['Covariance']}")
    a = g["Aleatoric"]
    print(f"  gates         : aleatoric {a['Type']} {a['Threshold']} +- {a['Delta']}, "
          f"confidence {g['Confidence']['Threshold']} +- {g['Confidence']['Delta']}, "
          f"y+ {g['Yplus']['Threshold']} +- {g['Yplus']['Delta']}, "
          f"low value {g['LowValue']['Threshold']} +- {g['LowValue']['Delta']}")
    m = cfg["metrics"]["LowValue"]
    print(f"  metrics       : low value {m['Threshold']} +- {m['Delta']} on |pi_out,true|")
    print(f"  evaluation    : domain filter {domain_desc}; outliers > {ev['OutlierPercent']:g} % (joint); "
          f"error change > {ev['MinErrorChangePercent']:g} % ({'on' if ev['MinErrorChangePercent'] > 0 else 'off'}); "
          f"always-corrected {'on' if ev['CompareAlways'] else 'off'}")
    print(f"  output        : {out_dir};  formats {cfg['general']['Formats']};  device "
          f"{'CPU' if gpu < 0 else 'GPU ' + str(gpu)}")
    print("=" * 80)


def _domain_description(ev: Mapping[str, Any]) -> str:
    if ev["DomainFilter"] == "retau" and ev["RetauMin"] > 0:
        return f"retau: |u_tau*delta/nu| > {ev['RetauMin']:g}"
    if ev["DomainFilter"] == "piout" and ev["PioutMin"] > 0:
        return f"piout: |pi_out,true| >= {ev['PioutMin']:g}"
    return "none"


def dry_run(cfg: Mapping[str, Any], keys: Sequence[str]) -> int:
    """Resolve every dataset file (and the models) without loading anything; returns the number of problems."""
    import pandas as pd
    from src_err.cli import missing_model  # torch-free, unlike src_err.data
    from src_err.utils import resolve_dataset_file, resolve_model_path

    problems = 0
    print("=" * 80)
    print("Models:")
    for role, name in cfg["models"].items():
        # Same rule as the run's input check and the loaders (a file is not an SNGP/error-model directory)
        ok = missing_model(role, name) is None
        path = resolve_model_path(name, directory=(role != "Baseline"))
        problems += 0 if ok else 1
        print(f"  {role:<9s}{'OK     ' if ok else 'MISSING'} {path}")
    print(f"Datasets ({len(keys)}):")
    total = 0
    for k in keys:
        try:
            path = resolve_dataset_file(k, cfg["data"]["Files"])
        except FileNotFoundError as exc:
            problems += 1
            print(f"  {k:<28s} MISSING  {exc}")
            continue
        try:
            with pd.HDFStore(path, mode="r") as store:
                storer = store.get_storer("output")
                n = storer.nrows if storer.nrows is not None else storer.shape[0]  # fixed format: shape
                flow_cols = list(store.select("flow_type", stop=1).columns)
        except Exception as exc:  # unreadable file: report, continue
            problems += 1
            print(f"  {k:<28s} UNREADABLE {path}: {type(exc).__name__}: {exc}")
            continue
        n = int(n) if n is not None else -1
        total += max(n, 0)
        note = "" if "delta" in flow_cols or "delta99" in flow_cols else "  (no delta column: window/retau impossible)"
        print(f"  {k:<28s} OK {n:>10,d} rows  {path}{note}")
    print(f"Total rows before the window: {total:,}; problems: {problems}")
    print("=" * 80)
    return problems


def run(args: argparse.Namespace) -> None:
    import src_err.utils as utils  # sets the TensorFlow environment before TF is imported
    from src_err.cli import check_inputs, select_gpu
    from src_err.config import load_benchmark_config, config_to_metadata
    from src_err.evaluation import write_summary

    try:
        cfg = load_benchmark_config(args.config)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"ERROR: {exc}")
    keys, groups = select_datasets(cfg, args.datasets)
    gpu = select_gpu(args, [cfg])
    out_dir = cfg["general"]["OutputDir"] if args.output_dir is None else args.output_dir
    max_samples = cfg["data"]["MaxSamplesPerDataset"] if args.max_samples is None else args.max_samples
    if args.scale is not None:
        cfg["plots"]["Scale"] = args.scale
    domain_desc = _domain_description(cfg["evaluation"])
    print_config(cfg, keys, groups, out_dir, gpu, max_samples, domain_desc)
    if args.dry_run:
        problems = dry_run(cfg, keys)
        sys.exit(1 if problems else 0)
    try:
        # A missing file is a config error: fail before the models are loaded, not as one skipped
        # dataset after hours of evaluation (datasets that fail to load later are still skipped)
        check_inputs([{**cfg, "data": {**cfg["data"], "Datasets": keys}}])
    except FileNotFoundError as exc:
        sys.exit(f"ERROR: {exc}")

    roles, group_roles = dataset_roles(keys, groups, cfg["data"]["TrainingDatasets"])
    display = cfg["display_names"]
    compare_always = cfg["evaluation"]["CompareAlways"]
    timings: Dict[str, float] = {}
    t0 = time.time()

    from src_err.pipeline import load_branch_models
    device = utils.configure_devices(gpu)
    # The benchmark has no class plots: no post-hoc groups / display names needed from [classes]
    models = load_branch_models(cfg["models"], {}, device, branch="turb")
    timings["load_models"] = time.time() - t0

    records: Dict[str, Dict[str, Any]] = {}
    for i, key in enumerate(keys):
        print("=" * 80)
        print(f"[{i + 1}/{len(keys)}] {key} ({roles[key]})")
        records[key] = run_dataset(key, models, cfg, max_samples)
        rec = records[key]
        if rec["status"] == "ok":
            u = rec["used"]
            eb, ef = (np.abs(rec["points"][f][u]) for f in ("e_base_pct", "e_final_pct"))
            print(f"  {rec['n_raw']:,} points -> {rec['n_domain']:,} after the domain filter -> {rec['n_eval']:,} "
                  f"evaluated -> {int(u.sum()):,} used; mean |e| baseline {eb.mean():.2f} %, final {ef.mean():.2f} % "
                  f"({rec['time']:.1f} s)")
            if rec.get("input_inconsistent_frac", 0.0) > 0.0:
                print(f"  WARNING: stored network inputs inconsistent with u1*n1/nu on "
                      f"{100 * rec['input_inconsistent_frac']:.1f} % of the points (data issue, see summary)")
        else:
            print(f"  SKIPPED: {rec['reason']}")
    timings["inference"] = time.time() - t0 - timings["load_models"]

    rows = aggregate_cases(records, keys, groups, roles, group_roles, display, cfg["evaluation"], compare_always)
    points = pool_points(records, keys, roles, groups, display)
    overall = overall_statistics(rows, points, compare_always)
    print_case_table(rows, overall, compare_always)
    print_domain_table(records, keys, domain_desc)
    skipped = {k: records[k]["reason"] for k in keys if records[k]["status"] != "ok"}
    if skipped:
        print(f"\nSKIPPED DATASETS ({len(skipped)}):")
        for k, why in skipped.items():
            print(f"  {k:<32s} {why}")
    inconsistent = {k: records[k]["input_inconsistent_frac"] for k in keys
                    if records[k].get("input_inconsistent_frac", 0.0) > 0.0}
    if inconsistent:
        print(f"\nWARNING: stored network inputs inconsistent with u1*n1/nu (fraction of points): {inconsistent}")

    utils.make_output_dir(out_dir)
    write_cases_csv(os.path.join(out_dir, "cases.csv"), rows)
    np.savez(os.path.join(out_dir, "points.npz"), **points)
    print(f"Saved: {os.path.join(out_dir, 'points.npz')} ({len(points['used']):,} points)")
    t = time.time()
    figures = make_figures(rows, points, cfg, out_dir, None)
    timings["figures"] = time.time() - t
    timings["total"] = time.time() - t0
    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cli": {"config": args.config, "gpu": gpu, "output_dir": out_dir, "max_samples_per_dataset": max_samples,
                "datasets": args.datasets, "scale": args.scale},
        "config": config_to_metadata(cfg),
        "models": models.paths,
        "git_commit": utils.git_commit_hash(REPO_ROOT),
        "versions": utils.software_versions(),
        "datasets": list(keys),
        "groups": {g: list(m) for g, m in groups.items()},
        "roles": {"datasets": {k: roles[k] for k in keys}, "groups": dict(group_roles)},
        "domain_filter": domain_desc,
        "overall": overall,
        "skipped": skipped,
        "input_inconsistent": inconsistent,
        "figures": figures,
        "timings": timings,
    }
    write_summary(os.path.join(out_dir, "overall.json"), summary)
    print("=" * 80)
    print(f"Done in {timings['total']:.1f} s; outputs in {out_dir}")


def replot(args: argparse.Namespace) -> None:
    """Regenerate the figures of a finished run without torch/TF (see the module docstring)."""
    import json

    run_dir = args.replot
    for name in ("cases.csv", "points.npz"):
        if not os.path.isfile(os.path.join(run_dir, name)):
            sys.exit(f"ERROR: --replot: {os.path.join(run_dir, name)} not found")
    if args.config is not None:
        from src_err.config import load_benchmark_config  # torch-free
        try:
            cfg = load_benchmark_config(args.config)
        except (ValueError, FileNotFoundError) as exc:
            sys.exit(f"ERROR: {exc}")
        source = cfg.get("config_file", args.config)
    else:
        path = os.path.join(run_dir, "overall.json")
        if not os.path.isfile(path):
            sys.exit(f"ERROR: --replot without a TOML needs {path}")
        with open(path, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)["config"]
        source = path
    if args.scale is not None:
        cfg["plots"]["Scale"] = args.scale
    out_dir = args.output_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 80)
    print(f"Replot of {run_dir} (plot options from {source}; scale {cfg['plots']['Scale']}) -> {out_dir}")
    rows = read_cases_csv(os.path.join(run_dir, "cases.csv"))
    with np.load(os.path.join(run_dir, "points.npz"), allow_pickle=False) as data:
        points = {k: data[k] for k in data.files}
    make_figures(rows, points, cfg, out_dir, None)
    heavy = sorted(m for m in ("torch", "tensorflow", "tf_keras") if m in sys.modules)
    print(f"Heavy frameworks imported during the replot: {heavy or 'none'}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _check_cwd()
    if args.replot is not None:
        replot(args)
        return
    if args.config is None:
        sys.exit("ERROR: a benchmark TOML is required (except with --replot)")
    run(args)


if __name__ == "__main__":
    main()
