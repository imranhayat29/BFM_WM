"""
Outputs of the a-priori test scripts (test_turb.py, test_lamturb.py)

Figures (spatial maps, C_f line plots, PDFs, smoothed tripcolor maps) from the `[plots]`
tables, `summary.json` and the optional `arrays.npz`. Truth enters only as evaluation data
(percent errors, truth C_f markers); the station selection of the line plots may use the
dataset u_tau in 'yplus' mode, which only picks plotting points.
"""
import datetime
import os
import sys
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from src_err.data import DatasetArrays
from src_err.evaluation import summarize, write_summary
from src_err.gating import blend_diagnostics
from src_err.pipeline import OOD_NAME, input_consistency, ood_and_nearwall_masks, per_point_arrays

# Spatial-map / tripcolor field key (src_err.plotting.FIELD_SPECS) -> pipeline output key
MAP_FIELDS = {
    "mu_pct": "mu_pct",
    "baseline": "e_base_pct",
    "corrected": "e_final_pct",
    "eqwm": "e_eqwm_pct",
    "sigma_pct": "sigma_pct",
    "confidence": "confidence",
}


def print_test_config(cfg: Mapping[str, Any], out_dir: str, gpu: int, max_samples: int) -> None:
    """
    Print the resolved test configuration

    Args:
        cfg: Validated test config
        out_dir: Output directory
        gpu: GPU index (-1 = CPU)
        max_samples: Sample cap (0 = none)
    """
    d, g = cfg["data"], cfg["gates"]
    print("=" * 80)
    print(f"Test configuration: {cfg.get('config_file')} (variant {cfg['variant']})")
    print(f"  datasets      : {d['Datasets']}")
    print(f"  window        : upy {d['upy']} (>= 1: no upper cut), downy {d['downy']}; "
          f"stride {d['Stride']}; max samples {max_samples or 'all'}")
    if d["Files"]:
        print(f"  files         : {d['Files']}")
    print(f"  EQWM law      : {cfg['eqwm']['Law']};  confidence covariance: {cfg['confidence']['Covariance']}")
    a = g["Aleatoric"]
    print(f"  gates         : aleatoric {a['Type']} {a['Threshold']} +- {a['Delta']}, "
          f"confidence {g['Confidence']['Threshold']} +- {g['Confidence']['Delta']}, "
          f"y+ {g['Yplus']['Threshold']} +- {g['Yplus']['Delta']}, "
          f"low value {g['LowValue']['Threshold']} +- {g['LowValue']['Delta']}")
    m = cfg["metrics"]["LowValue"]
    print(f"  metrics       : low value {m['Threshold']} +- {m['Delta']} on |pi_out,true|")
    if "switch" in cfg:
        print(f"  Kt/Km switch  : laminar iff kt_km <= {cfg['switch']['KtKmThreshold']}")
    print(f"  output        : {out_dir};  formats {cfg['general']['Formats']};  device "
          f"{'CPU' if gpu < 0 else 'GPU ' + str(gpu)}")
    print("=" * 80)


def _xlabel(variant: str) -> str:
    return r"$x/c$" if variant == "lamturb" else r"$x$"


def make_test_figures(out: Mapping[str, np.ndarray], arrays: DatasetArrays, class_names: Sequence[str],
                      labels: np.ndarray, cfg: Mapping[str, Any], out_dir: str) -> List[str]:
    """
    All figure families switched on in `[plots]`

    Args:
        out: Pipeline output (predict_turb / predict_lamturb)
        arrays: The evaluated DatasetArrays (x, U, delta)
        class_names: Plot class names (ending with 'OOD'), see pipeline.class_names_for_plot
        labels: Labels indexing `class_names` (labels_posthoc / labels_composite)
        cfg: Test config
        out_dir: Output directory

    Returns:
        Figure families written
    """
    import matplotlib.pyplot as plt
    from src_err import plotting
    from src_err.selection import select_stations

    plots = cfg["plots"]
    formats = cfg["general"]["Formats"]
    variant = cfg["variant"]
    xlabel = _xlabel(variant)
    x = np.asarray(arrays.x, dtype=float)
    n1, nu = arrays.U[:, 0], arrays.U[:, 2]
    n1_over_delta = n1 / arrays.delta
    ood, nearwall = ood_and_nearwall_masks(out, cfg["gates"])
    fields = {k: out[v] for k, v in MAP_FIELDS.items() if v in out}
    done: List[str] = []
    if not np.all(np.isfinite(x)):
        print(f"WARNING: {int(np.sum(~np.isfinite(x)))} points without an x coordinate are not plotted")

    # One n1 station per x (line plots, overlay on the maps)
    cf_opts = plots["cf_lines"]
    sel_idx = None
    if plots["CfLines"] or plots["maps"]["ShowSelected"] or plots["tripcolor"]["ShowSelected"]:
        try:
            _, _, sel_idx = select_stations(x, arrays.U, cf_opts)
            print(f"Selected {len(sel_idx)} stations ({cf_opts['Selection']})")
        except ValueError as exc:
            print(f"WARNING: station selection failed ({exc}); no line plots / overlays")

    if plots["SpatialMaps"]:
        print("=" * 80 + "\nSpatial maps")
        plotting.plot_spatial_maps(x, n1, arrays.delta, fields, labels, class_names, plots["maps"],
                                   out_dir=out_dir, formats=formats, selected_idx=sel_idx, ood_mask=ood,
                                   xlabel=xlabel)
        done.append("map")
        if variant == "lamturb":
            # LT-NB cell 75: the same fields against raw n1, zoomed on the thin laminar region
            plotting.plot_spatial_maps(x, n1, arrays.delta, fields, labels, class_names, plots["maps"],
                                       out_dir=out_dir, formats=formats, selected_idx=sel_idx, ood_mask=ood,
                                       raw_n1=True, xlabel=xlabel)
            done.append("map_rawn1")

    if plots["CfLines"] and sel_idx is not None and len(sel_idx) > 1:
        print("=" * 80 + "\nC_f line plots")
        s = sel_idx
        cf = {"baseline": plotting.piout_to_cf(out["y_base"][s], n1[s], nu[s]),
              "corrected": plotting.piout_to_cf(out["y_final"][s], n1[s], nu[s]),
              "eqwm": plotting.piout_to_cf(out["y_eqwm"][s], n1[s], nu[s])}
        truth_all = plotting.piout_to_cf(out["y_true"], n1, nu)  # evaluation only
        ok = np.isfinite(x) & np.isfinite(truth_all)
        plotting.plot_cf_lines(x[s], cf, x_all=x[ok], truth_all=truth_all[ok],
                               sigma_pct_sel=out["sigma_pct"][s], conf_sel=out["confidence"][s],
                               opts=cf_opts, out_dir=out_dir, formats=formats, xlabel=xlabel)
        done.append("cf_lines")

    if plots["Pdfs"]:
        print("=" * 80 + "\nPDFs")
        plotting.plot_pdfs({"baseline": out["e_base_pct"], "corrected": out["e_final_pct"],
                            "eqwm": out["e_eqwm_pct"]}, out["confidence"], plots["pdfs"],
                           out_dir=out_dir, formats=formats)
        done.append("pdf")

    if plots["Tripcolor"]:
        print("=" * 80 + "\nSmoothed tripcolor maps")
        tri = plots["tripcolor"]
        plotting.plot_tripcolor_maps(x, n1_over_delta, fields, labels, class_names,
                                     masks={"ood": ood, "nearwall": nearwall}, opts=tri,
                                     out_dir=out_dir, formats=formats, selected_idx=sel_idx,
                                     show_selected=tri["ShowSelected"], xlabel=xlabel)
        done.append("tri")
    plt.close("all")
    return done


def _ood_class_counts(labels: np.ndarray, class_names: Sequence[str], ood: np.ndarray) -> Dict[str, int]:
    """Points per plotted class with low-confidence points counted as 'OOD'."""
    lab = np.asarray(labels, dtype=np.int64).copy()
    names = list(class_names)
    lab[ood] = names.index(OOD_NAME)
    counts = np.bincount(lab, minlength=len(names))
    return {n: int(counts[i]) for i, n in enumerate(names)}


def _dataset_info(arrays: DatasetArrays) -> Dict[str, Any]:
    keys = arrays.keys()
    return {"keys": keys, "source_files": arrays.source_file.split(", "),
            "points": {k: int(np.sum(arrays.dataset_key == k)) for k in keys}}


def build_test_summary(out: Mapping[str, np.ndarray], arrays: DatasetArrays, cfg: Mapping[str, Any],
                       models: Mapping[str, Any], class_names: Sequence[str], labels: np.ndarray,
                       cli: Mapping[str, Any], figures: Sequence[str],
                       timings: Mapping[str, float]) -> Dict[str, Any]:
    """
    Contents of summary.json

    Args:
        out: Pipeline output
        arrays: Evaluated DatasetArrays
        cfg: Test config
        models: {branch: BranchModels}
        class_names: Plot class names (ending with 'OOD')
        labels: Labels indexing class_names
        cli: Command-line arguments
        figures: Figure families written
        timings: Wall times per stage (s)

    Returns:
        Summary dict (statistics from evaluation.summarize plus run metadata)
    """
    from src_err.config import config_to_metadata
    from src_err.pipeline import posthoc_class_names
    from src_err.utils import git_commit_hash, software_versions

    ood, nearwall = ood_and_nearwall_masks(out, cfg["gates"])
    model_info = {}
    for branch, m in models.items():
        model_info[branch] = {
            "paths": dict(m.paths),
            "sngp_num_classes": m.sngp.num_classes,
            "sngp_num_classes_train": m.sngp.num_classes_train,
            "sngp_trained_classes": m.sngp.class_names,
            "posthoc_groups": m.posthoc_groups,
            "display_names": m.display_names,
            "posthoc_class_names": posthoc_class_names(m),
            "baseline_window": {"upy": m.baseline.config.get("data", {}).get("upy"),
                                "downy": m.baseline.config.get("data", {}).get("downy")},
        }
    extra: Dict[str, Any] = {
        "script": os.path.basename(sys.argv[0]),
        "variant": cfg["variant"],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "cli": dict(cli),
        "config_file": cfg.get("config_file"),
        "config": config_to_metadata({k: v for k, v in cfg.items() if k != "config_file"}),
        "models": model_info,
        "data": dict(_dataset_info(arrays), input_inconsistent_fraction=input_consistency(
            arrays, next(iter(models.values())).baseline.config, verbose=False)),
        "eqwm_law": cfg["eqwm"]["Law"],
        "confidence_covariance": cfg["confidence"]["Covariance"],
        "gates": cfg["gates"],
        "metrics": cfg["metrics"],
        "class_names_plot": list(class_names),
        "class_counts_plot": _ood_class_counts(labels, class_names, ood),
        "fallback_masks": {"n_ood": int(ood.sum()), "frac_ood": float(ood.mean()),
                           "n_nearwall": int(nearwall.sum()), "frac_nearwall": float(nearwall.mean())},
        "figures": list(figures),
        "timings_s": {k: round(float(v), 2) for k, v in timings.items()},
        "versions": software_versions(),
        "git_commit": git_commit_hash("."),
    }
    if cfg["variant"] == "lamturb":
        lam = np.asarray(out["laminar"], dtype=bool)
        extra["lamturb"] = {
            "kt_km_threshold": cfg["switch"]["KtKmThreshold"],
            "n_laminar": int(lam.sum()), "n_turbulent": int((~lam).sum()),
            "frac_laminar": float(lam.mean()),
            "kt_km_source": {k: ("column" if np.any(np.isfinite(arrays.kt_km[arrays.dataset_key == k]))
                                 else "name inference") for k in arrays.keys()},
            "blend_laminar": blend_diagnostics(out["alpha_nw"][lam], out["alpha_c"][lam], out["alpha_a"][lam],
                                               verbose=False),
            "blend_turbulent": blend_diagnostics(out["alpha_nw"][~lam], out["alpha_c"][~lam],
                                                 out["alpha_a"][~lam], verbose=False),
        }
    names_no_ood = [n for n in class_names if n != OOD_NAME]
    return summarize(out, dataset_key=arrays.dataset_key, class_names=names_no_ood,
                     label_key="labels_posthoc", extra=extra)


def save_test_outputs(out: Mapping[str, np.ndarray], arrays: DatasetArrays, summary: Mapping[str, Any],
                      class_names: Sequence[str], cfg: Mapping[str, Any], out_dir: str) -> None:
    """
    Write summary.json and, with [general].SaveArrays, arrays.npz

    Args:
        out: Pipeline output
        arrays: Evaluated DatasetArrays
        summary: build_test_summary output
        class_names: Plot class names
        cfg: Test config
        out_dir: Output directory
    """
    write_summary(os.path.join(out_dir, "summary.json"), summary)
    if cfg["general"]["SaveArrays"]:
        n = len(arrays)
        data = per_point_arrays(out, n)
        data.update(x=arrays.x, n1=arrays.U[:, 0], u1=arrays.U[:, 1], nu=arrays.U[:, 2], delta=arrays.delta,
                    X=arrays.X, dataset_key=arrays.dataset_key.astype(str),
                    class_names_plot=np.asarray(class_names, dtype=str))
        if "composite_names" in out:
            data["composite_names"] = np.asarray(out["composite_names"], dtype=str)
        path = os.path.join(out_dir, "arrays.npz")
        np.savez_compressed(path, **data)
        print(f"Saved: {path} ({len(data)} arrays)")
