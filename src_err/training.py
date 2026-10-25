"""
Training orchestration of the SNGP confidence model and the error model

`run_branch` trains the SNGP and/or the error model of one baseline checkpoint (one branch):

1. load the baseline checkpoint and its train/valid split (`baseline_training_split`: stored
   indices of new checkpoints, otherwise the parent split seeded with [general].Seed);
2. load the additional datasets with the TOML n1/delta window and split them 80/20 with the
   seed (`split_extra`). SNGP and error model share these arrays and the split when their
   [data] specs and Seed are equal (the notebooks' shared split), otherwise each loads its own;
3. SNGP: class names (baseline: dataset key per sample, extras: h5 case_name) ->
   combined class map -> RemoveClasses -> MergeGroups (all by name); the baseline part is
   strided by DownsampleBaseline; z-score statistics from the full baseline-train data, or
   from (strided baseline + extras) train data if RecomputeNorm; train, save, diagnostics;
4. error model: targets eps = y - y_hat with the raw baseline prediction y_hat on the
   baseline train/valid and extras train/valid data; statistics over the combined train
   data; train (best-epoch restore, snapshots), save, diagnostics.

Both `train_turb.py` and `train_lamturb.py` (once per branch) call `run_branch`.

Smoke mode (`smoke=True`, login-node test): Epochs -> 2, SaveDir -> ./results/smoke/models,
Overwrite allowed, snapshots every epoch, and deterministic seeded row caps (20000 baseline
train, 5000 baseline valid, 5000 extras train, 5000 extras valid). The caps are applied
after the class map and the normalisation statistics have been computed from the full data,
so class maps and statistics are those of a full run; split_indices.npz always holds the full
splits (the caps are recorded in the metadata).
"""
import copy
import csv
import gc
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from src_err import utils
from src_err import classes as cls
from src_err import data as dat
from src_err.config import config_to_metadata

SMOKE_EPOCHS = 2
SMOKE_SAVE_DIR = os.path.join(".", "results", "smoke", "models")
SMOKE_RESULTS_DIR = os.path.join(".", "results", "smoke", "training")
RESULTS_DIR = os.path.join(".", "results", "training")
SMOKE_CAPS = {"baseline_train": 20000, "baseline_valid": 5000, "extra_train": 5000, "extra_valid": 5000}
BASELINE_BATCH = 65536
SNGP_PREDICT_BATCH = 512  # the GP layer forms a B x B covariance per batch in inference


@dataclass
class _Extras:
    """Additional datasets of one model: full arrays (class names) and their seeded split."""
    key: Tuple
    arrays: Optional[dat.DatasetArrays]
    split: Optional[dat.ExtraSplit]


# ---------------------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------------------


def smoke_config(cfg: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """
    Login-node smoke-test variant of a validated config

    Args:
        cfg: Validated SNGP or error-model branch config
        kind: 'sngp' or 'error'

    Returns:
        Deep copy with Epochs = 2, SaveDir = ./results/smoke/models, Overwrite = true
        (and SaveInterval = 1 for the error model)
    """
    out = copy.deepcopy(cfg)
    out["training"]["Epochs"] = SMOKE_EPOCHS
    out["general"]["SaveDir"] = SMOKE_SAVE_DIR
    out["general"]["Overwrite"] = True
    if kind == "error":
        out["training"]["Snapshots"]["SaveInterval"] = 1
    return out


def model_dir(cfg: Dict[str, Any]) -> str:
    """Output directory <SaveDir>/<ModelName> of a training config."""
    return os.path.join(cfg["general"]["SaveDir"], cfg["general"]["ModelName"])


def results_dir(cfg: Dict[str, Any], smoke: bool) -> str:
    """Diagnostics directory results/training/<ModelName> (results/smoke/training/... in smoke mode)."""
    return os.path.join(SMOKE_RESULTS_DIR if smoke else RESULTS_DIR, cfg["general"]["ModelName"])


def check_outputs(cfg: Optional[Dict[str, Any]], kind: str) -> None:
    """
    Fail fast (before any data is loaded) if a model would be overwritten without Overwrite

    Args:
        cfg: Branch config (None: nothing to check)
        kind: 'sngp' or 'error' (the error model's snapshot directory is checked as well)
    """
    if cfg is None or not cfg["general"]["Save"] or cfg["general"]["Overwrite"]:
        return
    paths = [model_dir(cfg)]
    if kind == "error" and cfg["training"]["Snapshots"]["SaveInterval"] > 0:
        paths.append(utils.snapshot_dir(cfg))
    for path in paths:
        if os.path.exists(path) and (not os.path.isdir(path) or os.listdir(path)):
            raise FileExistsError(f"{path} already exists; set [general] Overwrite = true to replace it, "
                                  "or choose another ModelName/SaveDir")


def _clear_old_snapshots(path: str) -> None:
    """With Overwrite, remove the snapshots of an earlier run (only ckpt_* SavedModel dirs)."""
    if not os.path.isdir(path):
        return
    entries = os.listdir(path)
    if all(e.startswith("ckpt_") for e in entries):
        import shutil
        shutil.rmtree(path)
        print(f"Removed the snapshots of an earlier run in {path}")
    else:
        print(f"WARNING: {path} holds files other than ckpt_* snapshots; left in place")


def _data_key(cfg: Dict[str, Any]) -> Tuple:
    """Everything that defines the additional-data arrays and their split."""
    d = cfg["data"]
    return (tuple(d["AddDatasets"]), float(d["upy"]), float(d["downy"]), float(d["TrainRatio"]),
            tuple(sorted(d["Files"].items())), int(cfg["general"]["Seed"]))


def _print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------------------


def _load_extras(cfg: Dict[str, Any], model_config: Dict[str, Any]) -> _Extras:
    """Additional datasets of one config (window from the TOML) and their seeded split."""
    d = cfg["data"]
    key = _data_key(cfg)
    if not d["AddDatasets"]:
        print("No additional datasets ([data] AddDatasets = [])")
        return _Extras(key=key, arrays=None, split=None)
    print(f"Additional datasets (upy = {d['upy']}, downy = {d['downy']}): {d['AddDatasets']}")
    arrays = dat.load_datasets(d["AddDatasets"], model_config, d["upy"], d["downy"],
                               files_override=d["Files"] or None)
    split = dat.split_extra(arrays, d["TrainRatio"], cfg["general"]["Seed"])
    return _Extras(key=key, arrays=arrays, split=split)


def _cap_index(n: int, cap: int, seed: int, salt: int) -> np.ndarray:
    """Deterministic seeded subsample (sorted) of range(n) with at most `cap` rows."""
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng([int(seed), int(salt)])
    return np.sort(rng.choice(n, size=cap, replace=False))


def baseline_predict(model, X: np.ndarray, batch: int = BASELINE_BATCH) -> np.ndarray:
    """
    Raw baseline output (pi_out, as trained; no magnitude taken) in batches

    Args:
        model: WallModel (its torch network is used in eval mode)
        X: (N, 3) inputs in the checkpoint's input space
        batch: Rows per forward pass

    Returns:
        (N,) float64 predictions
    """
    net = model.model
    net.eval()
    device = next(net.parameters()).device
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = torch.as_tensor(np.asarray(X[s:s + batch]), dtype=torch.float32, device=device)
            out[s:s + batch] = net(xb).reshape(-1).cpu().numpy()
    return out


def _release_baseline_data(model) -> None:
    """Drop the parent's copies of the baseline data (only the network is needed afterwards)."""
    for obj in (model, getattr(model, "data_handler", None)):
        if obj is None:
            continue
        for attr in ("input", "output", "input_train", "output_train", "input_valid", "output_valid", "flow_type"):
            if hasattr(obj, attr):
                setattr(obj, attr, None)
    gc.collect()


# ---------------------------------------------------------------------------------------
# SNGP
# ---------------------------------------------------------------------------------------


def _print_class_maps(combined: cls.ClassMap, removed: List[int], trained: cls.ClassMap,
                      display: List[str], num_classes: int) -> None:
    print("-" * 80)
    print(f"Combined classes ({len(combined)}), flow_to_label_combined:")
    for i, n in enumerate(combined.names):
        print(f"  {i:3d}  {n}" + ("   [removed]" if i in removed else ""))
    print(f"labels_to_remove: {removed}")
    print(f"Trained classes ({len(trained)}), flow_to_label_filt (display name):")
    for i, (n, d) in enumerate(zip(trained.names, display)):
        print(f"  {i:3d}  {n}" + (f"  ({d})" if d != n else ""))
    print(f"NumClasses = {num_classes}: {num_classes - len(trained)} untrained output units (OOD sinks)")
    print("-" * 80)


def _write_class_counts(out_dir: str, combined: cls.ClassMap, removed: List[int], to_trained: Dict[str, str],
                        counts_comb_tr: Dict[str, int], counts_comb_va: Dict[str, int],
                        trained: cls.ClassMap, display: List[str],
                        counts_tr: Dict[str, int], counts_va: Dict[str, int]) -> None:
    """class_counts.csv (trained classes) and class_counts_combined.csv (all combined classes)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "class_counts.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "class", "display_name", "n_train", "n_valid"])
        for i, n in enumerate(trained.names):
            w.writerow([i, n, display[i], counts_tr[n], counts_va[n]])
    path2 = os.path.join(out_dir, "class_counts_combined.csv")
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combined_label", "class", "status", "trained_class", "n_train", "n_valid"])
        for i, n in enumerate(combined.names):
            status = "removed" if i in removed else "kept"
            w.writerow([i, n, status, "" if i in removed else to_trained[n], counts_comb_tr[n], counts_comb_va[n]])
    print(f"Class counts written to {path} and {path2}")
    print(f"  {'label':>5}  {'class':<28} {'display name':<24} {'n_train':>10} {'n_valid':>10}")
    for i, n in enumerate(trained.names):
        print(f"  {i:5d}  {n:<28} {display[i]:<24} {counts_tr[n]:10d} {counts_va[n]:10d}")


def _display(class_map: cls.ClassMap, mapping: Dict[str, str]) -> List[str]:
    """Display names; keys of other maps (e.g. post-hoc group names) are expected, so no warning here."""
    return cls.display_names(class_map, {k: v for k, v in (mapping or {}).items() if k in class_map.index})


def _sngp_logits(model, Xn: np.ndarray, batch: int = SNGP_PREDICT_BATCH) -> np.ndarray:
    out = []
    for s in range(0, len(Xn), batch):
        out.append(np.asarray(model(Xn[s:s + batch], training=False)))
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def _sngp_diagnostics(model, Xva_n: np.ndarray, yva: np.ndarray, trained: cls.ClassMap, cfg: Dict[str, Any],
                      history: Dict[str, List[float]], out_dir: str) -> Dict[str, Any]:
    """Validation accuracy/confidence, confusion matrices (display names) and history plot."""
    from src_err import plotting
    from src_err import sngp
    from sklearn.metrics import confusion_matrix

    plotting.plot_sngp_history(history, out_dir=out_dir)
    if len(Xva_n) == 0:
        print("No validation data: confusion matrix skipped")
        return {}
    k, k_tr = int(cfg["classes"]["NumClasses"]), len(trained)
    probs = sngp.probabilities(_sngp_logits(model, Xva_n))
    conf = sngp.oneclass_confidence(probs, k_tr, k)
    pred = sngp.predict_labels(probs, k_tr)
    acc = float(np.mean(pred == yva))
    print(f"SNGP validation: accuracy {acc:.4f}, confidence mean {conf.mean():.4f} / median {np.median(conf):.4f} "
          f"({len(yva)} points)")
    display = _display(trained, cfg["classes"]["DisplayNames"])
    cm, ignored = sngp.confusion_matrix_known(yva, pred, k_tr)
    plotting.plot_confusion_matrix(cm, display, ignored_fraction=ignored, out_dir=out_dir, name="confusion_matrix")
    groups = cfg["classes"]["PosthocGroups"]
    if groups:
        _, merged_map, mapping = cls.posthoc_merge(probs, trained, groups)
        kk = len(merged_map)
        cm_p = confusion_matrix(mapping[yva], mapping[pred], labels=np.arange(kk))
        plotting.plot_confusion_matrix(cm_p, _display(merged_map, cfg["classes"]["DisplayNames"]),
                                       title="Normalized Confusion Matrix (Validation, post-hoc groups)",
                                       out_dir=out_dir, name="confusion_matrix_posthoc")
    return {"valid_accuracy": acc, "valid_confidence_mean": float(conf.mean())}


def train_sngp_branch(cfg: Dict[str, Any], model, ckpt_path: str, split: dat.BaselineSplit, extras: _Extras,
                      smoke: bool, out_dir: str) -> Dict[str, Any]:
    """
    Assemble the SNGP data, train, save and write diagnostics (step 3 of the module docstring)

    Args:
        cfg: Validated SNGP branch config (smoke-adjusted if smoke)
        model: Baseline WallModel (for its config)
        ckpt_path: Resolved baseline checkpoint path
        split: Baseline train/valid split
        extras: Additional datasets and their split
        smoke: Apply the smoke row caps
        out_dir: Diagnostics directory

    Returns:
        Summary dict (model path, class maps, sizes)
    """
    from src_err import sngp

    g, d, c = cfg["general"], cfg["data"], cfg["classes"]
    seed = int(g["Seed"])
    _print_header(f"SNGP {g['ModelName']}")
    stride = int(d["DownsampleBaseline"])

    # Class map from the full name lists (baseline keys in checkpoint order, then extras' case names)
    extra_names = extras.arrays.case_name if extras.arrays is not None else np.array([], dtype=str)
    combined = cls.combined_class_map(dat.training_dataset_keys(model.config), extra_names)
    removed = cls.labels_to_remove(combined, c["RemoveClasses"])

    # Parts: baseline strided by DownsampleBaseline (the extras never are)
    parts = {
        "baseline_train": (split.X_train[::stride], split.names_train[::stride]),
        "baseline_valid": (split.X_valid[::stride], split.names_valid[::stride]),
    }
    if extras.split is not None:
        parts["extra_train"] = (extras.split.train.X, extras.split.train.case_name)
        parts["extra_valid"] = (extras.split.valid.X, extras.split.valid.case_name)
    if stride > 1:
        print(f"DownsampleBaseline = {stride}: baseline train {len(split.X_train)} -> {len(parts['baseline_train'][0])}, "
              f"valid {len(split.X_valid)} -> {len(parts['baseline_valid'][0])}")

    def _assemble(which: str) -> Tuple[np.ndarray, np.ndarray]:
        keys = [k for k in parts if k.endswith(which)]
        X = np.concatenate([parts[k][0] for k in keys])
        names = np.concatenate([parts[k][1].astype(str) for k in keys])
        return X, cls.labels_from_names(names, combined)

    X_tr, lab_tr = _assemble("train")
    X_va, lab_va = _assemble("valid")

    # z-score statistics (before class removal, as the notebooks)
    if d["RecomputeNorm"]:
        mean_X, std_X = sngp.compute_norm_stats(X_tr)
        norm_src = "strided baseline + additional train data (RecomputeNorm)"
    else:
        mean_X, std_X = sngp.compute_norm_stats(split.X_train)
        norm_src = "full baseline train data"
    print(f"SNGP input statistics from {norm_src}: mean {mean_X.ravel()}, std {std_X.ravel()}")

    # Remove, then merge (by name; the renumbering follows the class map)
    keep_tr, lab_tr_r, map_r = cls.remove_classes(lab_tr, combined, c["RemoveClasses"])
    keep_va, lab_va_r, _ = cls.remove_classes(lab_va, combined, c["RemoveClasses"])
    y_tr, trained = cls.merge_groups(lab_tr_r, map_r, c["MergeGroups"])
    y_va, _ = cls.merge_groups(lab_va_r, map_r, c["MergeGroups"])
    X_tr, X_va = X_tr[keep_tr], X_va[keep_va]
    display = _display(trained, c["DisplayNames"])
    known = set(combined.names) | set(trained.names) | set(c["PosthocGroups"])
    unknown = [k for k in c["DisplayNames"] if k not in known]
    if unknown:
        print(f"WARNING: [classes] DisplayNames keys matching no class or group (ignored): {unknown}")
    _print_class_maps(combined, removed, trained, display, int(c["NumClasses"]))
    if int(c["NumClasses"]) <= len(trained):
        raise ValueError(f"[classes] NumClasses = {c['NumClasses']} must exceed the {len(trained)} trained classes")
    if c["PosthocGroups"]:
        cls.posthoc_merge(np.zeros((1, int(c["NumClasses"]))), trained, c["PosthocGroups"])  # validate names now

    # Trained class of every combined class (for the count table)
    m_lab, m_map = cls.merge_groups(np.arange(len(map_r)), map_r, c["MergeGroups"])
    to_trained = {n: m_map.names[m] for n, m in zip(map_r.names, m_lab)}
    counts_tr, counts_va = cls.class_counts(y_tr, trained), cls.class_counts(y_va, trained)
    _write_class_counts(out_dir, combined, removed, to_trained, cls.class_counts(lab_tr, combined),
                        cls.class_counts(lab_va, combined), trained, display, counts_tr, counts_va)

    smoke_info = None
    if smoke:
        # Caps per part, after class map and statistics: positions of the kept rows in each part
        part_of_tr = np.concatenate([np.full(len(parts[k][0]), i) for i, k in enumerate(parts) if k.endswith("train")])[keep_tr]
        part_of_va = np.concatenate([np.full(len(parts[k][0]), i) for i, k in enumerate(parts) if k.endswith("valid")])[keep_va]
        sel_tr, sel_va = [], []
        for i, k in enumerate(parts):
            rows = np.flatnonzero((part_of_tr if k.endswith("train") else part_of_va) == i)
            sub = rows[_cap_index(len(rows), SMOKE_CAPS[k], seed, i)]
            (sel_tr if k.endswith("train") else sel_va).append(sub)
        sel_tr, sel_va = np.sort(np.concatenate(sel_tr)), np.sort(np.concatenate(sel_va))
        print(f"Smoke caps {SMOKE_CAPS}: train {len(y_tr)} -> {len(sel_tr)}, valid {len(y_va)} -> {len(sel_va)}")
        X_tr, y_tr, X_va, y_va = X_tr[sel_tr], y_tr[sel_tr], X_va[sel_va], y_va[sel_va]
        smoke_info = {"caps": dict(SMOKE_CAPS), "seed": seed, "n_train": int(len(y_tr)), "n_valid": int(len(y_va))}
        # Metadata counts describe the rows actually trained on (the CSV keeps the full-data counts)
        counts_tr, counts_va = cls.class_counts(y_tr, trained), cls.class_counts(y_va, trained)

    Xtr_n = sngp.normalize_inputs(X_tr, mean_X, std_X)
    Xva_n = sngp.normalize_inputs(X_va, mean_X, std_X)
    utils.set_seeds(seed)
    model_sngp, history = sngp.train_sngp(Xtr_n, y_tr, Xva_n, y_va, cfg)

    metadata = sngp.sngp_metadata(
        num_classes=int(c["NumClasses"]), class_map_combined=combined, labels_to_remove=removed,
        class_map_trained=trained, mean_X=mean_X, std_X=std_X,
        include_extra_data=extras.split is not None, merge_groups=bool(c["MergeGroups"]),
        recompute_norm=d["RecomputeNorm"], downsample=stride, cfg=cfg, baseline_checkpoint=ckpt_path,
        add_datasets=d["AddDatasets"], window={"upy": d["upy"], "downy": d["downy"]},
        remove_classes=c["RemoveClasses"], merge_groups_def=c["MergeGroups"], display_names=c["DisplayNames"],
        posthoc_groups=c["PosthocGroups"], seed=seed, split_source=split.source,
        n_train=len(y_tr), n_valid=len(y_va), class_counts_train=counts_tr, class_counts_valid=counts_va,
        versions=utils.software_versions(), git_commit=utils.git_commit_hash(),
        config=config_to_metadata(cfg))
    if smoke_info is not None:
        metadata["bfmwm_v2"]["smoke"] = smoke_info
    split_indices = {
        "baseline_train_index": split.train_index,
        "baseline_valid_index": split.valid_index,
        "extra_train_index": extras.split.train_index if extras.split is not None else None,
        "extra_valid_index": extras.split.valid_index if extras.split is not None else None,
    }
    if stride > 1:
        split_indices["baseline_stride"] = np.int64(stride)

    path = None
    if g["Save"]:
        path = sngp.save_sngp(model_sngp, model_dir(cfg), metadata, history=history,
                              split_indices=split_indices, overwrite=g["Overwrite"])
    else:
        print("[general] Save = false: SNGP not saved")
    diag = _sngp_diagnostics(model_sngp, Xva_n, y_va, trained, cfg, history, out_dir)
    del model_sngp
    gc.collect()
    return {"path": path, "num_classes": int(c["NumClasses"]), "num_classes_train": len(trained),
            "flow_to_label_combined": combined.to_dict(), "labels_to_remove": removed,
            "flow_to_label_filt": trained.to_dict(), "n_train": int(len(y_tr)), "n_valid": int(len(y_va)),
            "diagnostics_dir": out_dir, **diag}


# ---------------------------------------------------------------------------------------
# Error model
# ---------------------------------------------------------------------------------------


def train_error_branch(cfg: Dict[str, Any], model, ckpt_path: str, split: dat.BaselineSplit, extras: _Extras,
                       smoke: bool, out_dir: str) -> Dict[str, Any]:
    """
    Assemble the error-model data, train, save and write diagnostics (step 4 of the module docstring)

    Args:
        cfg: Validated error-model branch config (smoke-adjusted if smoke)
        model: Baseline WallModel (raw predictions)
        ckpt_path: Resolved baseline checkpoint path
        split: Baseline train/valid split
        extras: Additional datasets and their split
        smoke: Apply the smoke row caps
        out_dir: Diagnostics directory

    Returns:
        Summary dict (model path, best epoch, sizes)
    """
    from src_err import error_model as em
    from src_err import plotting

    g, d = cfg["general"], cfg["data"]
    seed = int(g["Seed"])
    _print_header(f"Error model {g['ModelName']}")

    # Raw baseline predictions (no magnitude: the targets are the error of the raw output, as the notebooks)
    parts = {"baseline_train": (split.X_train, split.y_train), "baseline_valid": (split.X_valid, split.y_valid)}
    if extras.split is not None:
        parts["extra_train"] = (extras.split.train.X, extras.split.train.y)
        parts["extra_valid"] = (extras.split.valid.X, extras.split.valid.y)
    errs = {}
    for k, (X, y) in parts.items():
        errs[k] = em.error_targets(y, baseline_predict(model, X))
        print(f"  {k:<15} {len(X):9d} points, error mean {errs[k].mean():+.4g}, std {errs[k].std():.4g}")

    tr_keys = [k for k in parts if k.endswith("train")]
    va_keys = [k for k in parts if k.endswith("valid")]
    X_tr = np.concatenate([parts[k][0] for k in tr_keys])
    e_tr = np.concatenate([errs[k] for k in tr_keys])
    X_va = np.concatenate([parts[k][0] for k in va_keys])
    e_va = np.concatenate([errs[k] for k in va_keys])
    norm_stats = em.compute_norm_stats(X_tr, e_tr)
    print(f"Error-model statistics over the combined train data ({len(X_tr)} points): "
          f"x_mean {norm_stats['x_mean'].ravel()}, x_std {norm_stats['x_std'].ravel()}, "
          f"y_mean {float(norm_stats['y_mean'][0]):.6g}, y_std {float(norm_stats['y_std'][0]):.6g}")

    smoke_info = None
    if smoke:
        def _cap(keys):
            sel, off = [], 0
            for k in keys:
                n = len(parts[k][0])
                sel.append(off + _cap_index(n, SMOKE_CAPS[k], seed, list(parts).index(k)))
                off += n
            return np.concatenate(sel)
        s_tr, s_va = _cap(tr_keys), _cap(va_keys)
        print(f"Smoke caps {SMOKE_CAPS}: train {len(X_tr)} -> {len(s_tr)}, valid {len(X_va)} -> {len(s_va)}")
        X_tr, e_tr, X_va, e_va = X_tr[s_tr], e_tr[s_tr], X_va[s_va], e_va[s_va]
        smoke_info = {"caps": dict(SMOKE_CAPS), "seed": seed, "n_train": int(len(X_tr)), "n_valid": int(len(X_va))}

    Xtr_n = em.normalize_inputs(X_tr, norm_stats["x_mean"], norm_stats["x_std"])
    Xva_n = em.normalize_inputs(X_va, norm_stats["x_mean"], norm_stats["x_std"])
    etr_n = em.normalize_targets(e_tr, norm_stats["y_mean"], norm_stats["y_std"])
    eva_n = em.normalize_targets(e_va, norm_stats["y_mean"], norm_stats["y_std"])

    snap_dir = None
    if g["Save"] and cfg["training"]["Snapshots"]["SaveInterval"] > 0:
        snap_dir = utils.snapshot_dir(cfg)
        if g["Overwrite"]:
            _clear_old_snapshots(snap_dir)
    utils.set_seeds(seed)
    model_err, history, best_epoch = em.train_error_model(Xtr_n, etr_n, Xva_n, eva_n, cfg, snapshot_dir=snap_dir,
                                                          norm_stats=norm_stats)
    epochs_run = len(history.get("loss", []))
    print(f"best_epoch = {best_epoch} (of {epochs_run} epochs run)")

    metadata = em.error_metadata(
        cfg, baseline_checkpoint=ckpt_path, add_datasets=d["AddDatasets"],
        window={"upy": d["upy"], "downy": d["downy"]}, seed=seed, split_source=split.source,
        n_train=len(X_tr), n_valid=len(X_va), best_epoch=best_epoch, epochs_run=epochs_run,
        versions=utils.software_versions(), git_commit=utils.git_commit_hash(), config=config_to_metadata(cfg))
    if smoke_info is not None:
        metadata["smoke"] = smoke_info
    split_indices = {
        "baseline_train_index": split.train_index,
        "baseline_valid_index": split.valid_index,
        "extra_train_index": extras.split.train_index if extras.split is not None else None,
        "extra_valid_index": extras.split.valid_index if extras.split is not None else None,
    }
    path = None
    if g["Save"]:
        path = em.save_error_model(model_err, model_dir(cfg), norm_stats, history=history, metadata=metadata,
                                   split_indices=split_indices, overwrite=g["Overwrite"])
    else:
        print("[general] Save = false: error model not saved")
    plotting.plot_error_history(history, best_epoch, out_dir=out_dir)
    del model_err
    gc.collect()
    return {"path": path, "best_epoch": int(best_epoch), "epochs_run": epochs_run,
            "n_train": int(len(X_tr)), "n_valid": int(len(X_va)), "diagnostics_dir": out_dir}


# ---------------------------------------------------------------------------------------
# Branch driver
# ---------------------------------------------------------------------------------------


def run_branch(sngp_cfg: Optional[Dict[str, Any]], error_cfg: Optional[Dict[str, Any]], device: Any,
               smoke: bool = False, branch_label: str = "turb") -> Dict[str, Any]:
    """
    Train the SNGP and/or the error model of one baseline checkpoint (steps 1-4 of the module docstring)

    Args:
        sngp_cfg: Validated SNGP branch config, or None
        error_cfg: Validated error-model branch config, or None
        device: torch device of the baseline (from utils.configure_devices)
        smoke: Login-node smoke test (see the module docstring)
        branch_label: Name used in the log ('turb', 'lam')

    Returns:
        {'sngp': summary or None, 'error': summary or None, 'extras_shared': bool, ...}
    """
    if sngp_cfg is None and error_cfg is None:
        raise ValueError("run_branch needs an SNGP and/or an error-model config")
    if smoke:
        sngp_cfg = smoke_config(sngp_cfg, "sngp") if sngp_cfg is not None else None
        error_cfg = smoke_config(error_cfg, "error") if error_cfg is not None else None
    cfgs = [c for c in (sngp_cfg, error_cfg) if c is not None]
    ckpts = {c["baseline"]["Checkpoint"] for c in cfgs}
    if len(ckpts) != 1:
        raise ValueError(f"[{branch_label}] the SNGP and error-model TOMLs use different baseline checkpoints: {sorted(ckpts)}")
    checkpoint = ckpts.pop()
    check_outputs(sngp_cfg, "sngp")
    check_outputs(error_cfg, "error")

    _print_header(f"Branch '{branch_label}': " + " + ".join(
        n for n, c in (("SNGP " + (sngp_cfg or {}).get("general", {}).get("ModelName", ""), sngp_cfg),
                       ("error model " + (error_cfg or {}).get("general", {}).get("ModelName", ""), error_cfg)) if c)
        + (" [SMOKE TEST]" if smoke else ""))

    seeds = [int(c["general"]["Seed"]) for c in cfgs]
    utils.set_seeds(seeds[0])
    ckpt_path = utils.resolve_model_path(checkpoint)
    model = dat.load_baseline_model(checkpoint, device)

    # Baseline split(s): one per distinct seed (a different seed re-reads the data)
    splits: Dict[int, dat.BaselineSplit] = {}
    for s in dict.fromkeys(seeds):
        splits[s] = dat.baseline_training_split(model, checkpoint, s)
    model_config = model.config
    _release_baseline_data(model)

    # Additional datasets: shared when the [data] specs and Seed agree
    extras_s = extras_e = None
    sngp_extra = None
    if sngp_cfg is not None:
        extras_s = _load_extras(sngp_cfg, model_config)
    if error_cfg is not None:
        if extras_s is not None and extras_s.key == _data_key(error_cfg):
            print("Additional datasets and split shared by the SNGP and the error model")
            extras_e = extras_s
        else:
            if extras_s is not None:
                print("SNGP and error model differ in [data]/Seed: loading the error model's additional datasets")
            extras_e = None  # loaded after the SNGP is trained, to keep the peak memory low

    result: Dict[str, Any] = {"branch": branch_label, "checkpoint": ckpt_path, "smoke": smoke,
                              "sngp": None, "error": None,
                              "extras_shared": sngp_cfg is not None and error_cfg is not None and extras_e is extras_s}
    if sngp_cfg is not None:
        s_seed = int(sngp_cfg["general"]["Seed"])
        result["sngp"] = train_sngp_branch(sngp_cfg, model, ckpt_path, splits[s_seed], extras_s, smoke,
                                           results_dir(sngp_cfg, smoke))
        result["sngp"]["n_extra"] = 0 if extras_s.arrays is None else len(extras_s.arrays)
    if error_cfg is not None:
        if extras_e is None:
            if extras_s is not None:
                # Keep only what the comparison below needs, then free the SNGP's extras
                sngp_extra = None if extras_s.split is None else \
                    (len(extras_s.arrays), extras_s.split.train_index, extras_s.split.valid_index)
                extras_s = None
                gc.collect()
            extras_e = _load_extras(error_cfg, model_config)
            if sngp_cfg is not None and sngp_extra is not None and extras_e.split is not None:
                same = (sngp_extra[0] == len(extras_e.arrays)
                        and np.array_equal(sngp_extra[1], extras_e.split.train_index)
                        and np.array_equal(sngp_extra[2], extras_e.split.valid_index))
                result["extras_identical_split"] = bool(same)
                print(f"Additional data: SNGP {sngp_extra[0]} rows, error model {len(extras_e.arrays)} rows; "
                      f"identical split indices: {same}")
        e_seed = int(error_cfg["general"]["Seed"])
        result["error"] = train_error_branch(error_cfg, model, ckpt_path, splits[e_seed], extras_e, smoke,
                                             results_dir(error_cfg, smoke))
        result["error"]["n_extra"] = 0 if extras_e.arrays is None else len(extras_e.arrays)

    _print_header(f"Branch '{branch_label}' done")
    for kind in ("sngp", "error"):
        r = result[kind]
        if r is not None:
            extra = f", best_epoch {r['best_epoch']}" if kind == "error" else \
                f", K = {r['num_classes']}, K_tr = {r['num_classes_train']}"
            print(f"  {kind:<5}: {r['path'] or '(not saved)'} (train {r['n_train']}, valid {r['n_valid']}{extra}); "
                  f"diagnostics in {r['diagnostics_dir']}")
    del model
    gc.collect()
    return result
