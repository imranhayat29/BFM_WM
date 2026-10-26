"""
Data loading for the error model, the SNGP and the test scripts

- the baseline train/valid split comes from the parent WallModel (`model.load_data()`),
  seeded, or re-sliced with the split indices stored in new baseline checkpoints;
- additional and test datasets are read from the h5 files of the parent registry with the
  same lookup as `WallModelDataHandler.load_external_dataset`, but with columns selected by
  name and the n1/delta window applied consistently (`downy` honoured even if `upy >= 1`).

Array conventions: X (N,3) = [pi1, pi2, pi3] (InputScaling-2 columns),
y (N,) = pi_out = u_tau*n1/nu (>= 0), U (N,4) = [n1, u1, nu, u_tau].
"""
import os
import pickle
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.data_handler import COLUMN_MAP
# resolve_dataset_file lives in the torch-free utils, so that the scripts can check the dataset
# files before torch is imported; it is re-exported here for the loaders below and for callers
from src_err.utils import resolve_dataset_file  # noqa: F401

# Keys of the checkpoint [data] table that are settings, not datasets (`upy = 1` is an int == 1)
_NON_DATASET_KEYS = {"TBL", "upy", "downy", "near_wall_threshold", "partition"}

# unnormalized_inputs columns forming U, with fallbacks to flow_type columns (time-dependent cases)
_U_COLUMNS = ["y", "u1", "nu", "utau"]
_U_FALLBACKS = {"nu": "nu", "utau": "utau_inst"}


@dataclass
class DatasetArrays:
    """
    Per-point arrays of one or several datasets

    Attributes:
        X: (N, D) inputs, D = 3 for InputScaling 2
        y: (N,) target pi_out
        U: (N, 4) [n1, u1, nu, u_tau]
        x: (N,) streamwise coordinate ('x' or 'x_over_c'; NaN if absent)
        delta: (N,) boundary-layer thickness ('delta'; NaN if absent)
        case_name: (N,) flow_type 'case_name' (SNGP class names of additional datasets)
        kt_km: (N,) 'kt_km' column (NaN if absent; see ktkm_values)
        dataset_key: (N,) registry key of each point
        source_file: h5 file (comma-separated list after concat)
    """
    X: np.ndarray
    y: np.ndarray
    U: np.ndarray
    x: np.ndarray
    delta: np.ndarray
    case_name: np.ndarray
    kt_km: np.ndarray
    dataset_key: np.ndarray
    source_file: str = ""

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> "DatasetArrays":
        """
        Rows selected by an index or boolean mask

        Args:
            idx: Integer indices or boolean mask

        Returns:
            New DatasetArrays
        """
        kw = {f.name: getattr(self, f.name)[idx] for f in fields(self) if f.name != "source_file"}
        return DatasetArrays(source_file=self.source_file, **kw)

    @classmethod
    def concat(cls, parts: Sequence["DatasetArrays"]) -> "DatasetArrays":
        """
        Concatenate datasets in the given order

        Args:
            parts: DatasetArrays to stack

        Returns:
            New DatasetArrays
        """
        if not parts:
            raise ValueError("Nothing to concatenate")
        kw = {f.name: np.concatenate([getattr(p, f.name) for p in parts])
              for f in fields(cls) if f.name != "source_file"}
        files = list(dict.fromkeys(p.source_file for p in parts))
        return cls(source_file=", ".join(files), **kw)

    def keys(self) -> List[str]:
        """Dataset keys in order of first appearance."""
        return list(dict.fromkeys(self.dataset_key.tolist()))


@dataclass
class BaselineSplit:
    """Train/valid data of the baseline checkpoint (raw, un-normalised inputs)."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    names_train: np.ndarray  # dataset key per sample (model.flow_type)
    names_valid: np.ndarray
    train_index: np.ndarray
    valid_index: np.ndarray
    source: str  # 'checkpoint' | 'seeded_resplit'


@dataclass
class ExtraSplit:
    """Train/valid split of the concatenated additional datasets."""
    train: DatasetArrays
    valid: DatasetArrays
    train_index: np.ndarray
    valid_index: np.ndarray


# ---------------------------------------------------------------------------------------
# Baseline checkpoint and its training data
# ---------------------------------------------------------------------------------------


def load_baseline_model(checkpoint: str, device: Any):
    """
    Load a baseline WallModel checkpoint

    Args:
        checkpoint: Path relative to ./models (e.g. baseline/<file>.pth) or any path
        device: torch.device or string; map_location also handles checkpoints saved on cuda:1

    Returns:
        WallModel in eval mode (its `config` is the checkpoint's embedded config)
    """
    from src.wall_model import WallModel
    from src_err.utils import model_hint, resolve_model_path

    path = resolve_model_path(checkpoint)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {model_hint(checkpoint)}")
    print("=" * 80)
    print(f"Loading baseline checkpoint {path}")
    try:
        model = WallModel.load(path, device=str(device))
    except pickle.UnpicklingError:
        # torch >= 2.6 loads with weights_only=True, which rejects numpy arrays (e.g. split
        # indices saved as numpy); rebuild like WallModelBase.load from the trusted local file.
        ckpt = torch.load(path, map_location=str(device), weights_only=False)
        model = WallModel(ckpt["config"])
        model.device = torch.device(str(device))
        model._create_model()
        model.model.load_state_dict(ckpt["model_state_dict"])
        model.model.to(model.device)
        model.model.eval()
    cfg = model.config
    data_cfg = cfg.get("data", {})
    print(f"  config file at training : {cfg.get('config_file')}")
    print(f"  training datasets       : {training_dataset_keys(cfg)}")
    print(f"  upy / downy             : {data_cfg.get('upy')} / {data_cfg.get('downy')}")
    print(f"  near_wall_threshold     : {data_cfg.get('near_wall_threshold', 0.0)}")
    print(f"  InputScaling            : {cfg.get('model', {}).get('inputs', {}).get('InputScaling', 1)}")
    print(f"  HiddenLayers            : {cfg.get('model', {}).get('HiddenLayers')}")
    print("=" * 80)
    return model


def baseline_output(model: Any, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    """
    Raw baseline network output pi_out in batches (torch, no-grad, on the model's device)

    The error-model targets (eps = y - output) and the inference pipeline both use this
    function, so the error model is trained on exactly the quantity it later corrects.

    Args:
        model: WallModel (its `.model` is the torch network; inputs are used raw, as the parent)
        X: (N, D) inputs in the checkpoint's input space
        batch: Rows per forward pass

    Returns:
        (N,) float64 raw outputs in pi_out units (no magnitude taken; may be negative)
    """
    net = model.model
    net.eval()
    dev = next(net.parameters()).device
    X = np.asarray(X)
    out = np.empty(len(X), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(X), int(batch)):
            xb = torch.from_numpy(np.ascontiguousarray(X[s:s + batch])).float().to(dev)
            out[s:s + batch] = net(xb).reshape(-1).cpu().numpy()
    if model.config.get("training", {}).get("LogTransform", False):
        # Parent test path (src/wall_model.py): outputs trained on log(1 + y)
        out = np.exp(out) - 1.0
    return out


def training_dataset_keys(model_config: Dict[str, Any]) -> List[str]:
    """
    Baseline training datasets in the order of the parent loader (= SNGP class order)

    Args:
        model_config: Checkpoint config (model.config)

    Returns:
        Non-TBL [data] keys with value == 1 in dict order, then TBL_<angle> per TBL angle
    """
    data_cfg = model_config.get("data", {})
    keys = [k for k, v in data_cfg.items()
            if k not in _NON_DATASET_KEYS and isinstance(v, (int, float)) and v == 1]
    tbl = data_cfg.get("TBL", [])
    if isinstance(tbl, list):
        keys.extend(f"TBL_{a}" for a in tbl)
    return keys


def _checkpoint_split_indices(checkpoint_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Split indices stored by new baseline checkpoints (None for old checkpoints)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tr, va = ckpt.get("train_index"), ckpt.get("valid_index")
    if tr is None or va is None:
        return None
    return {"train": np.asarray(tr, dtype=np.int64), "valid": np.asarray(va, dtype=np.int64)}


def baseline_training_split(model, checkpoint_path: str, seed: int) -> BaselineSplit:
    """
    Train/valid data of the baseline, reproducibly

    The parent `model.load_data()` reads the checkpoint's datasets and splits them with an
    unseeded `np.random.permutation`; seeding numpy first makes that split reproducible.
    If the checkpoint stores its own split indices (new baselines) and they fit the data,
    they are used instead, so the SNGP/error model see exactly the baseline's train/valid sets.

    Args:
        model: WallModel from load_baseline_model
        checkpoint_path: Checkpoint file (path relative to ./models or any path), read for stored indices
        seed: Seed of the parent split

    Returns:
        BaselineSplit (unnormalised inputs are not reloaded: they are not needed)
    """
    from src_err.utils import resolve_model_path

    np.random.seed(int(seed))
    model.load_data()
    names = getattr(model, "flow_type", None)
    if names is None:  # parent without the flow_type port
        names = model.data_handler.flow_type
    names = np.asarray(names).astype(str)
    X = np.asarray(model.input)
    y = np.asarray(model.output).reshape(-1)
    n = len(X)

    stored = _checkpoint_split_indices(resolve_model_path(checkpoint_path))
    if stored is not None:
        tr, va = stored["train"], stored["valid"]
        if len(tr) + len(va) == n and (n == 0 or max(tr.max(initial=-1), va.max(initial=-1)) < n):
            source = "checkpoint"
        else:
            print(f"WARNING: checkpoint split indices ({len(tr)} + {len(va)}) do not match the "
                  f"{n} loaded samples (data changed?); using the seeded re-split")
            stored = None
    if stored is None:
        tr = np.asarray(model.train_index, dtype=np.int64)
        va = np.asarray(model.valid_index, dtype=np.int64)
        source = "seeded_resplit"

    print(f"Baseline split ({source}, seed {seed}): {len(tr)} train / {len(va)} valid of {n} samples")
    return BaselineSplit(X_train=X[tr], y_train=y[tr], X_valid=X[va], y_valid=y[va],
                         names_train=names[tr], names_valid=names[va],
                         train_index=tr, valid_index=va, source=source)


# ---------------------------------------------------------------------------------------
# External (additional / test) datasets
# ---------------------------------------------------------------------------------------


def _flow_column(flow_df: pd.DataFrame, names: Sequence[str], dtype=float) -> Optional[np.ndarray]:
    for name in names:
        if name in flow_df.columns:
            return flow_df[name].to_numpy().astype(dtype)
    return None


def load_dataset(key: str, model_config: Dict[str, Any], upy: float, downy: float,
                 files_override: Optional[Dict[str, str]] = None) -> DatasetArrays:
    """
    Load one dataset with the n1/delta window and the baseline's input columns

    Window (ratio = n1/delta): keep ratio <= upy only if upy < 1 (parent semantics: upy >= 1
    means no upper cut); keep ratio >= downy whenever downy > 0 (fix: the parent ignores
    downy when upy >= 1). Near-wall mode (`near_wall_threshold > 0` in the checkpoint config)
    mirrors the parent: keep u1_y_over_nu <= threshold and use sign*log1p(|x|) inputs.

    Args:
        key: Dataset key (registry key, ./data/<key>_data.h5, or [data.Files] entry)
        model_config: Baseline checkpoint config (InputScaling, near_wall_threshold)
        upy: Upper n1/delta cut (>= 1: none)
        downy: Lower n1/delta cut (0: none)
        files_override: Optional {key: path}

    Returns:
        DatasetArrays (targets as stored, i.e. magnitudes)
    """
    path = resolve_dataset_file(key, files_override)
    inputs_df = pd.read_hdf(path, key="inputs")
    output = pd.read_hdf(path, key="output").to_numpy()[:, 0].astype(float)
    unnorm_df = pd.read_hdf(path, key="unnormalized_inputs")
    flow_df = pd.read_hdf(path, key="flow_type")
    n_raw = len(output)

    # U = [n1, u1, nu, u_tau] by column name (fallbacks for the time-dependent cases)
    u_cols = []
    for name in _U_COLUMNS:
        if name in unnorm_df.columns:
            u_cols.append(unnorm_df[name].to_numpy().astype(float))
        elif name in _U_FALLBACKS and _U_FALLBACKS[name] in flow_df.columns:
            u_cols.append(flow_df[_U_FALLBACKS[name]].to_numpy().astype(float))
        else:
            raise ValueError(f"Dataset '{key}' ({path}): column '{name}' not found in unnormalized_inputs "
                             f"{list(unnorm_df.columns)}")
    U = np.column_stack(u_cols)
    n1 = U[:, 0]
    delta = _flow_column(flow_df, ["delta", "delta99"])
    xcoord = _flow_column(flow_df, ["x", "x_over_c"])
    case = _flow_column(flow_df, ["case_name"], dtype=str)
    kt_km = _flow_column(flow_df, ["kt_km"])
    nan = np.full(n_raw, np.nan)
    delta = nan.copy() if delta is None else delta
    xcoord = nan.copy() if xcoord is None else xcoord
    case = np.full(n_raw, key) if case is None else case
    kt_km = nan.copy() if kt_km is None else kt_km

    # n1/delta window, written as n1 vs ratio*delta like the parent (identical rows)
    mask = np.ones(n_raw, dtype=bool)
    if upy < 1.0 or downy > 0.0:
        if np.all(np.isnan(delta)):
            raise ValueError(f"Dataset '{key}': no 'delta' column in flow_type, cannot apply the n1/delta window")
        if upy < 1.0:
            mask &= n1 <= upy * delta
            # parent applies the lower cut inside the upy branch even for downy = 0 (drops n1 < 0 / NaN delta)
            mask &= n1 >= downy * delta
        else:
            mask &= n1 >= downy * delta

    # Near-wall mode of the baseline (BFM_WM version: no u1 > 0 mask, reversed flow kept)
    near_wall = float(model_config.get("data", {}).get("near_wall_threshold", 0.0) or 0.0)
    if near_wall > 0.0:
        mask &= inputs_df["u1_y_over_nu"].to_numpy() <= near_wall

    input_scaling = model_config.get("model", {}).get("inputs", {}).get("InputScaling", 1)
    columns = COLUMN_MAP.get(input_scaling, COLUMN_MAP[1])
    missing = [c for c in columns if c not in inputs_df.columns]
    if missing:
        raise ValueError(f"Dataset '{key}' ({path}) lacks input columns {missing} for InputScaling={input_scaling}")
    X = inputs_df[columns].to_numpy().astype(float)[mask]
    if near_wall > 0.0:
        X = np.sign(X) * np.log1p(np.abs(X))

    n = int(mask.sum())
    window = []
    if upy < 1.0:
        window.append(f"n1/delta <= {upy}")
    if downy > 0.0:
        window.append(f"n1/delta >= {downy}")
    if near_wall > 0.0:
        window.append(f"u1_y_over_nu <= {near_wall}")
    print(f"Loaded dataset {key}: {n_raw} -> {n} points ({', '.join(window) or 'no filter'}) from {path}")
    return DatasetArrays(X=X, y=output[mask], U=U[mask], x=xcoord[mask], delta=delta[mask],
                         case_name=case[mask].astype(str), kt_km=kt_km[mask],
                         dataset_key=np.full(n, key, dtype=object).astype(str), source_file=path)


def load_datasets(keys: Sequence[str], model_config: Dict[str, Any], upy: float, downy: float,
                  files_override: Optional[Dict[str, str]] = None, stride: int = 1,
                  max_samples: int = 0) -> DatasetArrays:
    """
    Load and concatenate several datasets

    Args:
        keys: Dataset keys (concatenated in this order)
        model_config: Baseline checkpoint config
        upy: Upper n1/delta cut (>= 1: none)
        downy: Lower n1/delta cut (0: none)
        files_override: Optional {key: path}
        stride: Keep every `stride`-th point of each dataset
        max_samples: > 0: deterministic evenly spaced subsample (np.linspace) of the concatenation

    Returns:
        DatasetArrays
    """
    if not keys:
        raise ValueError("No datasets given")
    stride = max(int(stride), 1)
    parts = []
    for key in keys:
        arr = load_dataset(key, model_config, upy, downy, files_override)
        if stride > 1:
            arr = arr.subset(np.arange(0, len(arr), stride))
        parts.append(arr)
    out = DatasetArrays.concat(parts)
    if max_samples and len(out) > max_samples:
        idx = np.linspace(0, len(out) - 1, int(max_samples), dtype=np.int64)
        print(f"Deterministically capping samples: {len(out)} -> {max_samples}")
        out = out.subset(idx)
    print(f"Loaded {len(keys)} dataset(s): {len(out)} points (stride {stride})")
    return out


def split_extra(arrays: DatasetArrays, train_ratio: float, seed: int) -> ExtraSplit:
    """
    Seeded train/valid split of the concatenated additional datasets

    This is the shared split of the notebooks: the SNGP and the error model use the same
    split when their AddDatasets, window, TrainRatio and Seed are equal.

    Args:
        arrays: Concatenated additional datasets
        train_ratio: Fraction of points in the train part
        seed: Seed of numpy.random.default_rng

    Returns:
        ExtraSplit
    """
    n = len(arrays)
    perm = np.random.default_rng(int(seed)).permutation(n)
    n_train = int(round(train_ratio * n))
    tr, va = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    # Sorted indices keep the dataset order; the membership is what the seed fixes
    print(f"Additional datasets split (seed {seed}): {len(tr)} train / {len(va)} valid")
    return ExtraSplit(train=arrays.subset(tr), valid=arrays.subset(va), train_index=tr, valid_index=va)


def infer_ktkm_from_name(name: str) -> float:
    """
    Kt/Km inferred from a flow name (LT-NB `infer_ktkm_from_flow_name`)

    Args:
        name: case_name or dataset key

    Returns:
        0.0 (laminar: 'laminar' in the name or 'fs_' prefix), 0.5 ('transition'), else 1.0
    """
    lower = str(name).lower()
    if "laminar" in lower or lower.startswith("fs_"):
        return 0.0
    if "transition" in lower:
        return 0.5
    return 1.0


def ktkm_values(arrays: DatasetArrays) -> np.ndarray:
    """
    Kt/Km per point: the 'kt_km' column where present, else inferred from the flow name

    The name rule uses case_name (as LT-NB, flow_type column 0), or the dataset key when the
    case name is empty.

    Args:
        arrays: DatasetArrays

    Returns:
        (N,) float array
    """
    kt = np.array(arrays.kt_km, dtype=float, copy=True)
    missing = np.isnan(kt)
    if missing.any():
        names = np.where(np.char.str_len(arrays.case_name.astype(str)) > 0, arrays.case_name, arrays.dataset_key)
        uniq, inv = np.unique(names[missing], return_inverse=True)
        kt[missing] = np.array([infer_ktkm_from_name(u) for u in uniq])[inv]
        for key in dict.fromkeys(arrays.dataset_key[missing].tolist()):
            vals = np.unique(kt[missing & (arrays.dataset_key == key)])
            print(f"  kt_km for dataset {key}: no column, inferred from the flow name -> {vals.tolist()}")
    return kt
