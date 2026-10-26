#!/usr/bin/env python3
# BFM_WM_v2 tools/: copied from BFM_WM/export_error_model_to_npz.py.
# Purpose (CharLES BFMv2 export): error-model SavedModel -> error_model_params.json/.npz, input of export_error_model_to_cuda_datafiles.py.
# Adaptations: sets TF_USE_LEGACY_KERAS=1 (setdefault) before TensorFlow can be imported; example usage updated to tools/. CLI, defaults and output formats are unchanged.
"""Export a TF SavedModel error model into a single .npz or .json for CUDA/C++ inference.

This exporter is designed to avoid TensorFlow Probability dependencies by reading
Dense layer kernels/biases directly from the SavedModel variable checkpoint.

It also bundles normalization stats from:
  <model_dir>/assets/error_norm_stats.json

Output formats:
    - NPZ: arrays under keys x_mean, x_std, y_mean, y_std, W_0, b_0, ...
    - JSON: a single object with normalization stats and a 'layers' array.

Dense kernel layout matches Keras: y = x @ W + b, so W has shape (in_dim, out_dim).

Example usage: (run from the BFM_WM_v2 repo root)
python tools/export_error_model_to_npz.py
       --model_dir ./models/error/error_model_Baseline_cleanedupNew_Add_ExpertExtFlow_CustomLayers_N60xL15_wVal_ep3811
       --out       ./results/export/error_turb/error_model_params.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass

import numpy as np

# BFM_WM_v2: the venv has TF 2.19 + tf_keras; select legacy Keras before TensorFlow is imported
# (TensorFlow is imported lazily below).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


_DENSE_RE = re.compile(r"^(?P<prefix>dense(?:_(?P<idx>\d+))?)/(?:kernel|bias)$")
_LAYER_WITH_WEIGHTS_RE = re.compile(
    r"^layer_with_weights-(?P<idx>\d+?)/(?P<which>kernel|bias)/\.ATTRIBUTES/VARIABLE_VALUE$"
)


@dataclass(frozen=True)
class DenseVarNames:
    layer_index: int
    kernel_name: str
    bias_name: str


def _load_norm_stats(model_dir: str) -> dict:
    path = os.path.join(model_dir, "assets", "error_norm_stats.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Normalization stats not found: {path}")

    with open(path, "r") as f:
        meta = json.load(f)

    x_mean = np.asarray(meta["x_mean"], dtype=np.float64).reshape(-1)
    x_std = np.asarray(meta["x_std"], dtype=np.float64).reshape(-1)
    y_mean = np.asarray(meta["y_mean"], dtype=np.float64).reshape(-1)
    y_std = np.asarray(meta["y_std"], dtype=np.float64).reshape(-1)

    if y_mean.size != 1 or y_std.size != 1:
        raise ValueError(
            f"Expected scalar y_mean/y_std but got shapes {y_mean.shape}, {y_std.shape}"
        )

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def _get_checkpoint_prefix(model_dir: str) -> str:
    prefix = os.path.join(model_dir, "variables", "variables")
    index_path = prefix + ".index"
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"SavedModel variables checkpoint not found: {index_path}"
        )
    return prefix


def _list_dense_vars(checkpoint_prefix: str) -> list[DenseVarNames]:
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as ex:
        raise RuntimeError(
            "TensorFlow is required to read the SavedModel checkpoint variables. "
            "Install tensorflow in the active environment."
        ) from ex

    import tensorflow as tf

    vars_list = tf.train.list_variables(checkpoint_prefix)

    kernels: dict[int, str] = {}
    biases: dict[int, str] = {}

    for name, _shape in vars_list:
        # Case 1: Keras checkpoint naming: dense/kernel, dense_1/bias, ...
        m_dense = _DENSE_RE.match(name)
        if m_dense:
            idx_str = m_dense.group("idx")
            layer_idx = int(idx_str) if idx_str is not None else 0
            if name.endswith("/kernel"):
                kernels[layer_idx] = name
            elif name.endswith("/bias"):
                biases[layer_idx] = name
            continue

        # Case 2: SavedModel variables naming: layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
        m_lww = _LAYER_WITH_WEIGHTS_RE.match(name)
        if m_lww:
            layer_idx = int(m_lww.group("idx"))
            which = m_lww.group("which")
            if which == "kernel":
                kernels[layer_idx] = name
            elif which == "bias":
                biases[layer_idx] = name
            continue

    layer_indices = sorted(set(kernels.keys()) | set(biases.keys()))
    dense_vars: list[DenseVarNames] = []

    for li in layer_indices:
        if li not in kernels or li not in biases:
            missing = []
            if li not in kernels:
                missing.append("kernel")
            if li not in biases:
                missing.append("bias")
            raise RuntimeError(
                f"Dense layer {li} is missing {', '.join(missing)} in checkpoint. "
                f"Found kernel? {li in kernels}, bias? {li in biases}"
            )
        dense_vars.append(
            DenseVarNames(layer_index=li, kernel_name=kernels[li], bias_name=biases[li])
        )

    if not dense_vars:
        raise RuntimeError(
            "No Dense kernel/bias variables found in checkpoint. "
            "This exporter expects either Keras Dense layers named 'dense', 'dense_1', ... "
            "or SavedModel variables named 'layer_with_weights-N/.../.ATTRIBUTES/VARIABLE_VALUE'."
        )

    # Sort by layer index: dense (0), dense_1 (1), ...
    dense_vars.sort(key=lambda d: d.layer_index)
    return dense_vars


def _load_dense_weights(checkpoint_prefix: str, dense_vars: list[DenseVarNames]) -> list[tuple[np.ndarray, np.ndarray]]:
    import tensorflow as tf

    weights: list[tuple[np.ndarray, np.ndarray]] = []
    for dv in dense_vars:
        W = tf.train.load_variable(checkpoint_prefix, dv.kernel_name)
        b = tf.train.load_variable(checkpoint_prefix, dv.bias_name)
        weights.append((np.asarray(W), np.asarray(b)))
    return weights


def export_npz(model_dir: str, out_path: str, dtype: str = "float32") -> None:
    norm = _load_norm_stats(model_dir)
    ckpt = _get_checkpoint_prefix(model_dir)

    dense_vars = _list_dense_vars(ckpt)
    dense_weights = _load_dense_weights(ckpt, dense_vars)

    np_dtype = np.float32 if dtype == "float32" else np.float64

    payload: dict[str, np.ndarray] = {
        "x_mean": norm["x_mean"].astype(np_dtype),
        "x_std": norm["x_std"].astype(np_dtype),
        "y_mean": norm["y_mean"].astype(np_dtype),
        "y_std": norm["y_std"].astype(np_dtype),
    }

    # Dense layers in order
    for i, (W, b) in enumerate(dense_weights):
        payload[f"W_{i}"] = np.asarray(W, dtype=np_dtype)
        payload[f"b_{i}"] = np.asarray(b, dtype=np_dtype)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **payload)

    # Summary
    print("Exported error-model parameters to:", out_path)
    print("D (input dim):", payload["x_mean"].size)
    print("Num Dense layers:", len(dense_weights))
    for i in range(len(dense_weights)):
        W = payload[f"W_{i}"]
        b = payload[f"b_{i}"]
        print(f"  Layer {i}: W {tuple(W.shape)}, b {tuple(b.shape)}")


def export_json(model_dir: str, out_path: str, dtype: str = "float32") -> None:
    """Export parameters to a single JSON file.

    Notes:
      - JSON is larger and slower to load than NPZ, but is easy to parse in many languages.
      - Arrays are written as nested Python lists.
    """
    norm = _load_norm_stats(model_dir)
    ckpt = _get_checkpoint_prefix(model_dir)

    dense_vars = _list_dense_vars(ckpt)
    dense_weights = _load_dense_weights(ckpt, dense_vars)

    np_dtype = np.float32 if dtype == "float32" else np.float64

    layers = []
    for i, (W, b) in enumerate(dense_weights):
        W = np.asarray(W, dtype=np_dtype)
        b = np.asarray(b, dtype=np_dtype)
        layers.append(
            {
                "index": int(i),
                "W_shape": [int(s) for s in W.shape],
                "b_shape": [int(s) for s in b.shape],
                "W": W.tolist(),
                "b": b.tolist(),
            }
        )

    data = {
        "format_version": 1,
        "model_dir": os.path.abspath(model_dir),
        "dtype": dtype,
        "x_mean": np.asarray(norm["x_mean"], dtype=np_dtype).tolist(),
        "x_std": np.asarray(norm["x_std"], dtype=np_dtype).tolist(),
        "y_mean": float(np.asarray(norm["y_mean"], dtype=np_dtype).reshape(-1)[0]),
        "y_std": float(np.asarray(norm["y_std"], dtype=np_dtype).reshape(-1)[0]),
        "D": int(np.asarray(norm["x_mean"]).size),
        "num_dense_layers": int(len(layers)),
        "dense_variable_names": [asdict(v) for v in dense_vars],
        "layers": layers,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f)

    print("Exported error-model parameters to:", out_path)
    print("D (input dim):", data["D"])
    print("Num Dense layers:", data["num_dense_layers"])
    for ly in layers:
        print(f"  Layer {ly['index']}: W {tuple(ly['W_shape'])}, b {tuple(ly['b_shape'])}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_dir",
        required=True,
        help="Path to TF SavedModel directory (contains saved_model.pb and variables/)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output file path (e.g. .../error_model_params.npz or .../error_model_params.json)",
    )
    p.add_argument(
        "--format",
        choices=["npz", "json"],
        default=None,
        help="Output format. If omitted, inferred from --out extension.",
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Data type used in exported arrays.",
    )
    args = p.parse_args()

    fmt = args.format
    if fmt is None:
        lower = args.out.lower()
        if lower.endswith(".json"):
            fmt = "json"
        elif lower.endswith(".npz"):
            fmt = "npz"
        else:
            raise SystemExit(
                "Could not infer --format from --out. Provide --format npz|json or use .npz/.json extension."
            )

    if fmt == "npz":
        export_npz(args.model_dir, args.out, dtype=args.dtype)
    elif fmt == "json":
        export_json(args.model_dir, args.out, dtype=args.dtype)
    else:
        raise SystemExit(f"Unsupported format: {fmt}")


if __name__ == "__main__":
    main()
