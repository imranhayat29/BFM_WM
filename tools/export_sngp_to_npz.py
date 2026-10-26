#!/usr/bin/env python3
# BFM_WM_v2 tools/: copied from BFM_WM/export_sngp_to_npz.py.
# Purpose (CharLES BFMv2 export): SNGP SavedModel -> sngp_params.npz (+ sngp_params_meta.json), input of export_sngp_to_cuda_datafiles.py.
# Adaptations: sets TF_USE_LEGACY_KERAS=1 (setdefault) before TensorFlow can be imported; example usage updated to tools/. CLI, defaults and output formats are unchanged.
"""Export an SNGP TF SavedModel into a compact .npz (+ optional small .json) for CUDA/C++ inference.

Why this exists:
- This workspace uses Keras 3, which cannot `tf.keras.models.load_model()` legacy SavedModels.
- `tf.saved_model.load()` *can* load the SavedModel and expose its variables.

This exporter reads:
- <model_dir>/assets/metadata.json  (mean/std and class counts)
- SavedModel variables via tf.saved_model.load(model_dir)

It writes an NPZ containing arrays needed to reproduce the notebook's inference:
- z-score normalization: mean_X_train, std_X_train
- input Dense: W_in, b_in
- residual Dense layers (SpectralNormalization wrappers): W_res_i, b_res_i
  NOTE: the notebook calls the model with `training=False`, which disables
  spectral-normalization power-iteration and uses the stored kernels directly.
- GP head (RandomFeatureGaussianProcess):
  - random feature Dense: W_rf, b_rf
  - output weights: W_out (no bias), b_out
  - (optional) gp_precision_matrix

The output is meant as a parameter bundle; it does NOT generate CUDA kernels.

Example usage: (run from the BFM_WM_v2 repo root)
python tools/export_sngp_to_npz.py 
        --model_dir ./models/sngp/sngp_model_Baseline_cleanedupNew_Add_ExpertExtFlow_ep4000_merged 
        --out_npz ./results/export/sngp_turb/sngp_params.npz 
        --also_write_json ./results/export/sngp_turb/sngp_params_meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Iterable

import numpy as np

# BFM_WM_v2: the venv has TF 2.19 + tf_keras; select legacy Keras before TensorFlow is imported
# (TensorFlow is imported lazily below).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def _get_model_dir_assets_metadata(model_dir: str) -> dict[str, Any]:
    meta_path = os.path.join(model_dir, "assets", "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def _flatten_1xD(x: Any, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    # metadata stores [[...]]
    return arr.reshape(-1)


def _tf_load_savedmodel_variables(model_dir: str):
    try:
        import tensorflow as tf  # noqa: F401
    except Exception as ex:
        raise RuntimeError(
            "TensorFlow is required to read the SavedModel variables. "
            "Install tensorflow in the active environment."
        ) from ex

    import tensorflow as tf

    obj = tf.saved_model.load(model_dir)
    return obj, list(obj.variables)


def _find_unique_var(
    variables: Iterable[Any],
    name: str,
    *,
    shape: tuple[int, ...] | None = None,
):
    matches = [v for v in variables if v.name == name]
    if shape is not None:
        matches = [v for v in matches if tuple(v.shape.as_list()) == tuple(shape)]

    if len(matches) != 1:
        shapes = [(v.name, tuple(v.shape.as_list())) for v in matches]
        raise ValueError(
            f"Expected exactly 1 variable for name={name!r} shape={shape}, got {len(matches)}. "
            f"Matches: {shapes}"
        )
    return matches[0]


def export_sngp_to_npz(
    model_dir: str,
    out_npz: str,
    *,
    dtype: str = "float32",
    include_gp_precision: bool = False,
    also_write_json: str | None = None,
) -> None:
    dtype_np = np.float32 if dtype == "float32" else np.float64

    meta = _get_model_dir_assets_metadata(model_dir)
    mean_X = _flatten_1xD(meta["mean_X_train"], dtype_np)
    std_X = _flatten_1xD(meta["std_X_train"], dtype_np)

    num_classes = int(meta["num_classes"])
    num_classes_train = int(meta["num_classes_train"])

    obj, variables = _tf_load_savedmodel_variables(model_dir)

    # --- Input Dense ---
    W_in = _find_unique_var(variables, "input_layer/kernel:0").numpy().astype(dtype_np)
    b_in = _find_unique_var(variables, "input_layer/bias:0").numpy().astype(dtype_np)

    Nin, Nhidden = map(int, W_in.shape)

    # --- Residual Dense layers (spectral_normalization wrappers) ---
    # Infer how many residual layers exist by scanning kernel names.
    res_layers = []
    i = 0
    while True:
        base = "spectral_normalization" if i == 0 else f"spectral_normalization_{i}"
        k_name = f"{base}/kernel:0"
        b_name = f"{base}/bias:0"
        try:
            W = _find_unique_var(variables, k_name).numpy().astype(dtype_np)
            b = _find_unique_var(variables, b_name).numpy().astype(dtype_np)
        except ValueError:
            break
        res_layers.append((W, b))
        i += 1

    if not res_layers:
        raise ValueError("No residual spectral_normalization layers found")

    # Sanity
    for li, (W, b) in enumerate(res_layers):
        if W.shape != (Nhidden, Nhidden):
            raise ValueError(f"Residual layer {li} kernel has shape {W.shape}, expected {(Nhidden, Nhidden)}")
        if b.shape != (Nhidden,):
            raise ValueError(f"Residual layer {li} bias has shape {b.shape}, expected {(Nhidden,)}")

    Nres = len(res_layers)

    # --- GP head variables (RandomFeatureGaussianProcess) ---
    # There are two variables named 'random_feature_gaussian_process/kernel:0' in this SavedModel.
    # Disambiguate by shape:
    #   (Nhidden, Ninducing) -> random feature Dense kernel
    #   (Ninducing, C)       -> output weights
    # Bias: (Ninducing,)
    # gp_output_bias: (C,)
    W_rf = _find_unique_var(
        variables,
        "random_feature_gaussian_process/kernel:0",
        shape=(Nhidden, 1024),
    ).numpy().astype(dtype_np)
    b_rf = _find_unique_var(variables, "random_feature_gaussian_process/bias:0").numpy().astype(dtype_np)

    Ninducing = int(W_rf.shape[1])
    if b_rf.shape != (Ninducing,):
        raise ValueError(f"Random feature bias shape {b_rf.shape} != ({Ninducing},)")

    W_out = _find_unique_var(
        variables,
        "random_feature_gaussian_process/kernel:0",
        shape=(Ninducing, num_classes),
    ).numpy().astype(dtype_np)
    b_out = _find_unique_var(
        variables,
        "random_feature_gaussian_process/gp_output_bias:0",
    ).numpy().astype(dtype_np)

    if b_out.shape != (num_classes,):
        raise ValueError(f"Output bias shape {b_out.shape} != ({num_classes},)")

    gp_precision = None
    if include_gp_precision:
        gp_precision = _find_unique_var(
            variables,
            "random_feature_gaussian_process/gp_precision_matrix:0",
        ).numpy().astype(dtype_np)

    # Mean-field settings used by notebook (compute_posterior_mean_probability)
    lambda_param = float(math.pi / 8.0)
    scale_cov_none = float(math.sqrt(1.0 + lambda_param))

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)

    npz_dict: dict[str, Any] = {
        "dtype": np.array(dtype, dtype=object),
        "Nin": np.array(Nin, dtype=np.int32),
        "Nhidden": np.array(Nhidden, dtype=np.int32),
        "Nres": np.array(Nres, dtype=np.int32),
        "Ninducing": np.array(Ninducing, dtype=np.int32),
        "num_classes": np.array(num_classes, dtype=np.int32),
        "num_classes_train": np.array(num_classes_train, dtype=np.int32),
        "mean_X_train": mean_X,
        "std_X_train": std_X,
        "W_in": W_in,
        "b_in": b_in,
        "W_rf": W_rf,
        "b_rf": b_rf,
        "W_out": W_out,
        "b_out": b_out,
        "lambda_param": np.array(lambda_param, dtype=dtype_np),
        "scale_cov_none": np.array(scale_cov_none, dtype=dtype_np),
    }

    for li, (W, b) in enumerate(res_layers):
        npz_dict[f"W_res_{li}"] = W
        npz_dict[f"b_res_{li}"] = b

    if gp_precision is not None:
        npz_dict["gp_precision_matrix"] = gp_precision

    np.savez(out_npz, **npz_dict)

    if also_write_json is not None:
        out = {
            "dtype": dtype,
            "arch": {
                "Nin": Nin,
                "Nhidden": Nhidden,
                "Nres": Nres,
                "Ninducing": Ninducing,
                "num_classes": num_classes,
                "num_classes_train": num_classes_train,
            },
            "mean_field": {
                "lambda_param": lambda_param,
                "scale_if_cov_none": scale_cov_none,
            },
            "mean_X_train": mean_X.tolist(),
            "std_X_train": std_X.tolist(),
            # Keep the full original metadata too (mappings, flags).
            "metadata": meta,
            "notes": {
                "keras3": "This SavedModel cannot be loaded via tf.keras.models.load_model in Keras 3; exporter uses tf.saved_model.load.",
                "spectral_norm": "Notebook calls model(..., training=False); SpectralNormalization wrapper does not power-iterate in that case. Kernels exported are the stored kernels.",
                "gp_precision_matrix": "Omitted unless include_gp_precision=True; it is large and not needed for logits-only inference.",
            },
        }
        os.makedirs(os.path.dirname(also_write_json) or ".", exist_ok=True)
        with open(also_write_json, "w") as f:
            json.dump(out, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Path to SNGP SavedModel directory")
    p.add_argument(
        "--out_npz",
        required=True,
        help="Output NPZ path (e.g. <model_dir>/assets/sngp_params.npz)",
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Numeric dtype to write",
    )
    p.add_argument(
        "--include_gp_precision",
        action="store_true",
        help="Include gp_precision_matrix (1024x1024) in NPZ",
    )
    p.add_argument(
        "--also_write_json",
        default=None,
        help="Optional small JSON sidecar (metadata + shapes; no big weights)",
    )

    args = p.parse_args()

    export_sngp_to_npz(
        args.model_dir,
        args.out_npz,
        dtype=args.dtype,
        include_gp_precision=bool(args.include_gp_precision),
        also_write_json=args.also_write_json,
    )


if __name__ == "__main__":
    main()
