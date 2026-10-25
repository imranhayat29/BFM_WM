#!/usr/bin/env python3
# BFM_WM_v2 tools/: copied from BFM_WM/export_error_model_to_cuda_datafiles.py.
# Purpose (CharLES BFMv2 export): error_model_params.json -> CharLES BFM_wall_error_data_{turb,lam}.hpp/.cu constant arrays.
# Adaptations: none in the code (numpy only); example usage updated to tools/. CLI, defaults and output formats are unchanged.
"""Export error model parameters from error_model_params.json into .cu/.hpp constant arrays.

This script is meant to mirror the pattern of the baseline data files BFM_wall_data.{cu,hpp}:
- a header declaring extern const arrays
- a .cu defining the arrays

Input:
  error_model_params.json produced by export_error_model_to_npz.py (JSON format).

Output arrays/layout (chosen to be convenient for a C-style forward pass):
  - input_layer_err: W0 stored as (out, in) row-major, length Nneurons * Nin
  - hidden_layers_err: W1..W_{H-1} each stored as (out, in) row-major, concatenated
  - bias_err: b0..b_{H-1} concatenated, length H * Nneurons
  - output_layer_err: W_last stored as (out, in) row-major, then b_last appended

Where:
  Nin = input dimension D
  H   = number of hidden layers (Dense layers with out_dim == Nneurons)
  Nout = output dimension (should be 2 for the error model: [mu, raw_sigma_param])

Normalization stats are also emitted:
  input_mean_err[Nin], input_std_err[Nin], output_mean_err, output_std_err

Note: This script intentionally does NOT generate any CUDA kernel code.

Example usage: (run from the BFM_WM_v2 repo root)
python tools/export_error_model_to_cuda_datafiles.py
             --json    ./results/export/error_turb/error_model_params.json
             --out_hpp ./results/export/BFM_wall_error_data_turb.hpp
             --out_cu  ./results/export/BFM_wall_error_data_turb.cu --prefix wall_error
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _as_array(x: Any, dtype: np.dtype) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _format_c_array(values: np.ndarray, per_line: int = 8) -> str:
    flat = values.reshape(-1)
    parts = []
    for i, v in enumerate(flat):
        # Use repr-like compact float, but keep enough precision.
        parts.append(f"{float(v):.9g}")
    lines = []
    for i in range(0, len(parts), per_line):
        chunk = parts[i : i + per_line]
        lines.append("    " + ", ".join(chunk) + ("," if i + per_line < len(parts) else ""))
    return "\n".join(lines)


def _transpose_to_out_in(W_in_out: np.ndarray) -> np.ndarray:
    # Keras Dense uses W shape (in, out). We store as (out, in) to match common
    # C-style indexing tmp[i] += W[i*in + j] * x[j].
    return np.ascontiguousarray(W_in_out.T)


def export_to_cu_hpp(
    json_path: str,
    out_hpp: str,
    out_cu: str,
    symbol_prefix: str = "wall",
    cti_type: str = "cti_ffp",
) -> None:
    data = _load_json(json_path)

    dtype = np.float32 if data.get("dtype", "float32") == "float32" else np.float64

    x_mean = _as_array(data["x_mean"], dtype).reshape(-1)
    x_std = _as_array(data["x_std"], dtype).reshape(-1)
    y_mean = float(data["y_mean"]) if not isinstance(data["y_mean"], list) else float(data["y_mean"][0])
    y_std = float(data["y_std"]) if not isinstance(data["y_std"], list) else float(data["y_std"][0])

    layers = data["layers"]
    if not layers:
        raise ValueError("No layers found in JSON")

    # Reconstruct W/b arrays
    Ws = []
    bs = []
    for ly in layers:
        W = _as_array(ly["W"], dtype)
        b = _as_array(ly["b"], dtype).reshape(-1)
        Ws.append(W)
        bs.append(b)

    Nin = int(x_mean.size)

    # Infer hidden width and count
    # Hidden layers are those whose out_dim is constant and != final out_dim.
    out_dims = [int(W.shape[1]) for W in Ws]  # W shape (in,out)
    if len(out_dims) < 2:
        raise ValueError(f"Expected at least 2 Dense layers, got {len(out_dims)}")

    Nout = int(out_dims[-1])
    if Nout != 2:
        # Keep generic, but most of your error models should be 2.
        pass

    # Hidden layers: all but last
    hidden_out_dims = out_dims[:-1]
    if len(set(hidden_out_dims)) != 1:
        raise ValueError(f"Hidden layers are not uniform: {hidden_out_dims}")
    Nneurons = int(hidden_out_dims[0])
    H = len(hidden_out_dims)

    # Sanity: first layer input dim should match Nin
    if int(Ws[0].shape[0]) != Nin:
        raise ValueError(f"Input dim mismatch: JSON x_mean has Nin={Nin} but W0 has in_dim={Ws[0].shape[0]}")

    # Build flattened arrays in (out,in) layout.
    W0_out_in = _transpose_to_out_in(Ws[0])  # (Nneurons, Nin)
    input_layer = W0_out_in.reshape(-1)

    # Hidden transitions: W1..W_{H-1} (there are H-1 of these), each (Nneurons, Nneurons) in out,in layout.
    hidden_mats = []
    for i in range(1, H):
        Wi_out_in = _transpose_to_out_in(Ws[i])
        if Wi_out_in.shape != (Nneurons, Nneurons):
            raise ValueError(f"Expected hidden W{i} to be {(Nneurons, Nneurons)} in out,in layout, got {Wi_out_in.shape}")
        hidden_mats.append(Wi_out_in.reshape(-1))
    hidden_layers = np.concatenate(hidden_mats, axis=0) if hidden_mats else np.zeros((0,), dtype=dtype)

    # Hidden biases b0..b_{H-1} (each length Nneurons)
    hidden_biases = []
    for i in range(0, H):
        bi = bs[i]
        if bi.shape[0] != Nneurons:
            raise ValueError(f"Expected bias b{i} length {Nneurons}, got {bi.shape}")
        hidden_biases.append(bi)
    bias_all = np.concatenate(hidden_biases, axis=0)

    # Output layer: W_last in (out,in) plus b_last appended
    W_last_out_in = _transpose_to_out_in(Ws[-1])  # (Nout, Nneurons)
    if W_last_out_in.shape[1] != Nneurons:
        raise ValueError(f"Expected output W to have in_dim={Nneurons}, got {W_last_out_in.shape}")
    b_last = bs[-1]
    if b_last.shape[0] != Nout:
        raise ValueError(f"Expected output bias length {Nout}, got {b_last.shape}")

    output_layer = np.concatenate([W_last_out_in.reshape(-1), b_last.reshape(-1)], axis=0)

    # Sizes
    input_len = int(input_layer.size)
    hidden_len = int(hidden_layers.size)
    bias_len = int(bias_all.size)
    output_len = int(output_layer.size)

    # Names
    pre = symbol_prefix
    # Match the existing wall-model naming pattern where possible.
    name_Nin = f"Nin_{pre}"
    name_Nout = f"Nout_{pre}"
    name_Nlayers = f"Nlayers_{pre}"
    name_Nneur = f"Nneurons_{pre}"
    name_act = f"act_fn_{pre}"

    # Normalization stats (error model uses standardization, not gain/offset mapping).
    name_in_mean = f"input_mean_{pre}"
    name_in_std = f"input_std_{pre}"
    name_out_mean = f"output_mean_{pre}"
    name_out_std = f"output_std_{pre}"

    # Network parameter blobs
    name_input = f"input_layer_{pre}"
    name_hidden = f"hidden_layers_{pre}"
    name_bias = f"bias_{pre}"
    name_output = f"output_layer_{pre}"

    # Header content
    hpp = []
    hpp.append('#include <cti_utils_gpu.hpp>')
    hpp.append('')
    # Nlayers follows the convention in BFM_wall_data.*: input layer + hidden layers + output layer.
    # Here: 1 (input->h1) + (H-1) hidden transitions + 1 output = H+1.
    Nlayers = H + 1
    hpp.append(f"extern const int {name_Nin};")
    hpp.append(f"extern const int {name_Nout};")
    hpp.append(f"extern const int {name_Nlayers};")
    hpp.append(f"extern const int {name_Nneur};")
    hpp.append(f"extern const int {name_act};")
    hpp.append('')
    hpp.append(f"extern const {cti_type} {name_in_mean}[{Nin}];")
    hpp.append(f"extern const {cti_type} {name_in_std}[{Nin}];")
    hpp.append(f"extern const {cti_type} {name_out_mean};")
    hpp.append(f"extern const {cti_type} {name_out_std};")
    hpp.append('')
    hpp.append(f"extern const {cti_type} {name_input}[{input_len}];")
    hpp.append(f"extern const {cti_type} {name_hidden}[{hidden_len}];")
    hpp.append(f"extern const {cti_type} {name_bias}[{bias_len}];")
    hpp.append(f"extern const {cti_type} {name_output}[{output_len}];")
    hpp.append('')

    # CU content
    cu = []
    cu.append(f'#include "{os.path.basename(out_hpp)}"')
    cu.append('')
    cu.append(f"const int {name_Nin} = {Nin};")
    cu.append(f"const int {name_Nout} = {Nout};")
    cu.append(f"const int {name_Nlayers} = {Nlayers};")
    cu.append(f"const int {name_Nneur} = {Nneurons};")
    # Activation flag: 0 = ReLU, which matches the error model hidden activations.
    cu.append(f"const int {name_act} = 0;")
    cu.append('')

    cu.append(f"const {cti_type} {name_in_mean}[{Nin}]{{")
    cu.append(_format_c_array(x_mean, per_line=8))
    cu.append("};")
    cu.append(f"const {cti_type} {name_in_std}[{Nin}]{{")
    cu.append(_format_c_array(x_std, per_line=8))
    cu.append("};")
    cu.append(f"const {cti_type} {name_out_mean} = {y_mean:.9g};")
    cu.append(f"const {cti_type} {name_out_std} = {y_std:.9g};")
    cu.append('')

    cu.append(f"const {cti_type} {name_input}[{input_len}]{{")
    cu.append(_format_c_array(input_layer, per_line=8))
    cu.append("};")
    cu.append(f"const {cti_type} {name_hidden}[{hidden_len}]{{")
    if hidden_len:
        cu.append(_format_c_array(hidden_layers, per_line=8))
    cu.append("};")
    cu.append(f"const {cti_type} {name_bias}[{bias_len}]{{")
    cu.append(_format_c_array(bias_all, per_line=8))
    cu.append("};")
    cu.append(f"const {cti_type} {name_output}[{output_len}]{{")
    cu.append(_format_c_array(output_layer, per_line=8))
    cu.append("};")

    os.makedirs(os.path.dirname(out_hpp) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_cu) or ".", exist_ok=True)

    with open(out_hpp, "w") as f:
        f.write("\n".join(hpp))
    with open(out_cu, "w") as f:
        f.write("\n".join(cu))

    print("Wrote:", out_hpp)
    print("Wrote:", out_cu)
    print(f"Nin={Nin}, Nneurons={Nneurons}, Nhidden={H}, Nout={Nout}")
    print(f"input_layer len={input_len}, hidden_layers len={hidden_len}, bias len={bias_len}, output_layer len={output_len}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="Path to error_model_params.json")
    p.add_argument("--out_hpp", required=True, help="Output .hpp path")
    p.add_argument("--out_cu", required=True, help="Output .cu path")
    p.add_argument("--prefix", default="err", help="Symbol suffix/prefix to avoid clashes (default: err)")
    p.add_argument("--cti_type", default="cti_ffp", help="C++ floating type name used in the header (default: cti_ffp)")
    args = p.parse_args()

    export_to_cu_hpp(args.json, args.out_hpp, args.out_cu, symbol_prefix=args.prefix, cti_type=args.cti_type)


if __name__ == "__main__":
    main()
