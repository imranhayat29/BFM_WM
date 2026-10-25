#!/usr/bin/env python3
# BFM_WM_v2 tools/: copied from BFM_WM/export_sngp_to_cuda_datafiles.py.
# Purpose (CharLES BFMv2 export): sngp_params.npz -> CharLES BFM_sngp_data_{turb,lam}.hpp/.cu constant arrays.
# Adaptations: none in the code (numpy only); example usage updated to tools/. CLI, defaults and output formats are unchanged.
"""Export SNGP parameters from an .npz bundle into .cu/.hpp constant arrays.

Input:
  sngp_params.npz produced by export_sngp_to_npz.py

Output:
  - a header (.hpp) declaring extern const arrays
  - a CUDA TU (.cu) defining those arrays

This mirrors the "data blob" pattern used elsewhere in this workspace.
It intentionally does NOT generate any CUDA kernel code.

Weight layout:
- Dense kernels in NPZ are stored in Keras layout (in, out).
- For C-style inference, we store kernels in (out, in) row-major so that:
    y_i = sum_j W_out_in[i*in + j] * x_j + b_i

SNGP forward pass (logits-only, matching SavedModel serving_default):
- Normalize input: x_norm = (x - mean) / std
- Input dense: h = x_norm @ W_in + b_in
- Residual blocks (Nres): h = h + relu(h @ W_res_i + b_res_i)
- Random features: z = cos(h @ W_rf + b_rf)
- Feature scaling: z *= sqrt(2 / Ninducing)
- Output: logits = z @ W_out + b_out

Mean-field / confidence computation is NOT emitted here; it is pure data export.

Example usage: (run from the BFM_WM_v2 repo root)
python tools/export_sngp_to_cuda_datafiles.py 
       --npz     ./results/export/sngp_turb/sngp_params.npz 
       --out_hpp ./results/export/BFM_sngp_data_turb.hpp 
       --out_cu  ./results/export/BFM_sngp_data_turb.cu --suffix sngp
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np


def _load_npz(path: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _as_float_array(x: Any, dtype: np.dtype) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def _transpose_to_out_in(W_in_out: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(W_in_out.T)


def _format_c_array(values: np.ndarray, per_line: int = 8) -> str:
    flat = values.reshape(-1)
    parts = [f"{float(v):.9g}" for v in flat]
    lines = []
    for i in range(0, len(parts), per_line):
        chunk = parts[i : i + per_line]
        lines.append("    " + ", ".join(chunk) + ("," if i + per_line < len(parts) else ""))
    return "\n".join(lines)


def export_to_cu_hpp(npz_path: str, out_hpp: str, out_cu: str, symbol_suffix: str = "sngp") -> None:
    blob = _load_npz(npz_path)

    dtype_str = str(blob.get("dtype", "float32"))
    dtype = np.float32 if "float32" in dtype_str else np.float64

    Nin = int(blob["Nin"])
    Nhidden = int(blob["Nhidden"])
    Nres = int(blob["Nres"])
    Ninducing = int(blob["Ninducing"])
    C = int(blob["num_classes"])
    Ctrain = int(blob["num_classes_train"])

    lambda_param = float(np.asarray(blob.get("lambda_param", np.pi / 8.0), dtype=np.float64))
    scale_cov_none = float(np.asarray(blob.get("scale_cov_none", np.sqrt(1.0 + np.pi / 8.0)), dtype=np.float64))

    mean_X = _as_float_array(blob["mean_X_train"], dtype).reshape(-1)
    std_X = _as_float_array(blob["std_X_train"], dtype).reshape(-1)

    if mean_X.size != Nin or std_X.size != Nin:
        raise ValueError(f"mean/std dims mismatch: mean={mean_X.size}, std={std_X.size}, Nin={Nin}")

    W_in = _as_float_array(blob["W_in"], dtype)
    b_in = _as_float_array(blob["b_in"], dtype).reshape(-1)
    if W_in.shape != (Nin, Nhidden) or b_in.shape != (Nhidden,):
        raise ValueError(f"Input layer shapes mismatch: W_in={W_in.shape}, b_in={b_in.shape}")

    W_res = []
    b_res = []
    for i in range(Nres):
        W = _as_float_array(blob[f"W_res_{i}"], dtype)
        b = _as_float_array(blob[f"b_res_{i}"], dtype).reshape(-1)
        if W.shape != (Nhidden, Nhidden) or b.shape != (Nhidden,):
            raise ValueError(f"Residual layer {i} shapes mismatch: W={W.shape}, b={b.shape}")
        W_res.append(W)
        b_res.append(b)

    W_rf = _as_float_array(blob["W_rf"], dtype)
    b_rf = _as_float_array(blob["b_rf"], dtype).reshape(-1)
    if W_rf.shape != (Nhidden, Ninducing) or b_rf.shape != (Ninducing,):
        raise ValueError(f"Random feature shapes mismatch: W_rf={W_rf.shape}, b_rf={b_rf.shape}")

    W_out = _as_float_array(blob["W_out"], dtype)
    b_out = _as_float_array(blob["b_out"], dtype).reshape(-1)
    if W_out.shape != (Ninducing, C) or b_out.shape != (C,):
        raise ValueError(f"Output shapes mismatch: W_out={W_out.shape}, b_out={b_out.shape}")

    # Flatten kernels in out-in layout.
    W_in_oi = _transpose_to_out_in(W_in)  # (Nhidden, Nin)
    W_res_oi = [_transpose_to_out_in(W) for W in W_res]  # each (Nhidden, Nhidden)
    W_rf_oi = _transpose_to_out_in(W_rf)  # (Ninducing, Nhidden)
    W_out_oi = _transpose_to_out_in(W_out)  # (C, Ninducing)

    # Concatenate residual weights/biases for compactness.
    W_res_all = np.concatenate([w.reshape(-1) for w in W_res_oi], axis=0)
    b_res_all = np.concatenate([b.reshape(-1) for b in b_res], axis=0)

    # Build files.
    guard = f"BFM_{symbol_suffix.upper()}_DATA_HPP"

    hpp = []
    hpp.append(f"#ifndef {guard}")
    hpp.append(f"#define {guard}")
    hpp.append("")
    hpp.append("#ifdef __cplusplus")
    hpp.append("extern \"C\" {")
    hpp.append("#endif")
    hpp.append("")

    # Scalars
    hpp.append(f"extern const int Nin_{symbol_suffix};")
    hpp.append(f"extern const int Nhidden_{symbol_suffix};")
    hpp.append(f"extern const int Nres_{symbol_suffix};")
    hpp.append(f"extern const int Ninducing_{symbol_suffix};")
    hpp.append(f"extern const int Nclasses_{symbol_suffix};")
    hpp.append(f"extern const int Nclasses_train_{symbol_suffix};")

    hpp.append(f"extern const float lambda_param_{symbol_suffix};")
    hpp.append(f"extern const float scale_cov_none_{symbol_suffix};")

    # Normalization
    hpp.append(f"extern const float input_mean_{symbol_suffix}[{Nin}];")
    hpp.append(f"extern const float input_std_{symbol_suffix}[{Nin}];")

    # Weights/biases
    hpp.append(f"extern const float W_in_{symbol_suffix}[{W_in_oi.size}];")
    hpp.append(f"extern const float b_in_{symbol_suffix}[{b_in.size}];")

    hpp.append(f"extern const float W_res_{symbol_suffix}[{W_res_all.size}];")
    hpp.append(f"extern const float b_res_{symbol_suffix}[{b_res_all.size}];")

    hpp.append(f"extern const float W_rf_{symbol_suffix}[{W_rf_oi.size}];")
    hpp.append(f"extern const float b_rf_{symbol_suffix}[{b_rf.size}];")

    hpp.append(f"extern const float W_out_{symbol_suffix}[{W_out_oi.size}];")
    hpp.append(f"extern const float b_out_{symbol_suffix}[{b_out.size}];")

    hpp.append("")
    hpp.append("#ifdef __cplusplus")
    hpp.append("}")
    hpp.append("#endif")
    hpp.append("")
    hpp.append(f"#endif // {guard}")

    cu = []
    cu.append(f'#include "{os.path.basename(out_hpp)}"')
    cu.append("")

    cu.append(f"const int Nin_{symbol_suffix} = {Nin};")
    cu.append(f"const int Nhidden_{symbol_suffix} = {Nhidden};")
    cu.append(f"const int Nres_{symbol_suffix} = {Nres};")
    cu.append(f"const int Ninducing_{symbol_suffix} = {Ninducing};")
    cu.append(f"const int Nclasses_{symbol_suffix} = {C};")
    cu.append(f"const int Nclasses_train_{symbol_suffix} = {Ctrain};")

    cu.append(f"const float lambda_param_{symbol_suffix} = {lambda_param:.9g}f;")
    cu.append(f"const float scale_cov_none_{symbol_suffix} = {scale_cov_none:.9g}f;")
    cu.append("")

    cu.append(f"const float input_mean_{symbol_suffix}[{Nin}] = {{\n{_format_c_array(mean_X)}\n}};")
    cu.append(f"const float input_std_{symbol_suffix}[{Nin}] = {{\n{_format_c_array(std_X)}\n}};")
    cu.append("")

    cu.append(f"const float W_in_{symbol_suffix}[{W_in_oi.size}] = {{\n{_format_c_array(W_in_oi)}\n}};")
    cu.append(f"const float b_in_{symbol_suffix}[{b_in.size}] = {{\n{_format_c_array(b_in)}\n}};")
    cu.append("")

    cu.append(f"const float W_res_{symbol_suffix}[{W_res_all.size}] = {{\n{_format_c_array(W_res_all)}\n}};")
    cu.append(f"const float b_res_{symbol_suffix}[{b_res_all.size}] = {{\n{_format_c_array(b_res_all)}\n}};")
    cu.append("")

    cu.append(f"const float W_rf_{symbol_suffix}[{W_rf_oi.size}] = {{\n{_format_c_array(W_rf_oi)}\n}};")
    cu.append(f"const float b_rf_{symbol_suffix}[{b_rf.size}] = {{\n{_format_c_array(b_rf)}\n}};")
    cu.append("")

    cu.append(f"const float W_out_{symbol_suffix}[{W_out_oi.size}] = {{\n{_format_c_array(W_out_oi)}\n}};")
    cu.append(f"const float b_out_{symbol_suffix}[{b_out.size}] = {{\n{_format_c_array(b_out)}\n}};")

    os.makedirs(os.path.dirname(out_hpp) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_cu) or ".", exist_ok=True)

    with open(out_hpp, "w") as f:
        f.write("\n".join(hpp) + "\n")

    with open(out_cu, "w") as f:
        f.write("\n".join(cu) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="Path to sngp_params.npz")
    p.add_argument("--out_hpp", required=True, help="Output header path")
    p.add_argument("--out_cu", required=True, help="Output .cu path")
    p.add_argument(
        "--suffix",
        default="sngp",
        help="Suffix used in symbol names (default: sngp)",
    )
    args = p.parse_args()

    export_to_cu_hpp(args.npz, args.out_hpp, args.out_cu, symbol_suffix=args.suffix)


if __name__ == "__main__":
    main()
