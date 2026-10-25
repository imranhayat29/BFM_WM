#!/usr/bin/env python3
# BFM_WM_v2 tools/: copied from BFM_WM/export_baseline_to_cude_datafiles.py (file-name typo 'cude' fixed).
# Purpose (CharLES BFMv2 export): compact baseline WallModel .pth -> CharLES BFM_wall_data_{turb,lam}.hpp/.cu constant arrays.
# Adaptations: repo root inserted into sys.path so that 'convert_to_cuda' and 'src.wall_model' import when run as tools/<script>; the parent directories of the outputs are created; example usage updated to tools/. CLI, defaults and output formats are unchanged.
"""Export a baseline wall-model checkpoint directly into .cu/.hpp constant arrays.

This is a baseline-specific wrapper around the generic PyTorch MLP exporter.
It loads a compact WallModel checkpoint (`.pth`), reconstructs the model with
`WallModel.load_compact`, and writes CUDA/C++ parameter blobs in the same style
as the existing baseline wall-model data files.

The compact checkpoint stores the network architecture and weights, but it does
not store separate input/output normalization statistics. This exporter
therefore defaults to identity scaling:
  - input_gains_wall  = 1 for each model input
  - input_offsets_wall = 0 for each model input
  - output_gain_wall = 1
  - output_offset_wall = 0

If your downstream inference path expects non-identity scaling, provide the
override values explicitly on the command line.

For the requested laminar baseline checkpoint in this workspace, the saved model
config resolves to:
  - InputScaling = 2
  - HiddenLayers = [40, 40, 40, 40]
  - OutputDim = 1

Example usage: (run from the BFM_WM_v2 repo root)
python tools/export_baseline_to_cuda_datafiles.py
       --checkpoint ./models/NN_wm_CH0_G0_S0_tn560665_vn140167_fds0_lds0_customw1_inputs2_final_ep30000_tl0.00032144_vl0.00031670.pth
       --out_hpp    ./results/export/BFM_wall_data_lam.hpp
       --out_cu     ./results/export/BFM_wall_data_lam.cu
"""

from __future__ import annotations

import argparse
import os
import sys

# BFM_WM_v2: running 'python tools/<script>.py' puts tools/ (not the repo root) on sys.path; add the repo
# root so that the top-level convert_to_cuda.py and the src/ package (src.wall_model) are importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from convert_to_cuda import (  # noqa: E402
    extract_linear_layers,
    infer_model_structure,
    load_model_from_checkpoint,
    write_mlp_params_split,
)


def _baseline_model_loader(checkpoint_path: str):
    from src.wall_model import WallModel

    wall_model = WallModel.load_compact(checkpoint_path, device="cpu")
    return wall_model.model


def _infer_structure(checkpoint_path: str) -> dict:
    model = load_model_from_checkpoint(checkpoint_path, _baseline_model_loader)
    linear_layers = extract_linear_layers(model)
    return infer_model_structure(linear_layers)


def export_baseline_to_cu_hpp(
    checkpoint_path: str,
    out_hpp: str,
    out_cu: str,
    input_gains: list[float] | None = None,
    input_offsets: list[float] | None = None,
    output_gain: float = 1.0,
    output_offset: float = 0.0,
    cti_type: str = "cti_ffp",
    act_fn: int = 0,
    zero_threshold: float = 1e-10,
) -> dict:
    structure = _infer_structure(checkpoint_path)
    n_inputs = structure["n_inputs"]

    resolved_input_gains = input_gains if input_gains is not None else [1.0] * n_inputs
    resolved_input_offsets = (
        input_offsets if input_offsets is not None else [0.0] * n_inputs
    )

    if len(resolved_input_gains) != n_inputs:
        raise ValueError(
            f"Expected {n_inputs} input gains for checkpoint {checkpoint_path}, "
            f"got {len(resolved_input_gains)}"
        )
    if len(resolved_input_offsets) != n_inputs:
        raise ValueError(
            f"Expected {n_inputs} input offsets for checkpoint {checkpoint_path}, "
            f"got {len(resolved_input_offsets)}"
        )

    # BFM_WM_v2: create the output directories, as the other exporters do (write_mlp_params_split does not)
    os.makedirs(os.path.dirname(out_hpp) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_cu) or ".", exist_ok=True)

    return write_mlp_params_split(
        checkpoint_path=checkpoint_path,
        header_filename=out_hpp,
        source_filename=out_cu,
        input_gains=resolved_input_gains,
        input_offsets=resolved_input_offsets,
        output_gain=output_gain,
        output_offset=output_offset,
        dtype_str=cti_type,
        model_loader=_baseline_model_loader,
        act_fn_id=act_fn,
        zero_threshold=zero_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a compact baseline WallModel checkpoint directly into "
            "CUDA/C++ data files."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the compact baseline checkpoint (.pth)",
    )
    parser.add_argument(
        "--out_hpp",
        required=True,
        help="Output header path (.hpp)",
    )
    parser.add_argument(
        "--out_cu",
        required=True,
        help="Output CUDA/source path (.cu)",
    )
    parser.add_argument(
        "--input_gains",
        nargs="+",
        type=float,
        default=None,
        help="Optional input gains. Defaults to identity scaling for each input.",
    )
    parser.add_argument(
        "--input_offsets",
        nargs="+",
        type=float,
        default=None,
        help="Optional input offsets. Defaults to zero for each input.",
    )
    parser.add_argument(
        "--output_gain",
        type=float,
        default=1.0,
        help="Optional output gain. Defaults to identity scaling.",
    )
    parser.add_argument(
        "--output_offset",
        type=float,
        default=0.0,
        help="Optional output offset. Defaults to zero.",
    )
    parser.add_argument(
        "--cti_type",
        default="cti_ffp",
        help="C++ floating-point type name used in generated files.",
    )
    parser.add_argument(
        "--act_fn",
        type=int,
        default=0,
        help="Activation function ID used by the consumer side (default: 0 for ReLU).",
    )
    parser.add_argument(
        "--zero_threshold",
        type=float,
        default=1e-10,
        help="Set very small weights/biases to zero before export.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    structure = export_baseline_to_cu_hpp(
        checkpoint_path=args.checkpoint,
        out_hpp=args.out_hpp,
        out_cu=args.out_cu,
        input_gains=args.input_gains,
        input_offsets=args.input_offsets,
        output_gain=args.output_gain,
        output_offset=args.output_offset,
        cti_type=args.cti_type,
        act_fn=args.act_fn,
        zero_threshold=args.zero_threshold,
    )

    print("Exported baseline model parameters to:")
    print("  Header:", args.out_hpp)
    print("  Source:", args.out_cu)
    print("  Structure:", structure)


if __name__ == "__main__":
    main()