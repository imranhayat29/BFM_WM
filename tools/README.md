# tools/: CharLES export scripts

These scripts export the models of this repository to the CharLES BFMv2 solver:

| Script | Purpose |
| --- | --- |
| `export_sngp_to_npz.py` | SNGP SavedModel -> parameter bundle `sngp_params.npz` |
| `export_sngp_to_cuda_datafiles.py` | `sngp_params.npz` -> CharLES data files `.hpp/.cu` |
| `export_error_model_to_npz.py` | error-model SavedModel -> parameter bundle `error_model_params.json` (or `.npz`) |
| `export_error_model_to_cuda_datafiles.py` | `error_model_params.json` -> CharLES data files `.hpp/.cu` |
| `export_baseline_to_cuda_datafiles.py` | baseline checkpoint `.pth` -> CharLES data files `.hpp/.cu` |

Run every script from the repository root (`python tools/<script>.py ...`) with the project venv active.
`--help` lists the options of each script.

## Exporting models to CharLES BFMv2

CharLES BFMv2 compiles the wall-model networks in as constant arrays, in the files
`src/charles/BFM_*_data_*.{hpp,cu}` of its source tree (not part of this repository). The export scripts write these files from the models of
this repository:

| Model | Step 1 | Step 2 | CharLES files |
| --- | --- | --- | --- |
| SNGP confidence (TensorFlow) | `export_sngp_to_npz.py` | `export_sngp_to_cuda_datafiles.py` | `BFM_sngp_data_{turb,lam}` |
| Error model (TensorFlow) | `export_error_model_to_npz.py` | `export_error_model_to_cuda_datafiles.py` | `BFM_wall_error_data_{turb,lam}` |
| Baseline wall model (PyTorch) | `export_baseline_to_cuda_datafiles.py` | - | `BFM_wall_data_{turb,lam}` |

The step-1 scripts need TensorFlow and set `TF_USE_LEGACY_KERAS=1` (if unset) before importing it; the step-2
SNGP and error-model scripts need only numpy; the baseline exporter needs torch (it imports `src.wall_model` and
`convert_to_cuda.py` from the repository root).

**Edits needed in CharLES.** The generated `.cu` files hold the numbers. The files in the CharLES tree add, by hand,
a block of `#define` lines at the top of each `.cu` (e.g. `W_in_sngp -> W_in_sngp_turb`) so that the turbulent and
laminar variants can be linked together, and their headers declare the sizes as `static const int` with a
`_turb`/`_lam` suffix. Re-apply these edits when you drop newly exported files into CharLES.

**Where to write.** The examples below export the turbulent reference models (the pretrained models in `./models/sngp/`, `./models/error/` and `./models/baseline/`)
into `./results/export/` (git-ignored); every exporter creates the parent directories of its outputs. Do not write
into the reference model directories: the `assets/` of the exported reference models (the 4000-epoch turbulent
and the laminar SNGP, both error models) already hold their bundles (`sngp_params.npz`, `sngp_params_meta.json`,
`error_model_params.json`). The examples reproduce these files (`error_model_params.json` differs only in the
`model_dir` it records).

### `export_sngp_to_npz.py`

Reads an SNGP SavedModel with `tf.saved_model.load` (no Keras model rebuild) and writes `sngp_params.npz`: the
input z-score statistics, the input Dense layer, the residual Dense layers (spectral-normalization kernels, used as
stored because inference runs with `training=False`), the random-feature GP head (`W_rf`, `b_rf`, `W_out`, `b_out`),
the sizes `Nin/Nhidden/Nres/Ninducing/num_classes/num_classes_train`, and the mean-field constants
`lambda_param = pi/8` and `scale_cov_none = sqrt(1 + pi/8)`.

- `--also_write_json PATH` writes a small sidecar (`sngp_params_meta.json`: metadata and shapes, no weights).
- `--include_gp_precision` also stores `gp_precision_matrix` (1024 x 1024).
- `--dtype float32|float64` sets the numeric type.
- The NPZ stores `dtype` as a string object, so read it with `np.load(..., allow_pickle=True)`.

Requirements on the model directory: `<model_dir>/assets/metadata.json` with the keys `mean_X_train`,
`std_X_train`, `num_classes` and `num_classes_train`, and the variable names `input_layer/{kernel,bias}`,
`spectral_normalization[_i]/{kernel,bias}` and `random_feature_gaussian_process/{kernel,bias,gp_output_bias}`
(plus `gp_precision_matrix` with `--include_gp_precision`). Models trained with `train_turb.py` or
`train_lamturb.py` meet them.

```bash
python tools/export_sngp_to_npz.py \
  --model_dir ./models/sngp/sngp_model_Baseline_cleanedupNew_Add_ExpertExtFlow_ep4000_merged \
  --out_npz ./results/export/sngp_turb/sngp_params.npz \
  --also_write_json ./results/export/sngp_turb/sngp_params_meta.json
```

### `export_sngp_to_cuda_datafiles.py`

Converts `sngp_params.npz` into a header declaring `extern const` arrays and a `.cu` defining them (kernels stored
`(out, in)` row-major; residual kernels and biases concatenated). Symbols carry `--suffix` (default `sngp`, e.g.
`W_in_sngp`, `Nclasses_train_sngp`). Produces CharLES `BFM_sngp_data_turb.{hpp,cu}` (turbulent SNGP, reference
model `..._ep4000_merged`) and `BFM_sngp_data_lam.{hpp,cu}` (laminar SNGP, reference model
`..._UPY0p8_ep20000_merged`), the SNGP confidence of the blended wall model.

```bash
python tools/export_sngp_to_cuda_datafiles.py \
  --npz ./results/export/sngp_turb/sngp_params.npz \
  --out_hpp ./results/export/BFM_sngp_data_turb.hpp --out_cu ./results/export/BFM_sngp_data_turb.cu --suffix sngp
```

### `export_error_model_to_npz.py`

Reads the Dense kernels and biases of an error-model SavedModel directly from its variable checkpoint
(`<model_dir>/variables/variables.index`, via `tf.train.list_variables`; TensorFlow Probability is not needed) and
bundles them with the normalization statistics into `error_model_params.json` (`--format json`, or inferred from
the `.json` extension of `--out`) or an `.npz` (`x_mean, x_std, y_mean, y_std, W_0, b_0, ...`).
`--dtype float32|float64` sets the numeric type.

Requirements on the model directory: `<model_dir>/assets/error_norm_stats.json` with the keys `x_mean`, `x_std`,
`y_mean`, `y_std` (written by the training scripts), and Dense layers named `dense[_i]/{kernel,bias}` or Keras
`layer_with_weights-i` checkpoint entries.

```bash
python tools/export_error_model_to_npz.py \
  --model_dir ./models/error/error_model_Baseline_cleanedupNew_Add_ExpertExtFlow_CustomLayers_N60xL15_wVal_ep3811 \
  --out ./results/export/error_turb/error_model_params.json
```

### `export_error_model_to_cuda_datafiles.py`

Converts `error_model_params.json` into `.hpp/.cu` arrays for a C-style forward pass: `input_layer`, the
concatenated `hidden_layers` and `bias`, `output_layer` (last kernel, then bias; 2 outputs: the mean and the raw
sigma parameter), the sizes `Nin/Nout/Nlayers/Nneurons/act_fn`, and `input_mean/input_std/output_mean/output_std`.
Symbols use `--prefix` (default `err`). The laminar CharLES file uses the default names; the turbulent one uses
`--prefix wall_error`, with the `act_fn` line commented out. Produces CharLES `BFM_wall_error_data_turb.{hpp,cu}`
and `BFM_wall_error_data_lam.{hpp,cu}` (the error model, CharLES keyword `ERROR_MODEL`).

```bash
python tools/export_error_model_to_cuda_datafiles.py \
  --json ./results/export/error_turb/error_model_params.json \
  --out_hpp ./results/export/BFM_wall_error_data_turb.hpp --out_cu ./results/export/BFM_wall_error_data_turb.cu \
  --prefix wall_error
```

### `export_baseline_to_cuda_datafiles.py`

Loads a compact baseline checkpoint with `WallModel.load_compact` (from `src/wall_model.py`) and writes it through
`convert_to_cuda.write_mlp_params_split` as `input_layer_wall`, `hidden_layers_wall`, `bias_wall`,
`output_layer_wall`, plus `input_gains_wall/input_offsets_wall/output_gain_wall/output_offset_wall`. The compact
`.pth` holds the architecture (`InputScaling`, `HiddenLayers`, ...) and the weights but no separate normalization,
so the gains and offsets default to identity (override with `--input_gains/--input_offsets/--output_gain/--output_offset`).
`--act_fn` (default 0 = ReLU) and `--zero_threshold` (default 1e-10: smaller weights and biases are set to zero) are
also exposed. Produces CharLES `BFM_wall_data_turb.{hpp,cu}` (from `cleaned_up_20251208_ymax_0_15_ymin_0_0025.pth`)
and `BFM_wall_data_lam.{hpp,cu}` (from `NN_wm_CH0_G0_S0_tn560665_..._inputs2_final_ep30000_...pth`), the BFM
baseline wall models.

```bash
python tools/export_baseline_to_cuda_datafiles.py \
  --checkpoint ./models/baseline/cleaned_up_20251208_ymax_0_15_ymin_0_0025.pth \
  --out_hpp ./results/export/BFM_wall_data_turb.hpp --out_cu ./results/export/BFM_wall_data_turb.cu
```
