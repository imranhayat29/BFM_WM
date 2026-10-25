# tools/: CharLES export scripts and the a-priori benchmark

This directory holds two kinds of tools:

| Script | Purpose |
| --- | --- |
| `export_sngp_to_npz.py` | SNGP SavedModel -> parameter bundle `sngp_params.npz` |
| `export_sngp_to_cuda_datafiles.py` | `sngp_params.npz` -> CharLES data files `.hpp/.cu` |
| `export_error_model_to_npz.py` | error-model SavedModel -> parameter bundle `error_model_params.json` (or `.npz`) |
| `export_error_model_to_cuda_datafiles.py` | `error_model_params.json` -> CharLES data files `.hpp/.cu` |
| `export_baseline_to_cuda_datafiles.py` | baseline checkpoint `.pth` -> CharLES data files `.hpp/.cu` |
| `apriori_benchmark.py` | multi-dataset a-priori benchmark of the turbulent wall model ([last section](#a-priori-benchmark-apriori_benchmarkpy)) |

Run every script from the repository root (`python tools/<script>.py ...`) with the project venv active.
`--help` lists the options of each script.

## Exporting models to CharLES BFMv2

CharLES BFMv2 compiles the wall-model networks in as constant arrays, in the files
`src/charles/BFM_*_data_*.{hpp,cu}` of its source tree (on the cluster:
`<CharLES_BFMv2 checkout>`). The export scripts write these files from the models of
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

**Where to write.** The examples below export the turbulent reference models (the pretrained models in `./models/`)
into `./results/export/` (git-ignored); every exporter creates the parent directories of its outputs. Do not write
into the reference model directories: their `assets/` already hold the bundles of the reference models
(`sngp_params.npz`, `sngp_params_meta.json`, `error_model_params.json`). The examples reproduce these files
(`error_model_params.json` differs only in the absolute `model_dir` it records).

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
  --model_dir ./models/sngp_model_Baseline_cleanedupNew_Add_ExpertExtFlow_ep4000_merged \
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
  --model_dir ./models/error_model_Baseline_cleanedupNew_Add_ExpertExtFlow_CustomLayers_N60xL15_wVal_ep3811 \
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
  --checkpoint ./models/cleaned_up_20251208_ymax_0_15_ymin_0_0025.pth \
  --out_hpp ./results/export/BFM_wall_data_turb.hpp --out_cu ./results/export/BFM_wall_data_turb.cu
```

## A-priori benchmark (`apriori_benchmark.py`)

The benchmark evaluates the turbulent wall model a priori (on stored datasets of wall-normal profiles, without
running a flow solver) on many datasets at once and compares, per dataset, the mean error of the baseline with the mean
error of the blended model. Every dataset goes through `src_err/pipeline.predict_turb`, the same code as
`test_turb.py`: baseline, error model, SNGP confidence, EQWM (equilibrium wall model) fallback, and the blend of
three smooth switches whose inputs use inference-time information only. Laminar and transitional datasets also take
this turbulent path (there is no Kt/Km switch in the benchmark). The models are loaded once; one pass per dataset
gives the blended, the always-corrected (`|y_base,raw + mu|` everywhere) and the EQWM predictions.

The configuration is `inputfiles/benchmark/apriori_benchmark.toml`. It is self-contained (its own model paths and
gate thresholds, with the same defaults as `test/test_turb.toml`), and every key is commented there.

### Running

```bash
python tools/apriori_benchmark.py benchmark/apriori_benchmark.toml --gpu 0          # all 80 datasets
python tools/apriori_benchmark.py benchmark/apriori_benchmark.toml --dry-run        # only resolve every dataset/model file
python tools/apriori_benchmark.py benchmark/apriori_benchmark.toml --gpu -1 \
  --datasets CH TBL_5 "Gaussian bump 2M" --max-samples 3000 --output-dir results/smoke/benchmark
python tools/apriori_benchmark.py --replot results/smoke/benchmark --scale log      # figures only, no torch/TF
```

| Option | Meaning |
| --- | --- |
| `<toml>` | benchmark TOML under `./inputfiles/` or a path; optional with `--replot` |
| `--gpu N` | GPU index, -1 = CPU; overrides `[general].GpuNum` |
| `--output-dir DIR` | output directory; overrides `[general].OutputDir` (with `--replot`: where the figures go, default `RUN_DIR`) |
| `--max-samples N` | per-dataset deterministic sample cap, applied after the domain filter (0 = all); overrides `[data].MaxSamplesPerDataset` |
| `--datasets k1 k2 ...` | evaluate only these keys; a `[groups]` name expands to its members, and a group is pooled from the members selected |
| `--scale linear\|log\|both` | scatter axes; overrides `[plots].Scale` |
| `--replot RUN_DIR` | regenerate every figure from `RUN_DIR/cases.csv` + `points.npz`, without torch, TensorFlow or inference; plot options from the given TOML, else from `RUN_DIR/overall.json` |
| `--dry-run` | validate the TOML and resolve every dataset file, then exit (no models loaded) |

### Processing of each dataset

1. **Load** with the `[data]` wall-distance window (default n1/delta in `[0.0025, 0.15]`, the training window of
   the turbulent baseline) and `Stride`.
2. **Domain filter** `[evaluation].DomainFilter`: `"retau"` keeps points with |u_tau delta/nu| > `RetauMin` (5),
   `"piout"` keeps |pi_out,true| >= `PioutMin` (2), `"none"` keeps all; a zero minimum disables the filter.
3. **Sample cap** `MaxSamplesPerDataset` (deterministic, evenly spaced).
4. **Prediction** with the pipeline.
5. **Percent errors** |e%| = 100 |e|/|pi_out,true|, blended with 100 |e|/sqrt(|pi_1|) where |pi_out,true| is small
   (ramp 2 +- 1, `[metrics].LowValue`), so that near-zero targets do not blow up the errors.
6. **Outlier mask** (joint): a point is dropped if any of |e_base%|, |e_final%|, |e_eqwm%| exceeds
   `OutlierPercent` (10000).
7. **Error-change filter** (optional, off by default): `MinErrorChangePercent > 0` keeps only points where
   the blended error differs from the baseline error by more than that value. It biases the statistics towards
   points the correction changes.

The dataset truth (the target and the dataset u_tau) only selects evaluation points and gives the percent errors;
it never enters a prediction. Missing dataset files and models are reported before the models are loaded; a dataset
that fails later (unreadable, or no points left after a filter) is recorded as skipped with its reason.

### Cases, roles and statistics

**Cases.** Every dataset that is not in a group is one case. Every `[groups]` entry is one case whose members'
points are pooled point-wise; members are not plotted individually. The default group `"Gaussian bump 2M"` pools the
four `gaussian_2M_*` subsets (APG, FPG, SEP, MAPG) the baseline was trained on.

**Roles.** A case is `training` (blue) if it is listed in `[data].TrainingDatasets`, otherwise `untrained`
(maroon); a group is `training` if its name or any member is listed. The default list holds the training sets of the
turbulent baseline plus the additional training sets of the error model and SNGP. The JHU transitional boundary
layer appears twice: `Transition_transition` (its transition region, which the error model and SNGP were trained on,
role `training`) and the full `Transition` case (role `untrained`).

**Per case:**

- mean, median and P90 of |e%| for the baseline, the final (blended), the always-corrected and the EQWM predictions;
- improvement = mean baseline |e%| - mean final |e%|, and the maximum point-wise improvement;
- mean sigma% (the error model's aleatoric uncertainty sigma as a percentage of the baseline output, with the same
  kind of low-value blend), mean confidence C,
  and the mean switch weights alpha_a (aleatoric), alpha_c (confidence) and alpha_nw (near-wall y+);
- fraction of points fully corrected (all three weights 1) and fraction of full EQWM fallback;
- point counts n_raw (after the window), n_domain (after the domain filter), n_eval (after the cap), n_used (after
  the outlier and error-change filters).

**Overall:** the unweighted mean over cases (each case counts once; the headline number), the point-weighted mean
over all used points, and both per role.

### Outputs

Written to `[general].OutputDir` (default `results/apriori_benchmark`):

| File | Content |
| --- | --- |
| `cases.csv` | one row per dataset, group member, group and skipped dataset (column `kind`), with the statistics above and the fraction of points whose stored network input is inconsistent with u1 n1/nu (a data check) |
| `overall.json` | resolved configuration, command line, model paths, git commit, package versions, overall averages, skipped datasets, timings |
| `points.npz` | float32 per-point errors, sigma%, C, switch weights and pi_out values; `dataset`/`case` indices into `dataset_keys`/`case_names`; the `training` and `used` masks |
| console | a summary table of the mean \|e%\| per case with the overall averages, and a table of the points removed by the domain filter per dataset |

Figures are named `<RunName>_<figure>[_log].<fmt>` (`[plots].Figures` selects them, `Formats` sets the formats):

| Figure | Content |
| --- | --- |
| `train_test` | one marker per case at (mean baseline \|e%\|, mean final \|e%\|), coloured by role, with the y = x line ("Same Performance"), the legend and, on linear axes, the dash-dot zoom box. With `Scale = "log"` or `"both"` also `train_test_log` on log-log axes: cases with non-positive or non-finite errors are skipped with a warning, and a limit <= 0 is set automatically from the data |
| `train_test_zoom` | the same scatter within `ZoomLimits` (linear only) |
| `sigma` | the scatter coloured by the mean sigma% of each case (jet colour map, `SigmaColorRange`) |
| `confidence` | the scatter coloured by the mean confidence C of each case (`ConfidenceColorRange`) |
| `pdf`, `pdf_confidence` | probability density functions (PDFs) of \|e%\| and of C, pooled over the used points; curves from `PdfCurves`, points from `PdfSubset`; kernel density estimates (KDEs) are fitted inside the plotted range, so the densities are conditional on that range |
| `pdf_always_vs_blended` | \|e%\| PDFs of the baseline, the always-corrected and the blended predictions (needs `CompareAlways = true`) |

### Method notes

- The percent errors use the low-value blend of step 5 instead of dividing by |pi_out,true| alone.
- One joint outlier mask is applied to all predictions, so the baseline, final and EQWM statistics of a case use
  the same points.
- The error-change filter is off by default (it biases the statistics).
- Only the datasets listed in `[data].Datasets` are evaluated; there is no automatic discovery of data files.
