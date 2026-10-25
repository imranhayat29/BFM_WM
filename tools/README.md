# tools/: CharLES BFMv2 export scripts and the a-priori benchmark

The export scripts convert the trained models of this repository into the constant-array data files compiled into the
CharLES BFMv2 solver (`<CharLES_BFMv2 checkout>/src/charles/BFM_*_data_*.{hpp,cu}`). They are copies of the
top-level export scripts of the original `BFM_WM` repo; CLIs, defaults and output formats are unchanged. Run them from
the repository root (`python tools/<script>.py ...`), with the project venv active. The adaptations are listed in the
header comment of each script: `export_baseline_to_cuda_datafiles.py` adds the repo root to `sys.path` (for
`convert_to_cuda.py` and `src.wall_model`) and creates the output directories, and the two TensorFlow readers set
`TF_USE_LEGACY_KERAS=1` (if unset) before TensorFlow is imported. `export_baseline_to_cuda_datafiles.py` was `export_baseline_to_cude_datafiles.py` in
`BFM_WM`; the typo in the file name is fixed.

The pipeline is two steps for the TensorFlow models (SavedModel -> parameter bundle -> `.hpp/.cu`) and one step for
the PyTorch baseline (`.pth` -> `.hpp/.cu`). The generated `.cu` files hold the numbers; the CharLES copies add a
block of `#define` lines (e.g. `W_in_sngp -> W_in_sngp_turb`) so that the turbulent and laminar variants can be
linked together, and their headers use `static const int` sizes with a `_turb`/`_lam` suffix. Re-applying those
edits by hand is still needed when the files are dropped into CharLES.

The examples below export the shipped turbulent models into `./results/export/` (git-ignored); every exporter creates
the parent directories of its outputs. Do not write into the shipped model directories: their `assets/` already hold
the bundles of the shipped models (`sngp_params.npz`, `sngp_params_meta.json`, `error_model_params.json`), which these
examples reproduce (`error_model_params.json` differs only in the absolute `model_dir` it records).

The a-priori benchmark `apriori_benchmark.py` is described in the [last section](#apriori_benchmarkpy-multi-dataset-a-priori-benchmark).

## `export_sngp_to_npz.py`

Reads an SNGP SavedModel with `tf.saved_model.load` (no Keras model rebuild) and writes `sngp_params.npz`: the
input z-score statistics, the input Dense layer, the residual Dense layers (spectral-normalization kernels, used as
stored because inference runs with `training=False`), the random-feature GP head (`W_rf`, `b_rf`, `W_out`, `b_out`),
the sizes `Nin/Nhidden/Nres/Ninducing/num_classes/num_classes_train`, and the mean-field constants `lambda_param = pi/8`
and `scale_cov_none = sqrt(1 + pi/8)`. `--also_write_json` writes a small sidecar (`sngp_params_meta.json`: metadata
and shapes, no weights). The NPZ stores `dtype` as a string object, so read it with `np.load(..., allow_pickle=True)`.
Requires `<model_dir>/assets/metadata.json` with the keys `mean_X_train`, `std_X_train`, `num_classes`,
`num_classes_train`, and the pinned variable names `input_layer/{kernel,bias}`,
`spectral_normalization[_i]/{kernel,bias}` and `random_feature_gaussian_process/{kernel,bias,gp_output_bias}`
(plus `gp_precision_matrix` with `--include_gp_precision`).

```bash
python tools/export_sngp_to_npz.py \
  --model_dir ./models/sngp_model_Baseline_cleanedupNew_Add_ExpertExtFlow_ep4000_merged \
  --out_npz ./results/export/sngp_turb/sngp_params.npz \
  --also_write_json ./results/export/sngp_turb/sngp_params_meta.json
```

## `export_sngp_to_cuda_datafiles.py`

Converts `sngp_params.npz` into a header declaring `extern const` arrays and a `.cu` defining them (kernels stored
`(out, in)` row-major; residual kernels and biases concatenated). Symbols carry `--suffix` (default `sngp`, e.g.
`W_in_sngp`, `Nclasses_train_sngp`). Feeds CharLES `BFM_sngp_data_turb.{hpp,cu}` (turbulent SNGP, ep4000) and
`BFM_sngp_data_lam.{hpp,cu}` (laminar SNGP, UPY0p8 ep20000), i.e. the SNGP confidence of the blended wall model.

```bash
python tools/export_sngp_to_cuda_datafiles.py \
  --npz ./results/export/sngp_turb/sngp_params.npz \
  --out_hpp ./results/export/BFM_sngp_data_turb.hpp --out_cu ./results/export/BFM_sngp_data_turb.cu --suffix sngp
```

## `export_error_model_to_npz.py`

Reads the Dense kernels/biases of an error-model SavedModel directly from its variable checkpoint
(`<model_dir>/variables/variables.index`, via `tf.train.list_variables`; no TensorFlow Probability needed) and bundles
them with the normalization statistics into `error_model_params.json` (`--format json`, or by the `.json` extension)
or an `.npz` (`x_mean, x_std, y_mean, y_std, W_0, b_0, ...`). Requires `<model_dir>/assets/error_norm_stats.json`
with the keys `x_mean`, `x_std`, `y_mean`, `y_std` (legacy format) and Dense layers named `dense[_i]/{kernel,bias}`
or Keras `layer_with_weights-i` checkpoint entries.

```bash
python tools/export_error_model_to_npz.py \
  --model_dir ./models/error_model_Baseline_cleanedupNew_Add_ExpertExtFlow_CustomLayers_N60xL15_wVal_ep3811 \
  --out ./results/export/error_turb/error_model_params.json
```

## `export_error_model_to_cuda_datafiles.py`

Converts `error_model_params.json` into `.hpp/.cu` arrays for a C-style forward pass: `input_layer`, concatenated
`hidden_layers` and `bias`, `output_layer` (last kernel then bias; 2 outputs: mean and raw sigma parameter), the
sizes `Nin/Nout/Nlayers/Nneurons/act_fn`, and `input_mean/input_std/output_mean/output_std`. Symbols use `--prefix`
(default `err`: the CharLES laminar file uses these names; the turbulent one matches `--prefix wall_error`, with the `act_fn` line commented out). Feeds CharLES
`BFM_wall_error_data_turb.{hpp,cu}` and `BFM_wall_error_data_lam.{hpp,cu}` (aleatoric error model, `ERROR_MODEL`).

```bash
python tools/export_error_model_to_cuda_datafiles.py \
  --json ./results/export/error_turb/error_model_params.json \
  --out_hpp ./results/export/BFM_wall_error_data_turb.hpp --out_cu ./results/export/BFM_wall_error_data_turb.cu \
  --prefix wall_error
```

## `export_baseline_to_cuda_datafiles.py`

Loads a compact baseline checkpoint with `WallModel.load_compact` (from `src/wall_model.py`) and writes it through
`convert_to_cuda.write_mlp_params_split` as `input_layer_wall`, `hidden_layers_wall`, `bias_wall`,
`output_layer_wall`, plus `input_gains_wall/input_offsets_wall/output_gain_wall/output_offset_wall`. The compact
`.pth` holds the architecture (`InputScaling`, `HiddenLayers`, ...) and weights but no separate normalization, so the
gains/offsets default to identity (override with `--input_gains/--input_offsets/--output_gain/--output_offset`);
`--act_fn` (default 0 = ReLU) and `--zero_threshold` (default 1e-10) are also exposed. Feeds CharLES
`BFM_wall_data_turb.{hpp,cu}` (`cleaned_up_20251208_ymax_0_15_ymin_0_0025.pth`) and `BFM_wall_data_lam.{hpp,cu}`
(`NN_wm_CH0_G0_S0_tn560665_..._inputs2_final_ep30000_...pth`), i.e. the BFM baseline wall models.

```bash
python tools/export_baseline_to_cuda_datafiles.py \
  --checkpoint ./models/cleaned_up_20251208_ymax_0_15_ymin_0_0025.pth \
  --out_hpp ./results/export/BFM_wall_data_turb.hpp --out_cu ./results/export/BFM_wall_data_turb.cu
```

## `apriori_benchmark.py` (multi-dataset a-priori benchmark)

Port of the `Error_vs_baseline_plotting` notebooks of the BFM_WM fork onto the shared v2 pipeline. They are called
EVB (`Error_vs_baseline_plotting.ipynb`), LOG (its log-log variant `_loglog`) and AIAA (the always-corrected vs blended
PDF of `_aiaa`) below; see Provenance in the main README. Every dataset goes through `src_err/pipeline.predict_turb`,
the same code as `test_turb.py` (baseline, error model, SNGP confidence, EQWM, 3-switch blend with inference-only
gates, magnitudes everywhere).
Laminar/transitional sets also take the turbulent path (there is no Kt/Km switch in the benchmark). The models are
loaded once; one pass per dataset gives the blended, the always-corrected and the EQWM predictions. Configuration:
`inputfiles/benchmark/apriori_benchmark.toml`, self-contained (own model paths and gate thresholds, same defaults as
`test/test_turb.toml`), every key commented.

```bash
python tools/apriori_benchmark.py benchmark/apriori_benchmark.toml --gpu 0          # full list (80 keys)
python tools/apriori_benchmark.py benchmark/apriori_benchmark.toml --dry-run        # resolve every dataset/model file only
python tools/apriori_benchmark.py benchmark/apriori_benchmark.toml --gpu -1 \
  --datasets CH TBL_5 "Gaussian bump 2M" --max-samples 3000 --output-dir results/smoke/benchmark
python tools/apriori_benchmark.py --replot results/smoke/benchmark --scale log      # figures only, no torch/TF
```

CLI: `<toml>` (under `./inputfiles/` or a path; optional with `--replot`), `--gpu N` (-1 = CPU), `--output-dir DIR`,
`--max-samples N` (per dataset, deterministic, after the domain filter), `--datasets k1 k2 ...` (subset; a `[groups]`
name expands to its members; a group is pooled from the members selected), `--scale linear|log|both`,
`--replot RUN_DIR` (regenerates every figure from `RUN_DIR/cases.csv` + `points.npz`; plot options from the given TOML,
else from `RUN_DIR/overall.json`; figures go to `--output-dir` or `RUN_DIR`), `--dry-run`.

Per dataset: load with the `[data]` window (default `[0.0025, 0.15]` n1/delta, the turbulent checkpoint's window, as
EVB), then the evaluation-domain filter `[evaluation].DomainFilter` (`"retau"`: keep |u_tau delta/nu| > `RetauMin`
= 5, EVB; `"piout"`: keep |pi_out,true| >= `PioutMin` = 2; `"none"`; a zero minimum disables), then the sample cap and
the pipeline. The truth only selects evaluation points and gives the percent errors (low-value blend 2 +- 1 on
|pi_out,true|); it never enters a prediction. Then a joint outlier mask (a point is dropped if any of |e_base%|,
|e_final%|, |e_eqwm%| exceeds `OutlierPercent` = 10000) and the optional error-change filter
(`MinErrorChangePercent`, off). Missing dataset files and models are reported before the models are loaded; a
dataset that fails later (unreadable or empty) is recorded as skipped with its reason.

Cases: every ungrouped dataset and every `[groups]` entry (default `"Gaussian bump 2M"` = the four `gaussian_2M_*`
subsets, pooled point-wise; members are not plotted individually). Role (`training`/`untrained`, blue/maroon) from
`[data].TrainingDatasets`; a group is training if it is listed or any member is. Per case: mean/median/P90 of |e%| for
the baseline, the final (blended), the always-corrected and the EQWM predictions; improvement (mean baseline - mean
final); max point-wise improvement; mean sigma% (baseline-normalised), mean C, mean alpha_a/alpha_c/alpha_nw; fraction
fully corrected and fraction of full EQWM fallback; n_raw (after the window), n_domain, n_eval (after the cap), n_used.
Overall: the unweighted mean over cases (EVB headline), the point-weighted mean, and both per role.

Outputs (in `OutputDir`): `cases.csv` (one row per dataset/member/group/skipped dataset; also the fraction of points
whose stored network input is inconsistent with u1 n1/nu), `overall.json` (resolved config, CLI, model paths, git
commit, versions, overall averages, skipped datasets, timings), `points.npz` (float32 per-point errors, sigma%, C,
alphas, pi_out values; `dataset`/`case` indices into `dataset_keys`/`case_names`, `training`, `used`), the EVB-style
console table and the domain-filter table, and the figures `<RunName>_<fig>[_log].<fmt>`: `train_test` (linear with
the zoom box and legend; `_log` log-log after the LOG notebook: non-positive cases skipped with a warning, limits <= 0
automatic), `train_test_zoom` (linear only), `sigma` and `confidence` (jet-coloured scatter), `pdf` and
`pdf_confidence` (pooled |e%| PDFs, curves from `PdfCurves`, subset from `PdfSubset`; KDEs are conditional on the
range), `pdf_always_vs_blended` (AIAA comparison, from the same pass, needs `CompareAlways`).

Differences from EVB: ep4000 SNGP instead of the unshipped ep3500; the 3-switch blend of `test_turb.py` with the
EQWM fallback (not EVB's binary gate); |e%| with the low-value blend instead of /(|t|+1e-6); one joint outlier mask
instead of per-array cuts; error-change filter off by default; no kNN/aleatoric-model sigma, binary gates, tau_w
metric, bar charts or automatic discovery; dataset key fixes (`gaussian_2M` -> 4 pooled subsets,
`Transition_transition` evaluated and used for the training role, `hump_APG_SEP2` added).
