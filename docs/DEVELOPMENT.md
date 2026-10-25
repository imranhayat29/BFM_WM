# BFM_WM_v2: developer notes

Provenance and maintenance notes for the owner and future developers. Users should read
[README.md](../README.md), [tools/README.md](../tools/README.md) and the comments in `inputfiles/**/*.toml`;
those files describe what the code does, without history. This file records where the code came from, how it
relates to the notebooks it replaces, what was decided and why, and what is still open.

Contents:
1. [Origin](#1-origin)
2. [Abbreviations and cell references](#2-abbreviations-and-cell-references)
3. [Notebook to v2 map](#3-notebook-to-v2-map)
4. [Behaviour relative to the notebooks](#4-behaviour-relative-to-the-notebooks)
5. [Data provenance](#5-data-provenance)
6. [Reference models](#6-reference-models)
7. [Annotated TOMLs](#7-annotated-tomls)
8. [Decision record](#8-decision-record)
9. [CharLES BFMv2 parity](#9-charles-bfmv2-parity)
10. [Analyses worth keeping](#10-analyses-worth-keeping)
11. [Development notes](#11-development-notes)
12. [Reproduction of the reference models with v2](#12-reproduction-of-the-reference-models-with-v2)

---

## 1. Origin

### 1.1 Repositories

| Name | Location | Role |
|---|---|---|
| upstream | [github.com/yuenongling/BFM_WM](https://github.com/yuenongling/BFM_WM), remote `upstream` | Baseline BFM wall model: `src/`, `train_wm_proposal.py`, `test_model*.py`, `wall_model_cases.py`, `data/` (git-lfs) |
| notebook fork, "BFM_WM" | [github.com/imranhayat29/BFM_WM](https://github.com/imranhayat29/BFM_WM), branch `wip-bfm-error-model`; local copy `<local clone of BFM_WM>` (treat as read-only) | The owner's fork that added the error model, the SNGP and the blended correction, driven by Jupyter notebooks (`src_err/*.py` helpers, notebooks at the top level) |
| BFM_WM_v2 (this repo) | `<this repository>` | Clone of upstream plus a script/TOML refactor of the notebook fork |
| expert script | `BFM_WM/test_expert_model.py`; standalone copy `<local copy of BFM_WM_expert>` | Earlier inference-only script of the blended model (argparse, `--gating-reference`); a reference for the test scripts' `summary.json` |
| data generation | [github.com/yuenongling/WM_data_creation](https://github.com/yuenongling/WM_data_creation); local `<local clone of WM_data_creation>` | Creates the HDF5 datasets; the place to regenerate data in another frame (see [9.3](#93-remaining-difference-the-input-sign-frame)) |
| CharLES BFMv2 | `<CharLES_BFMv2 checkout>` (local git) | LES solver that deploys the exported models |

### 1.2 The notebooks

All are in the top level of the notebook fork and are **not in its git history** (the two error-model notebooks are
git-ignored, the three benchmark notebooks are untracked). "Final version" means the file as it was on 2026-10-25:

| Abbrev. | File | Content | Cells | sha256 (first 12) |
|---|---|---|---|---|
| TURB-NB | `BFMWM_ErrorModel copy_gaussian2M_cleaned4context.ipynb` | turbulent-only: SNGP + error-model training, a-priori test on the Gaussian bump 2M | 69 | `398d9c66545e` |
| LT-NB | `BFMWM_ErrorModel copy_aairfoil2M-LamTurbModel-cleaned4context2.ipynb` | laminar-turbulent: two branches (`active_pipeline_key = 'lam'`/`'turb'`, run once per branch by hand), test on the A-airfoil 2M | 83 | `1dbfb59669dd` |
| EVB | `Error_vs_baseline_plotting.ipynb` | multi-dataset a-priori benchmark (baseline vs corrected mean error per dataset) | 47 | `efdca5a159fc` |
| LOG | `Error_vs_baseline_plotting_loglog.ipynb` | EVB with log-log scatter axes | 49 | `1b689c038cfb` |
| AIAA | `Error_vs_baseline_plotting_aiaa.ipynb` | EVB variant with the always-corrected vs blended error PDF | 49 | `9558f0129ebc` |

The many other `BFMWM_ErrorModel copy_*.ipynb` notebooks of the fork (per-case variants, aleatoric-only model,
a-posteriori class import, ...) were not ported. To read a notebook as a script:
`jupyter nbconvert --to script "<file>.ipynb"`, or a short `json` loop that prints `# %% cell <i>` before each cell
(this is how the cell numbers in [section 2](#2-abbreviations-and-cell-references) were counted).

### 1.3 Git layout of this repository

- `git clone` of upstream master at `7794771`; remote `origin` renamed to `upstream`. No remote of its own yet and
  nothing pushed (decision D1). No GitHub fork was created.
- Branch `master` = upstream; branch `v2` = the refactor, in logical commits:

  | Commit | Content |
  |---|---|
  | `22d60f6` | Port the fork's changes to the parent package (`src/`, `wall_model_cases.py`, `plot_utils.py`, `train_wm_proposal.py`) |
  | `7e19ad6` | Training data of the fork (git-lfs), see [section 5](#5-data-provenance) |
  | `b26286f` | `src_err/` |
  | `d48da23` | Training scripts and TOMLs |
  | `11332b4` | Test scripts and TOMLs |
  | `f65de67` | CharLES exporters and the a-priori benchmark |
  | `5eb44fb` | README and requirements |
  | `c459004` | Final-review fixes (single baseline-output helper, consistent model checks, docs); last commit whose TOMLs carry the notebook pointers |
  | `9acf1e8` | User docs for fresh users (no notebook references); this file; absolute path of the lam-turb test set |

  Later commits: `git log upstream/master..v2`.

- To follow upstream: `git fetch upstream && git merge upstream/master` on `v2`. Parent files were changed as
  little as possible (D4) so that this stays easy; every change is marked `# BFM_WM_v2:` (see [11.4](#114-conventions)).

### 1.4 What came from the notebook fork

| Item | v2 location | Notes |
|---|---|---|
| `src/` changes: `flow_type` exposure in `load_data()`, exact inverse of the signed log1p in near-wall plots, default `downy` 0.0, no `u1 > 0` mask in near-wall mode, the `upy` in the filter message | `src/wall_model.py`, `src/data_handler.py` | Ported unchanged (D5) |
| `plot_utils.py` guard for the missing personal style file | `plot_utils.py` | v2 never loads the style file |
| All dataset keys added in the fork (A-airfoil full-2D fields and region splits, `hump_*` regions, `transition_JHU`/`ph_B` full-2D fields, `aposteriori_*`) | `wall_model_cases.py` | Identical to the fork's file plus a comment block and the three `down0p005_up1_laminar_*` keys (new in v2) |
| `src_err/SNGP.py`, `error_model_functs.py`, `conditional_correction.py`, `select_by_x_wallnormal.py`, `pdf_utils.py`, `plot_err_utils.py`, `error_data_handler.py` | `src_err/sngp.py`, `error_model.py`, `gating.py`, `selection.py`, `plotting.py`, `data.py` | Rewritten, see [3.1](#31-fork-modules-to-v2-modules) |
| Export scripts `export_sngp_to_npz.py`, `export_sngp_to_cuda_datafiles.py`, `export_error_model_to_npz.py`, `export_error_model_to_cuda_datafiles.py`, `export_baseline_to_cude_datafiles.py`, `convert_to_cuda.py` | `tools/` (`convert_to_cuda.py` at the root) | Copied from the fork's top level; CLIs, defaults and output formats unchanged; file-name typo `cude` fixed. Adaptations (listed in each header): the baseline exporter inserts the repo root into `sys.path` (for `convert_to_cuda` and `src.wall_model`) and creates the output directories; the two TensorFlow readers `setdefault` `TF_USE_LEGACY_KERAS=1` before TensorFlow is imported; example usage updated to `tools/` |
| Data files | `data/` | See [section 5](#5-data-provenance) |
| The 7 model artifacts | `models/` (git-ignored) | Byte-identical copies, see [section 6](#6-reference-models) |

Changes made in v2 itself (not in the fork): `RuntimeError` instead of `raise "device is incorrect"` and the optional
`[general] Seed` in `train_wm_proposal.py`; split indices stored in baseline checkpoints (`src/trainer.py`,
`src/wall_model.py`); `inputfiles/` tracked (`.gitignore`); `test_model.py` usage line in the README.

---

## 2. Abbreviations and cell references

Code comments and docstrings in `src_err/`, `tools/` and the top-level scripts keep their notebook pointers on
purpose (traceability). They use:

| Token | Meaning |
|---|---|
| `TURB-NB`, `LT-NB`, `EVB`, `LOG`, `AIAA` | the notebooks of [1.2](#12-the-notebooks) |
| `cN`, `cell N`, `cells N-M` | **0-based index into the notebook's `cells` list, markdown cells included**, of the final version (e.g. `TURB-NB c64` = `nb["cells"][64]`) |
| `EVB F1`-`F4` | EVB's figures 1-4 (train/test scatter, zoom, σ-coloured, confidence-coloured) |
| "the notebooks" | TURB-NB and LT-NB together (they share most code) |
| "BFM_WM", "the fork" | the notebook fork (`BFM_WM/src_err/...` = its helper modules) |
| "shipped models" | the 7 reference models of [section 6](#6-reference-models) (the user docs say "reference models") |
| "parent", "upstream" | yuenongling/BFM_WM |
| "expert script" | `BFM_WM/test_expert_model.py` |
| `D*`, `S*`, `G*`, `E*`, `T*`, `B*` | decision and bug IDs of the decision log ([section 8](#8-decision-record)) |

Cell references found in the code (all 0-based):

| Reference | Content | v2 |
|---|---|---|
| TURB-NB c9-c11 / LT-NB c12-c14 | test datasets, `downsample_factor`, additional datasets | test `[data]`, training `[data] AddDatasets` |
| TURB-NB c14-c24 / LT-NB c14-c29 | SNGP flags, classes, model, training, confusion matrix (c24) | `src_err/sngp.py`, `classes.py`, `training.py`; `sngp/*.toml` |
| TURB-NB c26 / LT-NB c31 | 19-colour class map | `plotting.py` (class colours) |
| TURB-NB c30-c35 / LT-NB c34-c42, c46 | error-model flags, training, saving | `src_err/error_model.py`, `training.py`; `error/*.toml` |
| TURB-NB c40 / LT-NB c52 | EQWM via `model._eqwm_solve` (parent `log_law_solve`) | `gating.py` (`Law = "parent"`) |
| TURB-NB c42 / LT-NB c54 | gate thresholds, low-value threshold | test `[gates]`, `[metrics]` |
| LT-NB c2, c3, c6, c8 | `active_pipeline_key`, `kt_km_threshold`, `infer_ktkm_from_flow_name`, windows of the loaded configs | `train_lamturb.py --branch`, `[switch]`, `data.py`, `[data] upy/downy` |
| LT-NB c65 | 3-switch blend with the laminar-side guard, composite labels | `pipeline.predict_lamturb`, `lamturb.py` |
| LT-NB c70 | `mask_eqwm_fallback` (plot hatching) | `pipeline.ood_and_nearwall_masks` |
| TURB-NB c64 / LT-NB c74, c75 | spatial maps, raw-n1 maps of the laminar region | `plotting.py` maps, `reporting.py` |
| TURB-NB c63, c66, c67 / LT-NB c73, c81 | station selection, C_f lines, moving average | `selection.py`, `plotting.py` C_f lines, `[plots.cf_lines]` |
| LT-NB c78 | smoothed tripcolor maps | `plotting.py` tripcolor, `[plots.tripcolor]` |
| EVB c6, c7, c10, c26 | thresholds (`ERROR_THRESHOLD_PERCENT`, `Retau_thresh`, `thresh_err_change`), SNGP path, `RENAME_MAPPING`, `CASE_MARKER_MAP`/`_RULES` | benchmark `[evaluation]`, `[models]`, `[display_names]`, `plotting.py` markers |
| AIAA c48 | `plot_error_pdf_no_filter_vs_filter` | benchmark figure `pdf_always_vs_blended` |

---

## 3. Notebook to v2 map

### 3.1 Fork modules to v2 modules

| Fork (`BFM_WM/src_err/`) | v2 (`src_err/`) | Main changes |
|---|---|---|
| `SNGP.py` + notebook SNGP cells | `sngp.py`, `classes.py` | pinned layer names; dropout applied when `DropoutActive`; configurable architecture, class weights, LR decay unit; prediction z-scores with the metadata statistics; confidence on un-merged K, K_tr; labels = argmax over trained classes; optional GP covariance; class bookkeeping by name |
| `error_model_functs.py` + notebook error cells | `error_model.py` | vestigial features dropped ([4.4](#44-dropped-features)); best-epoch restore and snapshots as TOML keys; legacy `error_norm_stats.json` kept |
| `conditional_correction.py` (`compute_blended_correction`) | `gating.py` | same nesting; inference-only gates; CharLES EQWM law; low-value blend; laminar law |
| `select_by_x_wallnormal.py` | `selection.py` | |
| `pdf_utils.py`, `plot_err_utils.py`, notebook plot cells | `plotting.py`, `reporting.py` | unique file names, each figure family switchable, magnitudes |
| `error_data_handler.py`, notebook data cells | `data.py` | columns by name; seeded splits; `downy` honoured when `upy >= 1` |
| notebook test cells | `pipeline.py`, `evaluation.py`, `test_turb.py`, `test_lamturb.py` | one shared pipeline for both test scripts and the benchmark |
| notebook flags | `config.py` + TOMLs | validated schemas; unknown keys are errors |
| `hierarchical_error_model.py`, `optionB.py`, `optionC.py`, `SNGP-backup.py`, `make_hdf5_from_csv*.py`, `load_coords_from_txt.py` | — | not ported |

### 3.2 Notebook flags to TOML keys

| Notebook flag (cell) | v2 |
|---|---|
| `train_flag`, `model_path` (baseline; LT-NB `model_path_lam/_turb`) | `train_wm_proposal.py`; `[baseline] Checkpoint` (training), `[models] Baseline` (test) |
| `active_pipeline_key` (LT-NB c2) | `train_lamturb.py --branch lam\|turb\|both`; `[lam]`/`[turb]` tables |
| `train_sngp_flag` / `save_sngp_flag` (TURB-NB c14, LT-NB c18) | `--sngp`; `[general] Save`, `Overwrite` |
| `train_err_model_flag` / `save_flag` (TURB-NB c30, LT-NB c35) | `--error`; `[general] Save`, `Overwrite` |
| `include_extra_data_flag`, `add_datasets` (TURB-NB c11, LT-NB c14) | `[data] AddDatasets` (`[]` = none), `TrainRatio` |
| `config_loaded_model['data']['upy'/'downy']` (LT-NB c8) | `[data] upy`, `downy`, per model |
| `labels_to_remove` (integers; TURB-NB c16) | `[classes] RemoveClasses` (names) |
| `merge_groups_flag`, `group_definitions` (TURB-NB c18) | `[classes.MergeGroups]` (names) |
| `rename_mapping` (TURB-NB c20) | `[classes.DisplayNames]` (stored in the SNGP metadata; test TOMLs may override) |
| `apply_posthoc_merging`, `posthoc_group_definitions` | `[classes.PosthocGroups]` (training: metadata; test: `[classes]`) |
| `num_classes` (TURB-NB c19) | `[classes] NumClasses` |
| `downsample_sngp_data_factor` | `[data] DownsampleBaseline` |
| `recompute_norm_with_extra_flag` | `[data] RecomputeNorm` |
| SNGP `epochs`, `batch_size`, `lr_schedule` | `[training] Epochs`, `BatchSize`, `LearningRate`, `EndLearningRate`, `DecayUnit` |
| SNGP architecture (hard-coded) | `[model] NumLayers`, `NumHidden`, `SpecNormBound`, `NumInducing`, `GpKernelScale`, `Dropout`, `DropoutActive` |
| class weights (always balanced) | `[training] ClassWeights` |
| `custom_err_layers_flag`, `Nneurons`, `Ndepths` (giving `custom_err_hidden_layers`) | `[model] HiddenLayers` |
| σ floor `1e-4 + softplus` (hard-coded) | `[model] SigmaFloor` |
| error `epochs`, `error_lr_schedule` (`PolynomialDecay(1e-3, epochs/1.5, 8e-5)`) | `[training] Epochs`, `LearningRate`, `EndLearningRate`, `DecayFraction` |
| `EarlyStopping(restore_best_weights=True)`, `SaveEveryNEpochs` (interval from the baseline TOML's `SaveInterval`) | `[training.EarlyStopping]`, `[training.Snapshots] SaveInterval` |
| `test_datasets`, `downsample_factor` (TURB-NB c9/c10, LT-NB c12/c13) | test `[data] Datasets`, `Stride` |
| `thresh_aleatoric_type/_percent/_ratio`, `thresh_conf`, `thresh_yplus` + `delta_*` (TURB-NB c42, LT-NB c54) | test `[gates] Aleatoric`, `Confidence`, `Yplus` |
| `utau_thresh`, `threshold_basis`, `y_apost_true_thresh` | `[gates] LowValue` (on the baseline) and `[metrics] LowValue` (on the truth) |
| `kt_km_threshold` (LT-NB c3) | `[switch] KtKmThreshold` |
| `model._eqwm_solve` | `[eqwm] Law` |
| `selection_mode`, `wall_normal_parameter*`, `x_transition`, `ramp_width`, `window_size` | `[plots.cf_lines]` `Selection`, `Value`, `Upstream`, `Downstream`, `XTransition`, `RampWidth`, `MovingAverage` |
| `use_y_apost_flag`, `set_yapostpred_sign_by_cftrue_flag` (TURB-NB c43/c48, LT-NB c55/c60) | dropped (LES a-posteriori mode; truth-based sign) |
| EVB `SNGP_MODEL_PATH`, `ERROR_THRESHOLD_PERCENT`, `Retau_thresh`, `thresh_err_change`, `RENAME_MAPPING` | benchmark `[models] Sngp`, `[evaluation] OutlierPercent`, `DomainFilter`/`RetauMin`, `MinErrorChangePercent`, `[display_names]` |

### 3.3 Notebook values that differ from the v2 TOMLs

Values that are equal in both (gates 10±5 % / 0.7±0.05 / 4±2 for turbulent and 15±5 % / 0.85±0.1 / 4±2 for
lam-turb, `kt_km_threshold` 0.002, `downsample_factor` 2, the test windows (TURB-NB n1/δ <= 0.15 with `downy` 0;
LT-NB the full profiles, v2 `upy` 1.0), class lists, architectures, learning rates, batch sizes, the mean-field
confidence with v = 1 (`cov=None`), C_f-line settings, tripcolor limits of LT-NB c78) are not repeated.

| Setting | Notebook | v2 | Why |
|---|---|---|---|
| Turbulent SNGP in `test_turb` | TURB-NB `..._ep5000_merged` | `..._ep4000_merged` (both test scripts) | the one CharLES runs |
| Turbulent SNGP in the benchmark | EVB `..._ep3500_merged` (11 trained classes, not in `models/`) | `..._ep4000_merged` | same |
| Error-model epochs (turb) | `epochs = 3811` | `Epochs = 10000`, best epoch restored | 3811 is the best epoch of a 10000-epoch run, not the recipe |
| Additional-data window, laminar | LT-NB c8: `upy = 1.25`, `downy = 0.01` (ignored because `upy >= 1`) | SNGP `upy 0.8`, error `upy 1.0`, both `downy 0` | match the reference models' files ([section 5](#5-data-provenance)) |
| Window, turbulent branch of LT-NB | LT-NB c8 final state: `upy = 1.0`, `downy = 0.01` (ignored) | `upy 0.15`, `downy 0` (as TURB-NB) | the turbulent-only recipe; the final LT-NB state was edited after the reference models were trained |
| Laminar additional data, error model | final LT-NB c14: the `down0p02_up0p8` files | `down0p005_up1` files | the reference "UPY1" model's statistics match these ([10.3](#103-statistics-reproduction)) |
| Low-value threshold | TURB-NB `utau_thresh 0.005` on the truth; LT-NB `threshold_basis 'y_apost_true'`, `y_apost_true_thresh 50` plus a raw-σ hack in the gate | `[gates] LowValue` 2±1 on the baseline, `[metrics] LowValue` 2±1 on the truth | no truth in the gates; one consistent rule |
| Sign of the corrected value | TURB-NB `set_yapostpred_sign_by_cftrue_flag = 1` (truth sign); LT-NB `= 0` (sign of π1) | magnitudes | no truth; magnitudes everywhere |
| Test truth | TURB-NB u_τ interpolated from the test data at each x; LT-NB C_f from `surface_data.txt` | dataset target per point | single, exact truth |
| OOD region in plots | TURB-NB C < 0.75 (threshold + delta); LT-NB C < threshold; y+ mask hard-coded 4 | C < `Confidence.Threshold`, y+ < `Yplus.Threshold` | consistent with the gates |
| EVB error-change filter | `thresh_err_change = 0.1` | `MinErrorChangePercent = 0` (off) | the filter biases the means |
| EVB percent error | `/(\|t\| + 1e-6)`, per-array outlier cuts | low-value blend 2±1, one joint outlier mask | consistent with the test scripts |
| EVB window | the turbulent checkpoint's `[0.0025, 0.15]` | same | — |

### 3.4 Class lists: notebook indices and v2 names

The notebooks listed classes by integer index into the combined class list (checkpoint `[data]` order, then the
alphabetical h5 `case_name`s of the additional datasets, renumbered after removal). v2 uses the names:

| Notebook | v2 |
|---|---|
| TURB-NB `labels_to_remove = [1,21,24,25,26,27,28]` | `RemoveClasses = ["SYN", "airfoil", "naca_0012_top4n12", "naca_4412_top1n", "naca_4412_top2n", "naca_4412_top4n", "naca_4412_top10n"]` |
| TURB-NB `group_definitions` (`merge_groups_flag = 1`) | `[classes.MergeGroups]` APG, FPG, SEP, NASA Hump FPG, NASA Hump SEP, Curved BL FPG, Curved BL APG |
| TURB-NB `posthoc_group_definitions`, `apply_posthoc_merging` | `[classes.PosthocGroups]` APG TBL, FPG TBL, Separation |
| LT-NB lam `group_definitions` `{0,5}`, `{1,3}`, `{2,4}` | `"Laminar ZPG/APG/FPG" = [FS_x, aairfoil_laminar_x]` |
| LT-NB lam `labels_to_remove = []`, `apply_posthoc_merging = False` | `RemoveClasses = []`, `PosthocGroups = {}` |
| LT-NB turb | same lists as TURB-NB |

Additional-dataset classes are named by the h5 `case_name` (e.g. `hump_FPG` → `FPG`, `aairfoil_2M` → `airfoil`), as in
the notebooks. The derived integer labels reproduce the reference metadata exactly.

### 3.5 EVB/LOG/AIAA to `tools/apriori_benchmark.py`

- Every dataset runs through `pipeline.predict_turb`, the code of `test_turb.py`, so the benchmark uses the 3-switch
  blend with the EQWM fallback instead of EVB's binary gate; laminar/transitional sets take the turbulent path.
- EVB's 75 keys with three fixes, giving 80: `gaussian_2M` (`gaussian_2M_data_ALL.h5`, x <= 0.099, no SEP region)
  replaced by the four `gaussian_2M_*` subsets pooled into the group "Gaussian bump 2M"; `Transition_transition`
  evaluated next to `Transition` and used for the training role; `hump_APG_SEP2` (an SNGP/error-model training set)
  added. EVB's 26 training keys become 30 with the same fixes.
- LOG → `--scale log|both` (`_log` figures; non-positive cases skipped with a warning).
  AIAA → `pdf_always_vs_blended` from the same pass (`CompareAlways`), instead of a second run.
- `CASE_MARKER_MAP`/`CASE_MARKER_RULES` and `RENAME_MAPPING` are verbatim (some marker rules are unreachable, as in
  EVB; `RENAME_MAPPING` keys that match no evaluated dataset, e.g. `hump`, `transition_JHU`, `spinning_Driver`, are
  ignored).
- Figures: `train_test` = EVB F1, `train_test_zoom` = F2, `sigma` = F3, `confidence` = F4. The AIAA y label is
  available as `YLabel = "Corrected Prediction Error (%)"` (default "Expert Model Error (%)" as EVB).
- Kept from EVB: the unweighted mean over cases as the headline average, PDFs pooled over all points
  (`PdfSubset = "all"`), the Re_τ domain filter (`RetauMin` 5), the 10000 % outlier threshold, the console summary
  table and the Re_τ-filter table.
- Dropped: kNN and aleatoric-model σ, binary gates, τ_w metric, bar charts, automatic dataset discovery.

---

## 4. Behaviour relative to the notebooks

### 4.1 Changed defaults (old behaviour available as an option)

| Item | Notebooks | v2 default | Option for the old behaviour | Why |
|---|---|---|---|---|
| SNGP LR decay | `PolynomialDecay(decay_steps=epochs)` counts optimiser steps, so the floor is reached early | decays over all epochs | `DecayUnit = "step"` | intended schedule |
| SNGP dropout | `dropout_rate=0.1` never applied (`training` not propagated) | applied | `DropoutActive = false` | intended regularisation |
| EQWM | parent `log_law_solve` (log law + near-wall cubic below y+ 23) | CharLES two-layer law (u+ = y+ for y+ <= 11.0623, else ln(y+)/0.41 + 5.2) | `[eqwm] Law = "parent"` | parity with CharLES |
| Aleatoric gate metric | `'percentage'` or `'ratio'` | `"percentage"` with the low-value blend | `Type = "ratio"` (set Threshold 1.0, Delta 0.5) | — |

The reference SNGPs were trained with the `"step"` / inactive-dropout behaviour.

### 4.2 Changed behaviour (no option)

- **Laminar points (lam-turb).** LT-NB c65 switched the confidence and near-wall fallbacks off on laminar points (it
  clamped C and y+ above the gates and printed "Confidence and near-wall EQWM checks disabled on lam-side points").
  v2 applies both and falls back to the laminar law τ_w = μ|u1|/y1 (no dp/ds term), i.e. π_out = sqrt(|π1|), with
  its own y+_nw = sqrt(|π1|). Same in CharLES since `b1316d9`. LT-NB did **not** fall back to the EQWM on laminar
  points (an earlier review draft claimed it did).
- **Truth is evaluation-only.** Gates, signs, fallbacks and y+ use only inference-time quantities (the notebooks
  used the truth for the low-value threshold, the y+ of the near-wall gate and, in TURB-NB, the sign).
- **Magnitudes everywhere** in predictions, errors and plots.
- **Corrected candidate** y_corr = |NN(π) + μ| (see [4.3](#43-notebook-bugs-fixed)).
- **Dataset truth** per point instead of interpolated u_τ or a C_f file.

### 4.3 Notebook bugs fixed

IDs refer to the decision log's bug list.

| ID | Notebook behaviour | v2 |
|---|---|---|
| — | always-corrected candidate `\|\|NN(π)\| + μ\|`, off by 2\|NN(π)\| where the raw output is negative, inconsistent with the error-model target ε = y − NN(π) | `\|NN(π) + μ\|` ([10.1](#101-negative-raw-baseline-outputs)) |
| B1 | unseeded splits: baseline re-split on every load, unseeded 80/20 split of the additional data (error-model validation contained baseline-training points) | `Seed`; stored split indices reused |
| B2 | test-time SNGP inputs z-scored with the session's baseline statistics | statistics from the SNGP metadata |
| B3 | post-hoc merging changed the confidence normaliser (ln 43 instead of ln 39) | confidence on the un-merged classes; post-hoc groups change labels and plots only |
| B4 | TURB-NB: corrected % error at low u_τ missing ×100 | superseded by the low-value blend |
| B5 | LT-NB: Kt/Km compared with `<=` in two places and `<` elsewhere | laminar iff `kt_km <= KtKmThreshold` (CharLES convention) |
| B6 | OOD threshold differed between notebooks; y+ mask hard-coded 4 | thresholds from `[gates]` |
| B7 | labels = argmax over all 50/30 outputs (collided with composite laminar labels) | argmax over the trained classes |
| B8 | LT-NB: test x-filter used the last dataset only | x/y filters dropped |
| B9 | "ycoord" filter used the `albert` column | dropped |
| B10 | interpolated truth took its sign from an arbitrary row per x | dataset truth, magnitudes |
| B11 | expert script with `--gating-reference baseline`: sign mismatch between truth/baseline and corrected/EQWM | magnitudes |
| B12 | LT-NB plot cells c75-c80: `NameError` (`utau_apost`), NaN from sqrt of negative τ_w, overwritten files | rewritten plotting with unique names |
| B13 | OOD relabel mutated `num_classes_train`; re-runs appended duplicate classes | no side effects |
| B14 | `train_wm_proposal.py`: `raise "device is incorrect"` (a TypeError) | `RuntimeError` |
| — | `downy` silently ignored when `upy >= 1` (parent loader) | honoured whenever > 0 |
| — | EQWM % error computed differently from the others | same low-value blend on the truth for every prediction |
| — | LT-NB needed two manual passes; one top-to-bottom run gave a hybrid result | one script, `[lam]`/`[turb]` tables |
| — | class bookkeeping by integer index (depends on checkpoint order, alphabetical extra names, renumbering after removal) | by name, validated; reproduces the reference metadata exactly |
| — | FS laminar data: unnormalised training inputs misaligned by 8400 wall rows (never used downstream) | code dropped |

### 4.4 Dropped features

| Feature | Status in the notebooks | Reason |
|---|---|---|
| sample weighting (`weight_flag`: extras ×10, low-error down-weighting) | vestigial | not used by the reference models |
| signed-log10 inputs (`X_error_logscale_flag`, `log_cols`) | vestigial, half-wired | breaks CharLES |
| alternative error targets: C_f, δ-based, τ_w-scaled (`composite_error_definition`, `rel_def_tauw`) | vestigial, half-wired | breaks CharLES |
| inner-region mode of the error model (`near_wall_threshold`/`linear_flag`, `XlogNearWall`) | vestigial | the baseline keeps its near-wall support |
| aleatoric-only model, PyTorch error model (`modified_files/error_model_pytorch`), low_ale retrain, BNN, kNN/kernel "true aleatoric" (optionB/C) | not in the final notebooks | |
| hierarchical training, sparse Sobolev regularisation | — | owner's request |
| a-priori error-model predictions as a training metric | LT-NB only, never plotted | |
| GP covariance at inference (`return_covmat`), dummy Gaussian-cloud classes, 3D dummy-grid confidence map | vestigial / removed experiments | (the GP-covariance confidence is back as `[confidence] Covariance = "gp"`) |
| x/y coordinate filters | buggy (B8, B9) | |
| LES a-posteriori mode (`use_y_apost_flag`, `data/aposteriori_*`) and the `error_apst_dummy` check | off in the finals | out of scope |
| truth-based signs and gates | live | no truth in predictions |
| dead plot code (`average_error_by_xcoord`, airfoil y(x), `plot_error_distribution_perc`) | unused | |

---

## 5. Data provenance

`data/` holds **training data only** (git-lfs; all 144 upstream LFS files plus the additions below, about 930 MB
pulled from upstream). Differences from upstream:

| File | Source | Change | Why |
|---|---|---|---|
| `FS_{ZPG,APG,FPG}_data.h5` | fork's local, uncommitted versions | replaced; about 5x more points (`FS_ZPG`: 1.94 MB → 9.32 MB) | the versions both reference baselines were trained on; FS_* are classes of the turbulent SNGP and part of the turbulent error-model data too |
| `curved_TBL_{APG,FPG}_data.h5` | fork's local versions | replaced; numerically identical, case names `curved_BL_APG`/`curved_BL_FPG` instead of `curved_BL` | the SNGP needs two distinct classes |
| `aairfoil_2M_full2Dfield_down0p02_up0p8_laminar_{ZPG,APG,FPG}.h5` | fork, untracked | added | additional data of the laminar SNGP (n1/δ 0.02-0.8, `kt_km` column) |
| `aairfoil_2M_full2Dfield_down0p005_up1_laminar_{ZPG,APG,FPG}.h5` | fork, untracked (dated 2026-04-14) | added (review decision, 2026-10-25) | additional data of the laminar error model (n1/δ 0.005-1); the reference "UPY1" model's statistics match them |
| `gaussian_2M_data_{MAPG,FPG}.h5` | **upstream** | unchanged | the fork's local copies are corrupted ([5.1](#51-corrupted-gaussian-copies-in-the-fork)) |

- `hump_*` regions are upstream files.
- Candidate generators of the A-airfoil region files in the fork: `split_aairfoil_2M_by_region_generateFull2D.py`,
  `split_dataset_by_up_region.py` (not verified which produced the shipped files).
- Not included (D2/D7): the reference C_f/geometry files `BFM_WM/data/Gray_x_vs_Cf.csv` (Gaussian-bump C_f),
  `WM_data_creation/airfoil_Tamaki/2M/surface_data.txt` (A-airfoil x/c, C_f; LT-NB's truth) and
  `BFM_WM/tmp/aairfoil-coords.txt` (airfoil geometry). v2 uses the dataset truth instead.
- `wall_model_cases.py` registers 79 keys whose files are not in `data/` (`aposteriori_*`, `aairfoil_10M*`, the other
  `aairfoil_2M_full2Dfield*` and `aairfoil_2M_{laminar,transition,turbulent}`, `transition_JHU_full2Dfield*`,
  `ph_B_full2Dfield*`); their files are in `BFM_WM/data/`.

### 5.1 Corrupted gaussian copies in the fork

`BFM_WM/data/gaussian_2M_data_MAPG.h5` and `gaussian_2M_data_FPG.h5` store network inputs equal to 2.5e-13 × π
(scaled by ν²). The turbulent reference baseline errs about 100 % on them, against a median of 2.0 % / 6.8 % on
upstream's files, and the reference SNGP and error model behave as if trained on consistent data, so **none of the
reference models was trained on the corrupted copies**. v2 first copied them (D2) and switched to upstream's on
2026-10-25 01:29 ("Data correction" block). Any artifact made with the corrupted files shows a turbulent SNGP input
mean near 955 instead of about 990 (reference 993.39) and an error mean near 2.8 instead of 0.43.

### 5.2 Lam-turb test set

`aairfoil_2M_full2Dfield_down0_up1_cfgauss_w30_tr3_xalltoall.h5` (145 MB): full 2D-field extraction of the A-airfoil
at Re = 2M (wall-normal profiles from leading to trailing edge, x/c in [0, 1.01], `kt_km` column), made with the
WM_data_creation scripts. It is untracked in the fork and exists only at
`/path/to/aairfoil_2M_full2Dfield_down0_up1_cfgauss_w30_tr3_xalltoall.h5`;
`inputfiles/test/test_lamturb.toml` reads it by that absolute path (`[data.Files]`). Before 2026-10-25 the TOML used
the relative path `../BFM_WM/data/...` (fork checked out next to v2).

---

## 6. Reference models

The 7 artifacts used by the final notebooks, byte-identical copies of `BFM_WM/models/` in
`<models location>/` (30.9 MB). `models/` is git-ignored, as in upstream
and the fork (D3). None was trained with seeds, so their splits cannot be reproduced exactly.

| Model | Role | Used by | Recipe / provenance | CharLES |
|---|---|---|---|---|
| `cleaned_up_20251208_ymax_0_15_ymin_0_0025.pth` | turbulent baseline (both variants) | TURB-NB, LT-NB turb, EVB | `train_wm_proposal.py` with `input2_1204_v1new_0.15_0.0025.toml` (no longer in the fork; `input2_0.15_0.0025.toml` matches it, now `baseline/baseline_turb.toml`); 20×6, n1/δ 0.0025-0.15, includes FS_*; W&B on | `BFM_wall_data_turb` |
| `NN_wm_CH0_..._inputs2_final_ep30000_tl0.00032144_vl0.00031670.pth` | laminar baseline | LT-NB lam | `inputfiles/input2_laminar.toml` + `downy 0.005` (now `baseline/baseline_lam.toml`); 40×4, FS_ZPG/APG/FPG, `upy 1.25` (no window); trained on GPU 1 | `BFM_wall_data_lam` |
| `sngp_model_Baseline_cleanedupNew_Add_ExpertExtFlow_ep4000_merged` | turbulent SNGP, 4000 epochs, K = 50, K_tr = 12, input mean 993.39 | LT-NB turb; default of both v2 test scripts and the benchmark | LT-NB turb pass; recipe = `sngp/sngp_turb.toml` with 4000 epochs, `DecayUnit "step"`, `DropoutActive false` | `BFM_sngp_data_turb` |
| `sngp_model_Baseline_cleanedupNew_Add_ExpertExtFlow_ep5000_merged` | turbulent SNGP, 5000 epochs | TURB-NB | TURB-NB recipe (`sngp/sngp_turb.toml`, 5000 epochs, legacy quirks) | never exported |
| `sngp_model_Baseline_selftrained_lam_Add_ExpertExtFlowUPY0p8_ep20000_merged` | laminar SNGP, 20000 epochs, K = 30, K_tr = 3 | LT-NB lam | `down0p02_up0p8` laminar files, `DownsampleBaseline 10`, `RecomputeNorm` | `BFM_sngp_data_lam` |
| `error_model_Baseline_cleanedupNew_Add_ExpertExtFlow_CustomLayers_N60xL15_wVal_ep3811` | turbulent error model, 15×60 | TURB-NB, LT-NB turb | best val_loss epoch (3811) of a 10000-epoch run (`training_history.json` has 10000 epochs; tf_keras restores the best weights at the end even without an early stop) | `BFM_wall_error_data_turb` |
| `error_model_Baseline_selftrained_lam_Add_ExpertExtFlowUPY1_CustomLayers_N60xL6_wVal_ep4000` | laminar error model, 6×60 | LT-NB lam | 4000 epochs, best epoch 3985 restored; data = `down0p005_up1` laminar files, shown by its normalisation statistics ([10.3](#103-statistics-reproduction)); file dates do not decide it (model saved 2026-04-14 16:26, `up1` files 17:42/18:08, `up0p8` files 21:35) | `BFM_wall_error_data_lam` |

- Name tags: "ExpertExtFlow" = the 12 additional external-flow datasets; "UPY0p8"/"UPY1" = the n1/δ upper limit of
  the laminar A-airfoil file set (not the `upy` key); "cleanedupNew"/"selftrained_lam" = the baseline.
- The SavedModels need Python 3.11 (cloudpickled bytecode) and tf_keras; metadata of the reference SNGPs has only
  the legacy keys (no `bfmwm_v2` block), so their post-hoc groups and display names come from the test TOMLs.
- Their `assets/` also hold the export bundles (`sngp_params.npz`, `sngp_params_meta.json`, `error_model_params.json`),
  which `tools/` reproduces (`error_model_params.json` differs only in the recorded `model_dir`).
- The v2 training TOMLs default to `*_v2` names so that the reference models are never overwritten.

---

## 7. Annotated TOMLs

Before the user docs were cleaned, every TOML carried per-key notebook pointers (e.g. `TURB-NB labels_to_remove =
[1,21,24,25,26,27,28]`, `LT-NB group_definitions for 'lam' ({0,5}, {1,3}, {2,4})`, `notebooks: downsample_factor 2`,
`EVB thresh_err_change 0.1`, `TURB-NB used ..._ep5000_merged`, `LT-NB cell 78`). They are kept in two places:

- `docs/dev_local/inputfiles_annotated/` (git-ignored, local only; same tree as `inputfiles/`);
- git history: `git show c459004:inputfiles/<path>`, e.g. `git show c459004:inputfiles/sngp/sngp_turb.toml`.

To compare with the current files: `git diff c459004 -- inputfiles/`. The important pointers are also in
[3.2](#32-notebook-flags-to-toml-keys) and [3.3](#33-notebook-values-that-differ-from-the-v2-tomls).

---

## 8. Decision record

Full log: `<decision log>` (outside the repository).
Binding blocks: "★ DECISIONS" (2026-10-24), "Late decisions", "Data correction", "Review decisions (2026-10-25)";
later blocks override earlier ones, and the "Rec." sections below them are history. Summary of the binding state:

### 8.1 Standing principles

1. **No true-target information in any prediction.** Gates, signs, fallbacks and y+ use only quantities available at
   inference time, as in CharLES. The truth only selects evaluation points and gives the errors. Checked with the
   no-truth test of [11.3](#113-checks).
2. **Magnitudes everywhere** (predictions, errors, plots).
3. **Training data self-contained in `data/`** via git-lfs; models may stay git-ignored (training scripts regenerate them).
4. **Parent functionality preserved**: upstream files changed minimally, each change marked and documented; parent
   tools keep working.
5. **Changed parent defaults are documented** in the files, `src/README.md` and the README.
6. **CharLES compatibility**: saved-model formats and metadata keys read by the exporters stay unchanged;
   architecture keys warn when they leave CharLES's hard-coded values.
7. **Quirks of the reference models stay reproducible** as options when a default changes.

### 8.2 Decisions by topic

**Repository and parent code**
- D1: remote `upstream`, branch `v2` from `7794771`, logical commits, nothing pushed.
- D4: unused parent files (`apost/`, `table_utils/`, `test_conversion_cuda/`, `fds.py`, `utils_imb.py`, examples,
  `test_model_*`, ...) stay untouched, for painless upstream merges.
- D5: port all fork changes to `src/` (see [1.4](#14-what-came-from-the-notebook-fork)) plus the `RuntimeError` fix.

**Scripts and CLI**
- S1: one training script per variant (`train_turb.py`, `train_lamturb.py`) with `--sngp` and/or `--error`; test
  scripts `test_turb.py`, `test_lamturb.py`; baseline stays `train_wm_proposal.py`.
- S3: test configuration by TOML plus `--gpu`, `--output-dir`, `--max-samples`.
- S4: paths are cwd-relative as in the parent (run from the repository root; TOMLs resolved under `./inputfiles/`).
  The proposed `chdir` was not adopted.
- Preflight (review fixes): TOML kind sniffed and wrong-kind TOMLs reported as such; every dataset file and model
  checked before data or models load; the benchmark `--dry-run` applies the same model rule (`cli.missing_model`).
- `--smoke`: 2 epochs, capped samples, outputs under `results/smoke/`.

**TOMLs**
- D6: `inputfiles/{baseline,sngp,error,test,benchmark}/`; `baseline_turb.toml` = the turbulent checkpoint's config,
  `baseline_lam.toml` = `input2_laminar.toml` + `downy 0.005`; `input2_example.toml` stays at the top level for the
  parent's usage. Only `GpuNum`, `UseWandb` and `Seed` differ from the embedded configs.
- S2: lam-turb TOMLs hold `[lam]` and `[turb]` tables; a shared `[general]` is inherited.
- S6: explicit `ModelName` **required** (per branch); `*_v2` defaults; existing models replaced only with `Overwrite`.
- S7: `Seed` in every TOML; full metadata (`bfmwm_v2` block: config, datasets, groups, checkpoint, seed, versions) and
  `split_indices.npz` in `assets/`; legacy keys unchanged.
- S8: class bookkeeping by name (`RemoveClasses`, `MergeGroups`, `PosthocGroups`, `DisplayNames`); unknown names,
  duplicates and collisions are errors.
- Every key is commented; unknown keys are errors.

**Data** — see [section 5](#5-data-provenance): D2 (git-lfs, `--local`; training data only; test set by path),
"Data correction" (upstream's gaussian MAPG/FPG; the other 5 replacements stay), review decision (laminar error-model
data = `down0p005_up1`; laminar SNGP keeps `up0p8`), absolute cluster path for the lam-turb test set.

**SNGP**
- Keep: additional datasets as extra classes; remove/merge by name; `DownsampleBaseline`, `RecomputeNorm` as general
  options; display names and post-hoc groups (labels/plots only); OOD relabel in plots; confusion-matrix and history
  plots.
- Add: architecture keys (CharLES defaults 6/128/0.9/1024/1.0, warning otherwise); `ClassWeights`; optional
  GP-covariance confidence (default `"none"` = CharLES); pinned layer names.
- Change: per-epoch LR decay and active dropout by default (G12/G13 overridden), old behaviour as options.
- Defaults: turbulent-only 5000 epochs; lam-turb turb branch 4000; laminar 20000 epochs, 30 classes, downsample 10,
  recompute norm, `upy 0.8`, `downy 0`.

**Error model**
- Keep: best-epoch restore (optional patience), snapshots (own `SaveInterval` key), history files.
- Drop: a-priori test metrics, vestigial features, side models, old gate variants, hierarchical training, sparse
  regularisation.
- Defaults: turb 10000 epochs (best kept), 15×60, `upy 0.15`/`downy 0`; lam 4000 epochs, 6×60, `down0p005_up1` files,
  `upy 1.0`/`downy 0`.

**Gates, blend, EQWM, laminar law**
- y+ of the near-wall switch from the EQWM u_τ (turbulent) or the laminar law (laminar).
- σ% gate metric 100σ/|π_out,b| blended with 100σ/sqrt(|π1|), ramp on |π_out,b| with threshold 2, Δ 1;
  `"ratio"` (σ/|μ|) kept as an option.
- Laminar points: both fallbacks use the laminar law without dp/ds; Kt/Km laminar if `kt_km <= 0.002`.
- EQWM: CharLES two-layer law by default; `Law = "parent"` reproduces `log_law_solve`; both vectorised.
- Corrected candidate: |raw + μ| only; the notebook form is not kept as an option (measured effect in [10.1](#101-negative-raw-baseline-outputs)).
- Input frame: no `InputFrame` option; to-do (retrain in the u2-positive frame), see [9.3](#93-remaining-difference-the-input-sign-frame).

**Testing and evaluation**
- Truth = dataset targets; % errors use the same low-value blend (2±1 on |π_out,true|) for every prediction incl. EQWM.
- Turbulent SNGP ep4000 in both test scripts; ep5000 kept in `models/` but unused.
- Figures: spatial maps, C_f lines, PDFs, smoothed tripcolor maps, each switchable; `summary.json`; optional
  `arrays.npz`.
- Output naming kept from the notebooks: "corrected" in test outputs = the blended y_final; "always corrected" = y_corr.

**Benchmark** ("Late decisions"): own self-contained TOML; EVB keys with the three fixes; two colour roles; blended and
always-corrected in one pass; joint 10000 % outlier cut; error-change filter off; `DomainFilter "retau"` (|u_τδ/ν| > 5)
or `"piout"` (|π_out,true| >= 2); `cases.csv`, `overall.json`, `points.npz`; three overall averages (unweighted over
cases, point-weighted, per role); `[groups]`; y-label option; `--replot` without torch/TF.

**Exports** (D8): exporters copied to `tools/` with unchanged CLIs, defaults and formats; saved-model formats kept
(SNGP variable names, 1024 features, metadata keys `num_classes`, `num_classes_train`, `mean_X_train`, `std_X_train`;
`error_norm_stats.json` with scalar `y_mean`/`y_std`; plain Dense stack); outputs to `results/export/` (not into model
directories). The `#define` suffix blocks and `static const int` sizes of the CharLES copies are still edited by hand.

**Environment** (D9): Python 3.11 (cloudpickled bytecode), torch 2.7.1, TF 2.19 + tf_keras, TFP 0.25,
tf-models-official 2.19.1, pinned in `requirements.txt`.

**CharLES** — see [section 9](#9-charles-bfmv2-parity).

### 8.3 Recommendations that were overridden

| Proposal (log section 1-6) | Final |
|---|---|
| D2: data via a `BFM_WM_DATA` env var pointing at the fork | copied into `data/` with git-lfs |
| D3: track the models in git | git-ignored |
| D6: drop `input2_example.toml` | kept |
| D7: bundle reference C_f/geometry files in `data/reference/` | not included |
| S4: scripts `chdir` to the repo root | cwd-relative, run from the root |
| S6: optional auto-generated `ModelName` | required |
| S9/G12/G13: keep the SNGP quirks as defaults | new defaults, quirks as options |
| T1/T2: truth from interpolation or C_f file; gating reference truth by default | dataset truth only; no truth in gates |
| T3: keep the LES a-posteriori mode | dropped |
| T4: default SNGP as in each notebook | ep4000 in both |
| T8: keep the lam-side guard as an option | removed; laminar-law fallbacks |
| E4: keep a-priori error-model metrics | dropped |
| Table "lam error upy 1.0 (UPY1)" with `up0p8` files | `down0p005_up1` files (review decision) |
| D2: replace 7 files | 5 (gaussian MAPG/FPG reverted) |

---

## 9. CharLES BFMv2 parity

### 9.1 What matches

Checked against `src/charles/cti_solver_ig_bcs_kernels.hpp` and `wm/*.hpp`:
- ramps (`smoothBlendAbove/BelowThreshold`, incl. the Δ <= 0 hard switch), nesting of the three switches, separate
  confidence and near-wall fallbacks;
- EQWM two-layer law (y* 11.0623, κ 0.41, B 5.2): agrees with a float32 emulation of CharLES's relaxed Newton to
  4.3e-7 (and with brentq to 4e-16);
- confidence: mean-field factor π/8 with v = 1 (CharLES hard-codes 1.18012667), normaliser ln(K − K_tr + 1) with the
  un-merged K (50/12 turb, 30/3 lam), no post-hoc merging;
- error-model de-normalisation incl. σ = (1e-4 + softplus)·y_std; aleatoric percent with the low-value blend;
- laminar faces: laminar law for both fallbacks, `kt_km <= KT_KM_THRESHOLD`;
- corrected candidate |u_b + ε| and the magnitude of the baseline in the inner blend.

### 9.2 CharLES commits made for parity (2026-10-24/25)

| Commit | Change |
|---|---|
| `b1316d9` | Laminar faces: the near-wall law loses the dp/ds term (τ_w = μ\|u1\|/y1, was μ\|u1\|/y1 − y1·sign(u1)·dp/ds/2 clipped >= 0) and the low-confidence fallback becomes the same law (was the uncorrected laminar baseline); laminar faces never use the EQWM. New keywords `ALEATORIC_LOWVALUE_CUTOFF` (2.0) / `ALEATORIC_LOWVALUE_DELTA` (1.0): the aleatoric percent blends 100\|σ\|/\|u_τ,b\| with 100\|σ\|/sqrt(ν\|u1\|/y1), weighted by the baseline y+. |
| `f14da85` | Inner aleatoric blend uses `utau_baseline_abs`: the signed raw baseline partly cancelled the corrected magnitude where the raw output is negative. |

Executables: the pre-`b1316d9` GPU build is `bin/charles_ig_gpu.04e26c5.exe` (the L4 BFMv2 case links to it); the
pre-`f14da85` build is `bin/charles_ig_gpu.b1316d9.exe`. The four L3 BFMv2 cases use `src/charles` and pick up the
changes on their next run. Nothing was run after these commits.

### 9.3 Remaining difference: the input sign frame

CharLES BFMv2 (GFIX_v3) builds the network inputs in a local u2-positive frame: sign(π1) = sign(u1·u2),
π3 = |u2| n1/ν, dp/ds projected on the u2 direction. The datasets (and therefore training and every a-priori test) use
the global-x frame. For a point with u2 < 0 CharLES feeds (|π1|, −π2, |π3|) where v2 feeds (π1, π2, π3), and the
networks are not reflection-symmetric. Measured with the reference models (n1/δ <= 0.15, stride 2, points with
π3 < 0):

| Dataset | Points with u2 < 0 | Baseline mean \|e\| dataset → CharLES frame | Always-corrected | Median C |
|---|---|---|---|---|
| gaussian_2M_SEP | 72 % (9086/12596) | 7.5 % → 17.8 % (median 4.2 → 14.2) | 5.0 % → 22.6 % | 1.000 → 0.992 |
| hump_APG_SEP2 | 92 % (131/143) | 11.0 % → 23.2 % | 11.7 % → 40.7 % | 0.993 → 0.927 |

Plan (owner, README To do): regenerate the training data in the u2-positive frame (WM_data_creation) and retrain all
models. Rationale: a deployed wall model does not know the global flow direction, so reversal cannot be defined
relative to it; u2 is the local reference that decides whether the flow is locally reversed and which way the pressure
gradient points, and it gives the symmetry that a prediction (a magnitude) must not depend on whether the local flow
runs forward or backward when u1 and u2 point the same way. The sign of τ_w is assigned afterwards from the global
direction of u1. An inference-only `InputFrame` option was considered and rejected (it only makes the current models
worse on reversed flow). Notes for an implementation: keep |π1| frame-invariant; only InputScaling 2 is covered; with
`near_wall_threshold > 0` the inputs are sign·log1p(|x|); CharLES deadbands and time filtering are not reproduced.

---

## 10. Analyses worth keeping

### 10.1 Negative raw baseline outputs

The baseline is trained on |u_τ| but its linear output layer can return small negative values near the wall and
around separation. Reference models, benchmark window and filters (turbulent) and `test_lamturb.toml` (lam-turb):

| Set | Points | raw < 0 | Most negative raw (π_out) |
|---|---|---|---|
| benchmark, 80 datasets | 1.48 M | 417 (0.03 %) in 16 datasets | mostly −0.04 to −3.3; −5.3 (gaussian_1M); −51.3 (2 points, hump_FPG_SEP) |
| lam-turb test set, all | 278 740 | 1814 (0.65 %) | −1954 |
| — laminar points | 18 846 | 246 (1.3 %) | −0.42 |
| — turbulent points | 259 894 | 1568 (0.6 %) | −1954 |

Effect of the corrected-candidate form on those points: on the laminar points the mean always-corrected error is
12 % with |raw + μ| against 95 % with the notebooks' ||raw| + μ|. On turbulent benchmark points the comparison goes
either way (e.g. gaussian_1M 45 % → 65 %, bub_B 53 % → 44 %), because those points are rare and hard. The blended
error changes by at most 0.06 percentage points per dataset (largest: hump_FPG_SEP). Decision: |raw + μ|, the form
consistent with the error-model target ε = y − raw and with CharLES.

### 10.2 Input frame

See [9.3](#93-remaining-difference-the-input-sign-frame). Fraction of negative inputs (n1/δ <= 0.15, full data):
gaussian_2M_SEP π1 < 0 on 99.3 %, π3 < 0 on 72.2 % of 25 191 points; hump_APG_SEP2 π1 < 0 on 100 %, π3 < 0 on 92.3 %
of 285; gaussian_2M_APG 0 %. On the lam-turb test set 5.8 % of the turbulent points have u2 < 0.

### 10.3 Statistics reproduction

v2's data pipeline against the stored statistics of the reference models (current `data/`):

| Model | v2 | Reference |
|---|---|---|
| turbulent SNGP input mean (`mean_X_train`, baseline train part) | [990.2, 8.29, 1261.9] | [993.4, 8.31, 1265.3] |
| turbulent error model x_mean[0] / error mean / error std (v2 training, seeded split) | 981.08 / 0.442 / 34.08 | 982.57 / 0.442 / 39.32 |
| laminar error model with `down0p005_up1` (v2 smoke run) y_mean / y_std | −1.059 / 10.81 | −1.062 / 10.82 |
| laminar error model with `down0p02_up0p8` (the rejected recipe) y_mean / y_std | −0.508 / 5.89 | −1.062 / 10.82 |

Differences are within the sampling noise of the splits. The turbulent error std depends strongly on the split, because
the error targets are heavy-tailed (in v2's split: train std 34.3, valid std 54.8). Name-based class handling reproduces the reference
`flow_to_label_combined`, `labels_to_remove` and `flow_to_label_filt` of both SNGPs exactly. All exporters run on
v2-trained (smoke) models; the SNGP bundle has the same keys as the reference one, and the sizes match CharLES
(K/K_tr 50/12 and 30/3, 3→128×6, 1024 features; error models 16/7 layers of width 60 in the CharLES count).

### 10.4 Other checks (2026-10-25 reviews)

- GP-covariance confidence on the ep4000 SNGP: logits identical, variance independent of batch size (7.8e-8),
  about 0.3 s per 1024 points on 2 CPU threads.
- `parent` EQWM law matches `src/baseline_models.log_law_solve` to 6e-14.
- Smoke models trained before the data correction were stale (turbulent SNGP mean 955, error mean 2.8); regenerated
  13:26 on 2026-10-25.
- Latent issue fixed in `c459004`: training and inference used two helpers for the raw baseline output
  (LogTransform handled only at inference); now one helper, `data.baseline_output`.

The review reports and check scripts lived in a temporary session directory and are not preserved; the results
above are the part worth keeping.

---

## 11. Development notes

### 11.1 Environment on the cluster

- Shared venv: `<path-to-venv>` (8.5 GB). Its `bin/activate` unsets
  `PYTHONPATH`, because the auto-loaded modules put Python 3.10 packages there that shadow the venv's (numpy import
  error). Build any new venv with `env -u PYTHONPATH`.
- Base interpreter: the shared cluster Python 3.11.6 (`module load <python3.11 module>`), which exists on
  the login and the GPU nodes. Until 2026-10-25 it was `/usr/bin/python3.11`, which exists on the login nodes only
  (the GPU nodes have `/usr/bin/python3.9`). The first five GPU jobs therefore failed with `No module named
  numpy/torch`, and the trainings of [section 12](#12-reproduction-of-the-reference-models-with-v2) ran the venv's
  packages with the cluster Python through `PYTHONPATH`. The venv was then re-pointed to the cluster Python with
  `venv --upgrade --without-pip` (packages and `activate` untouched; the changed files are backed up in
  `<venv backup>/`). Plain `source .../activate` was then checked
  in a P100 job, and CPU results were bit-identical to those under the old interpreter.
- If that cluster build ever disappears, re-point the venv to another CPython 3.11 (the compiled packages are cp311):
  `rm <venv>/bin/python3.11 && <python3.11> -m venv --upgrade --without-pip <venv>`.
- Login node: limit threads (`OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1`); keep checks
  light. No Slurm submission without the owner's approval.
- git-lfs 3.8.1 is in `~/.local/bin`, enabled with `--local` in this repository only.

### 11.2 Outputs

| Path | Content |
|---|---|
| `models/<ModelName>/` | trained SavedModels + `assets/` |
| `models/checkpoints/<ModelName>/ckpt_XXXX/` | error-model snapshots |
| `results/training/<ModelName>/` | confusion matrices, class counts, histories |
| `results/smoke/{models,training}/` | `--smoke` outputs (Overwrite allowed) |
| `results/test_turb_gaussian_2M/`, `results/test_lamturb/` | test outputs |
| `results/apriori_benchmark/` | benchmark |
| `results/export/` | exporter outputs |

`models/` and `results/` are git-ignored.

### 11.3 Checks

From the repository root, with the venv active and `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2`:

```bash
# CLIs
for s in train_turb.py train_lamturb.py test_turb.py test_lamturb.py tools/apriori_benchmark.py; do python $s --help; done
# config loading and preflight (no torch/TF)
python tools/apriori_benchmark.py benchmark/apriori_benchmark.toml --dry-run
# training smoke tests (2 epochs, capped samples, results/smoke/)
python train_turb.py --sngp sngp/sngp_turb.toml --error error/error_turb.toml --gpu -1 --smoke
python train_lamturb.py --sngp sngp/sngp_lamturb.toml --error error/error_lamturb.toml --gpu -1 --smoke
# test scripts on a subsample
python test_turb.py test/test_turb.toml --gpu -1 --max-samples 2000 --output-dir results/smoke/test_turb
python test_lamturb.py test/test_lamturb.toml --gpu -1 --max-samples 2000 --output-dir results/smoke/test_lamturb
# benchmark subset and a torch/TF-free replot
python tools/apriori_benchmark.py benchmark/apriori_benchmark.toml --gpu -1 --datasets CH TBL_5 "Gaussian bump 2M" \
    --max-samples 3000 --output-dir results/smoke/benchmark
python tools/apriori_benchmark.py --replot results/smoke/benchmark --scale log
```

Smoke-run sanity values ([10.3](#103-statistics-reproduction)): turbulent SNGP `mean_X_train` near [990, 8.3, 1262],
turbulent error `y_mean` near 0.43, laminar error `y_mean`/`y_std` near −1.06/10.8.

**No-truth check.** Predictions must be bit-identical when the truth arrays are replaced by noise; only the
evaluation fields (`y_true`, `e_*_pct`) may change. Passed for both variants on 2026-10-25 (21 turbulent and 23
lam-turb prediction fields). Script used (run from the root as `python no_truth_check.py turb|lamturb [N]`):

```python
import sys, dataclasses
import numpy as np
import src_err.utils as utils  # TF environment first
from src_err.config import load_test_config
from src_err.data import load_datasets
from src_err.pipeline import load_branch_models, predict_lamturb, predict_turb

variant, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4000
cfg = load_test_config(f"test/test_{variant}.toml", variant)
device = utils.configure_devices(-1)
d = cfg["data"]
if variant == "turb":
    turb = load_branch_models(cfg["models"], cfg["classes"], device, branch="turb")
else:
    turb = load_branch_models(cfg["turb"]["models"], cfg["turb"]["classes"], device, branch="turb")
    lam = load_branch_models(cfg["lam"]["models"], cfg["lam"]["classes"], device, branch="lam")
arrays = load_datasets(d["Datasets"], turb.baseline.config, d["upy"], d["downy"],
                       files_override=d["Files"], stride=d["Stride"], max_samples=n)
rng = np.random.default_rng(1234)
U2 = np.array(arrays.U, copy=True)
U2[:, 3] = rng.uniform(1e-3, 1e3, size=len(U2))                      # dataset u_tau
fake = dataclasses.replace(arrays, y=rng.uniform(1e-3, 1e4, size=len(arrays.y)).astype(arrays.y.dtype), U=U2)
run = (lambda a: predict_turb(turb, a, cfg, verbose=False)) if variant == "turb" else \
      (lambda a: predict_lamturb(lam, turb, a, cfg, verbose=False))
o1, o2 = run(arrays), run(fake)
evals = {k for k in o1 if k == "y_true" or (k.startswith("e_") and k.endswith("_pct"))}
bad = [k for k in o1 if k not in evals and np.asarray(o1[k]).tobytes() != np.asarray(o2[k]).tobytes()]
print("FAIL" if bad else "PASS", bad)
sys.exit(1 if bad else 0)
```

### 11.4 Conventions

- Upstream files: every change carries a `# BFM_WM_v2:` comment and is listed in `src/README.md` and the README
  ("Changes relative to upstream"); keep it that way so upstream merges stay reviewable.
- New code goes into `src_err/` (library), the top-level scripts (thin CLIs) and `tools/`. Import `src_err.utils`
  before anything that imports TensorFlow (it sets `TF_USE_LEGACY_KERAS=1`); keep config loading and `--replot`
  free of torch/TF imports.
- Notebook pointers belong in code comments and this file, never in the user docs (README files, TOML comments).
- Every TOML key has a schema entry in `src_err/config.py` and a comment in the shipped TOML; values shown in the test
  and benchmark TOMLs equal the schema defaults except where the header says otherwise.
- Docstrings: summary line, then `Args:`/`Returns:`.
- Keep the legacy model formats (see the exports decision in [8.2](#82-decisions-by-topic)); CharLES and the
  exporters depend on them.

### 11.5 Open items

- Input sign frame ([9.3](#93-remaining-difference-the-input-sign-frame)).
- The lam-turb test set and the reference models are available only on the cluster.
- Output naming: "corrected" in test outputs means the blended prediction (kept from the notebooks); a rename to
  "blended" would need the alias kept for old TOMLs.
- The CharLES copies of the exported data files still need the `#define` suffix edits by hand.

---

## 12. Reproduction of the reference models with v2

### 12.1 Trainings

The default training TOMLs at `c459004`, trained on P100 GPUs (normal QOS) on 2026-10-25 with the cluster job recipe
of the README. The `[turb]` branch of `error_lamturb.toml` has the same recipe and seed as `error_turb.toml` and was not
trained separately. Outputs: `models/*_v2` on the cluster (git-ignored).

| v2 model | Recipe | Time | Result (reference in parentheses) |
|---|---|---|---|
| `sngp_model_turb_v2` | `sngp_turb.toml`, 5000 epochs | 60 min | val acc final 0.730 (ep5000: 0.754), best 0.805 (0.795); val loss 0.665 (0.663) |
| `sngp_model_lamturb_turb_v2` | `sngp_lamturb.toml` `[turb]`, 4000 epochs | 49 min | val acc final 0.760 (ep4000: 0.694), best 0.784 (0.746); val loss 0.653 (0.727) |
| `sngp_model_lam_v2` | `sngp_lamturb.toml` `[lam]`, 20000 epochs | 34 min | val acc final 0.977 (0.976); val loss 0.053 (0.051) |
| `error_model_turb_v2` | `error_turb.toml`, 10000 epochs | 41 min | best val NLL −3.787 at epoch 4405 (−3.985 at 3811); y_mean 0.442 (0.442), y_std 34.08 (39.32) |
| `error_model_lam_v2` | `error_lamturb.toml` `[lam]`, 4000 epochs | 8 min | best val NLL −4.736 at epoch 3998 (−4.762 at 3985); y_mean −1.059 (−1.062), y_std 10.81 (10.82) |

- All three SNGPs reproduce the reference class maps exactly. The v2 SNGPs use the per-epoch LR decay and active
  dropout ([4.1](#41-changed-defaults-old-behaviour-available-as-an-option)); the reference ones used neither, which
  explains the higher turbulent validation accuracy of v2.
- The NLL values are on z-scored targets, so they are not directly comparable when y_std differs (heavy-tailed
  turbulent error targets, see [10.3](#103-statistics-reproduction)).

### 12.2 A-priori comparison

Both test scripts on their full test sets and the full benchmark, once with the reference models and once with the v2
models (GPU, 20 min). The scratch files are in `results/compare/` on the cluster (`run_compare.sh`, the TOML copies,
`compare_report.py`). Mean |e| in %; for the A-airfoil the medians are in parentheses.

| Test | Models (SNGP + error model) | Baseline | Blended | Always-corrected | C mean | Fully corrected |
|---|---|---|---|---|---|---|
| `test_turb` (Gaussian bump 2M, 41 616 points) | reference ep4000 + ep3811 | 6.84 | 2.50 | 3.42 | 0.977 | 75.6 % |
| | v2 `lamturb_turb` (4000) + `error_turb_v2` | 6.84 | 2.41 | 3.94 | 0.996 | 71.1 % |
| | reference ep5000 + ep3811 | 6.84 | 2.24 | 3.42 | 0.977 | 76.0 % |
| | v2 `turb` (5000) + `error_turb_v2` | 6.84 | 2.35 | 3.94 | 0.998 | 71.1 % |
| `test_lamturb` (A-airfoil 2M, 278 740 points) | reference | 43.14 (6.00) | 43.51 (4.56) | 31.16 (5.00) | 0.790 | 45.1 % |
| | v2 | 43.14 (6.00) | 42.95 (5.05) | 38.44 (5.01) | 0.814 | 55.2 % |

| Benchmark (77 cases, 1.48 M points) | Baseline | Blended | Always-corrected | C mean |
|---|---|---|---|---|
| reference, unweighted over cases | 9.62 | 8.81 | 9.00 | 0.984 |
| v2, unweighted over cases | 9.62 | 8.60 | 9.12 | 0.990 |
| reference, point-weighted | 6.35 | 4.05 | 3.25 | |
| v2, point-weighted | 6.35 | 4.07 | 3.40 | |
| reference, training cases (27) | 7.38 | 6.35 | 4.17 | |
| v2, training cases (27) | 7.38 | 5.76 | 4.15 | |
| reference, untrained cases (50) | 10.82 | 10.14 | 11.61 | |
| v2, untrained cases (50) | 10.82 | 10.13 | 11.80 | |

Conclusion: the v2 pipeline reproduces the reference models within training noise. Blended errors are equal or slightly
lower, always-corrected errors slightly higher (the turbulent error model; on the A-airfoil the median is unchanged and
the mean is driven by a few large errors), and the confidence is higher. The test TOMLs and CharLES keep the reference
models; the v2 models are available for comparison.
