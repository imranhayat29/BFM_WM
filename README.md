# BFM_WM_v2

Building-block flow model (BFM) wall model with an **error model** (aleatoric uncertainty and a correction
of the baseline) and an **SNGP confidence model**, combined in a blended "expert" prediction.

This repository is a fork of [yuenongling/BFM_WM](https://github.com/yuenongling/BFM_WM) (git remote
`upstream`). The baseline wall model, its data handling and its training/testing scripts are upstream's
(with the changes listed in [Changes relative to upstream](#changes-relative-to-upstream)). Everything
else, the error model, the SNGP confidence model, the blended correction and their training and test
scripts, replaces the Jupyter notebooks of the `BFM_WM` error-model fork with scripts and TOML configs.
The formulas match those of the CharLES BFMv2 solver. One difference remains, the sign frame of the network
inputs on reversed-flow points (see [To do](#to-do)).

Two variants are kept separate:

| Variant | Branches | Training | Testing |
|---|---|---|---|
| **Turbulent-only** | one baseline, SNGP and error model | `train_turb.py` | `test_turb.py` |
| **Laminar-turbulent** | a laminar and a turbulent branch, hard-switched by the $k_t/k_m$ sensor | `train_lamturb.py` | `test_lamturb.py` |

## Repository layout

```
train_wm_proposal.py     baseline wall-model training (upstream)
test_model.py, ...       baseline testing tools (upstream)
train_turb.py            SNGP and/or error-model training, turbulent-only
train_lamturb.py         SNGP and/or error-model training, laminar + turbulent branches
test_turb.py             a-priori test of the blended prediction, turbulent-only
test_lamturb.py          a-priori test of the blended prediction, laminar-turbulent
inputfiles/
  input2_example.toml    commented example baseline config in upstream's format (see Workflow, step 1)
  baseline/              baseline configs of the shipped checkpoints (baseline_turb, baseline_lam)
  sngp/                  SNGP training configs (sngp_turb, sngp_lamturb)
  error/                 error-model training configs (error_turb, error_lamturb)
  test/                  test configs (test_turb, test_lamturb)
  benchmark/             multi-dataset a-priori benchmark config
src/                     baseline package (upstream)
src_err/                 error model, SNGP, blended correction, data, metrics, plots
tools/                   CharLES export scripts and the a-priori benchmark (see tools/README.md)
wall_model_cases.py      dataset registry (upstream + BFM_WM additions)
data/                    HDF5 datasets (git-lfs)
models/                  trained models (git-ignored)
```

`src_err/` modules:

| Module | Content |
|---|---|
| `config.py` | Reads and validates the SNGP, error-model, test and benchmark TOMLs (unknown keys are errors) |
| `data.py` | Baseline checkpoint and train/valid split, dataset loading with the n1/δ window, additional-data split, $k_t/k_m$ |
| `classes.py` | SNGP class bookkeeping by name: remove, merge, post-hoc groups, display names |
| `sngp.py` | SNGP model, training, saving/loading (with metadata), batched prediction, confidence |
| `error_model.py` | Error model build/training/saving/loading/prediction, normalisation statistics |
| `gating.py` | EQWM and laminar law, aleatoric metric, ramps, the 3-switch blend |
| `lamturb.py` | $k_t/k_m$ switch and composite class labels |
| `pipeline.py` | The shared inference pipeline of the test scripts and the benchmark |
| `evaluation.py` | Percent errors and summary statistics |
| `training.py` | Orchestration of the training scripts |
| `reporting.py`, `plotting.py`, `selection.py` | Figures, `summary.json`, station selection for line plots |
| `cli.py` | Command-line helpers shared by the training and test scripts, including the check that every dataset and model file exists |
| `utils.py` | Seeds, devices, JSON, hashing and path helpers |

## Setup

- Python **3.11** is required: the saved error models contain cloudpickled Python 3.11 bytecode, and the
  TensorFlow stack is pinned for legacy-Keras SavedModels.
  ```bash
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```
  A venv only links to its base interpreter. To run batch jobs, build it from a Python 3.11 that also
  exists on the compute nodes (e.g. one loaded with `module load`).
- On the cluster, the environment already exists: `source <path-to-venv>/bin/activate`.
  Its base interpreter `/usr/bin/python3.11` exists on the login nodes only, so this works there but not in
  GPU jobs (see [Running on a GPU node](#running-on-a-gpu-node-slurm)).
- Run all scripts from the repository root. Paths are relative to it: `./inputfiles`, `./models`, `./data`,
  `./results`.
- `TF_USE_LEGACY_KERAS=1` is set automatically by `src_err` before TensorFlow is imported.
- On a shared login node, limit the threads, because torch and TensorFlow otherwise see every core:
  `export OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1`.
- The data are stored with git-lfs: `git lfs install --local && git lfs pull`.

## Data

Data are stored in `data/` directories. They cover data summarized in [Table](https://docs.google.com/spreadsheets/d/18hcHtwxN7-uUAkYSo2UL1Tny2SpPDvQiWsKEqjHOlh0/edit?usp=sharing) and the data generation repository [Repo](https://github.com/yuenongling/WM_data_creation?tab=readme-ov-file).

`data/` holds the **training data** of all shipped models. Differences from upstream's data:

| File | Change | Why |
|---|---|---|
| `FS_ZPG_data.h5`, `FS_APG_data.h5`, `FS_FPG_data.h5` | replaced (about 5x more points) | Falkner-Skan data the laminar baseline was trained on |
| `curved_TBL_APG_data.h5`, `curved_TBL_FPG_data.h5` | replaced (same numbers, case names `curved_BL_APG`/`curved_BL_FPG` instead of `curved_BL`) | the SNGP needs the two classes to be distinct |
| `aairfoil_2M_full2Dfield_down0p02_up0p8_laminar_{ZPG,APG,FPG}.h5` | added | additional training data of the laminar SNGP (n1/δ 0.02-0.8, with a `kt_km` column) |
| `aairfoil_2M_full2Dfield_down0p005_up1_laminar_{ZPG,APG,FPG}.h5` | added | additional training data of the laminar error model (n1/δ 0.005-1, with a `kt_km` column) |

- The lam-turb test set `aairfoil_2M_full2Dfield_down0_up1_cfgauss_w30_tr3_xalltoall.h5` is **not shipped**.
  It is a full 2D-field extraction of the A-airfoil at Re = 2M, with the `kt_km` column. `test/test_lamturb.toml`
  reads it by path (`[data.Files]`, relative to the repository root) from the BFM_WM fork checked out next to
  this repository, `../BFM_WM/data/`. If the file lives elsewhere, edit that path (or place the file in `data/`
  and remove the entry).
- `wall_model_cases.py` registers every case of the BFM_WM fork. Some keys point to files that are not shipped
  (the comment at the top of the file lists them).
- In the BFM_WM fork, the local copies of `gaussian_2M_data_MAPG.h5` and `gaussian_2M_data_FPG.h5` are
  corrupted: their stored network inputs are scaled by ν². This repository keeps upstream's correct versions.
  None of the shipped models was trained on the corrupted copies.

## Workflow

### 1. Baseline wall model (upstream)

```bash
python train_wm_proposal.py baseline/baseline_turb.toml    # turbulent baseline (also the turbulent branch)
python train_wm_proposal.py baseline/baseline_lam.toml     # laminar baseline
python train_wm_proposal.py input2_example.toml            # commented example config
```

- The config path is relative to `./inputfiles/`.
- `input2_example.toml` is a commented baseline config in upstream's format (a multi-dataset InputScaling-2
  network). It logs to Weights & Biases (`UseWandb = true`); set it to `false` to run without.
- `baseline_turb.toml` and `baseline_lam.toml` are the exact configurations embedded in the two shipped
  checkpoints. Only `GpuNum`, `UseWandb` and the new `Seed` differ.
- New checkpoints store their train/valid split indices. The SNGP and error-model training reuses them, so
  both see the baseline's actual train/valid split. Older checkpoints get a seeded re-split.
- `train_wm_proposal.py` requires a GPU (`GpuNum >= 0`).

### 2. SNGP confidence model and error model

```bash
# turbulent-only
python train_turb.py --sngp sngp/sngp_turb.toml --error error/error_turb.toml [--gpu N]
# laminar-turbulent (both branches, or one with --branch lam|turb)
python train_lamturb.py --sngp sngp/sngp_lamturb.toml --error error/error_lamturb.toml [--branch lam] [--gpu N]
# login-node smoke test: 2 epochs, capped samples, saved under results/smoke/
python train_turb.py --sngp sngp/sngp_turb.toml --error error/error_turb.toml --gpu -1 --smoke
```

- Either `--sngp` or `--error` may be given alone.
- Before any data are loaded, the TOMLs are validated (a TOML of the wrong kind, e.g. an error-model TOML
  given with `--sngp`, is reported as such) and every `AddDatasets` file and baseline checkpoint of all
  selected branches is looked up, so a typo fails within seconds, also with `--smoke`.
- With both, the baseline checkpoint must be the same. The additional datasets and their 80/20 split are
  shared when their `[data]` settings and `Seed` agree.
- Models are saved to `<SaveDir>/<ModelName>` (default `./models`). An explicit `ModelName` is required, and
  existing models are only replaced with `Overwrite = true`.
- Each model directory contains the SavedModel and `assets/`:
  - SNGP: `metadata.json` (the legacy keys read by the CharLES exporters, plus a `bfmwm_v2` block with the
    full config, class groups, datasets, seed and versions), `history.json` and `split_indices.npz`.
  - Error model: `error_norm_stats.json` (legacy format), `metadata.json`, `training_history.json`,
    `val_loss.csv` and `split_indices.npz`. Periodic snapshots go to `<SaveDir>/checkpoints/<ModelName>/`.
- Diagnostics (confusion matrices, class counts, training histories) go to `results/training/<ModelName>/`.
- The error model keeps the weights of its best validation epoch. The shipped turbulent model
  `…_ep3811` is the best epoch of a 10000-epoch run.

### 3. A-priori testing

```bash
python test_turb.py test/test_turb.toml [--gpu N] [--output-dir DIR] [--max-samples N]
python test_lamturb.py test/test_lamturb.toml [--gpu N] [--output-dir DIR] [--max-samples N]
```

As in training, the TOML, every test dataset file and every model are checked before the models are loaded.

The output directory (`[general] OutputDir`) receives:
- spatial maps of the errors, σ, confidence and class labels (`map_*`), and raw-n1 maps of the laminar
  region for lam-turb;
- $|C_f|$ line plots at one n1 station per x (`cf_lines*`);
- error and confidence PDFs (`pdf_*`);
- smoothed tripcolor maps with the OOD and near-wall-fallback regions hatched (`tri_*`);
- `summary.json`, with error statistics overall and per dataset, blend diagnostics, class counts and the
  lam/turb switch counts;
- optionally `arrays.npz`, with every per-point field.

The test truth is the dataset target. **No prediction uses any true-target information**: gates, fallbacks and
y+ use only quantities available at inference time, as in CharLES. All comparisons are on magnitudes.

### 4. Multi-dataset a-priori benchmark

```bash
python tools/apriori_benchmark.py benchmark/apriori_benchmark.toml [--gpu N] [--datasets k1 k2 ...] [--scale linear|log|both]
python tools/apriori_benchmark.py --replot results/apriori_benchmark --scale log   # redraw without TF/torch
```

Per-dataset (and per-group) error statistics of the baseline, blended, always-corrected and EQWM
predictions go to `cases.csv`, `overall.json` and `points.npz`. Figures: baseline-vs-corrected scatter
(linear, zoom, log-log), σ- and confidence-coloured scatters, and PDFs. Details are in `tools/README.md`.

### 5. Export to CharLES BFMv2

`tools/export_*` write the baseline, SNGP and error-model weights in the formats the CharLES BFMv2
`BFM_*_data` files use (see `tools/README.md`).

### Running on a GPU node (Slurm)

The scripts use one GPU (`--gpu 0` inside a job). The venv's base interpreter must exist on the compute
node. If it does not, `source <venv>/bin/activate && python ...` silently runs another Python, and the job
fails at the first import (`ModuleNotFoundError: No module named 'numpy'`).

On the cluster, the GPU nodes have only `/usr/bin/python3.9`, so the shared torch-venv cannot be activated there.
Its packages run with the cluster Python 3.11.6 (the interpreter of `module load <python3.11 module>`,
same ABI), with the venv's `site-packages` on `PYTHONPATH`. This setup was checked on a P100 node (torch with
CUDA, TensorFlow on the GPU, tf_keras + TensorFlow Probability, tf-models); full training and test jobs have
not been run with it yet. Example job script, submitted from the repository root with `sbatch job.sh`:
```bash
#!/bin/bash
#SBATCH -A <account> -p <gpu-partition> --gres=gpu:1 -c 8 --mem=48G -t 08:00:00
VENV=<path-to-venv>
PY=<cluster python3.11>
# replaces the PYTHONPATH of the auto-loaded modules, whose Python 3.10 packages would shadow the venv's
export PYTHONPATH=$VENV/lib/python3.11/site-packages
export OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
"$PY" train_turb.py --sngp sngp/sngp_turb.toml --gpu 0
```

## The blended prediction

Inputs (InputScaling 2): $\pi_1 = u_1 n_1/\nu$, $\pi_2 = u_p n_1/\nu$, $\pi_3 = u_2 n_1/\nu$. Output:
$\pi_{out} = u_\tau n_1/\nu$, which is the $y^+$ of that $u_\tau$.

1. **Baseline:** $\hat y_b = |\mathrm{NN}(\pi)|$. The network is trained on magnitudes, but its linear output
   layer can return small negative values (close to the wall and around separation).
2. **Error model:** a Gaussian NLL network on z-scored inputs gives the error mean $\mu$ and the aleatoric
   $\sigma$ (in $\pi_{out}$ units). It is trained on $\epsilon = y - \mathrm{NN}(\pi)$, the error of the raw
   output, so the corrected candidate is $y_{corr} = |\mathrm{NN}(\pi) + \mu|$.
3. **SNGP confidence:** $p = \mathrm{softmax}\big(\mathrm{logits}/\sqrt{1+\tfrac{\pi}{8} v}\big)$, with
   $v = 1$ by default (as in CharLES) or the GP variance (`[confidence] Covariance = "gp"`).
   Then $C = 1 - H(q)/\ln(K-K_{tr}+1)$, with $q = [\sum_{c<K_{tr}} p_c,\ p_{K_{tr}}, \dots, p_{K-1}]$.
   Post-hoc class groups change only labels and plots, never $C$.
4. **Aleatoric metric** (low-value blend):
   $m = w\,\frac{100\sigma}{|\hat y_b|} + (1-w)\frac{100\sigma}{\sqrt{|\pi_1|}}$, with
   $w = r_\uparrow(|\hat y_b|; 2, 1)$. The `Type = "ratio"` option gives $\sigma/|\mu|$ instead.
5. **Blend:**
   - Ramps: $r_\uparrow(m;t,d) = \mathrm{clip}\big(\frac{m-(t-d)}{2d},0,1\big)$ and
     $r_\downarrow(m;t,d) = \mathrm{clip}\big(\frac{t+d-m}{2d},0,1\big)$.
   - Weights: $\alpha_a = r_\downarrow(m)$, $\alpha_c = r_\uparrow(C)$, $\alpha_{nw} = r_\uparrow(y^+_{nw})$.
   - Prediction: $y = \alpha_{nw}\big[\alpha_c(\alpha_a y_{corr} + (1-\alpha_a)\hat y_b) + (1-\alpha_c)\,y_{fb}\big] + (1-\alpha_{nw})\,y_{fb}$.
6. **Fallback $y_{fb}$:**
   - **Turbulent points:** the EQWM, which is the CharLES two-layer law: $u^+ = y^+$ for
     $y^+ \le 11.0623$, else $u^+ = \ln(y^+)/0.41 + 5.2$, solved from $|\pi_1| = u^+ y^+$. Here
     $y^+_{nw} = \pi_{out,EQWM}$. `[eqwm] Law = "parent"` uses the notebooks' `log_law_solve` instead.
   - **Laminar points** (lam-turb, $k_t/k_m \le$ `KtKmThreshold` = 0.002): the laminar law
     $\tau_w = \mu|u_1|/y_1$, i.e. $\pi_{out} = \sqrt{|\pi_1|}$, for both fallbacks. Its own $y^+_{nw} = \sqrt{|\pi_1|}$.
7. **Evaluation:** $e\% = w_t\,\frac{100(y-\hat y)}{y} + (1-w_t)\,\frac{100(y-\hat y)}{\sqrt{|\pi_1|}}$, with
   $w_t = r_\uparrow(y; 2, 1)$ on the true $\pi_{out}$.

## Configuration

Every key of every TOML is explained by comments in the shipped files. The notebook flags map to them as follows:

| Notebook | v2 |
|---|---|
| `train_flag`, `model_path` (baseline) | `train_wm_proposal.py`; `[baseline] Checkpoint` |
| `train_sngp_flag` / `save_sngp_flag`, `train_err_model_flag` / `save_flag` | run `train_*.py` with `--sngp` / `--error`; `[general] Save`, `Overwrite` |
| `include_extra_data_flag`, `add_datasets` | `[data] AddDatasets` (`[]` = none), `TrainRatio` |
| `config_loaded_model['data']['upy'/'downy']` | `[data] upy`, `downy`, per model (`downy` is honoured even if `upy` ≥ 1) |
| `labels_to_remove` (integers) | `[classes] RemoveClasses` (names) |
| `merge_groups_flag`, `group_definitions` | `[classes.MergeGroups]` (names) |
| `rename_mapping` | `[classes.DisplayNames]` (stored in the SNGP metadata; test TOMLs may override) |
| `apply_posthoc_merging`, `posthoc_group_definitions` | `[classes.PosthocGroups]` |
| `num_classes` | `[classes] NumClasses` |
| `downsample_sngp_data_factor` | `[data] DownsampleBaseline` |
| `recompute_norm_with_extra_flag` | `[data] RecomputeNorm` |
| SNGP `epochs`, `batch_size`, `lr_schedule` | `[training] Epochs`, `BatchSize`, `LearningRate`, `EndLearningRate`, `DecayUnit` |
| SNGP architecture (hard-coded) | `[model] NumLayers`, `NumHidden`, `SpecNormBound`, `NumInducing`, `GpKernelScale`, `Dropout`, `DropoutActive` |
| class weights (always balanced) | `[training] ClassWeights` |
| `custom_err_layers_flag`, `Nneurons`, `Ndepths` | `[model] HiddenLayers` |
| error `epochs`, `error_lr_schedule` | `[training] Epochs`, `LearningRate`, `EndLearningRate`, `DecayFraction` |
| `EarlyStopping` / `SaveEveryNEpochs` (from the baseline TOML) | `[training.EarlyStopping]`, `[training.Snapshots]` |
| `test_datasets`, `downsample_factor` | test `[data] Datasets`, `Stride` |
| `thresh_aleatoric_*`, `thresh_conf`, `thresh_yplus` (+ deltas) | test `[gates] Aleatoric`, `Confidence`, `Yplus` |
| `utau_thresh`, `threshold_basis`, `y_apost_true_thresh` | `[gates] LowValue` (on the baseline) and `[metrics] LowValue` (on the truth) |
| `kt_km_threshold` | `[switch] KtKmThreshold` |
| `selection_mode`, `wall_normal_parameter*`, `x_transition`, `ramp_width`, `window_size` | `[plots.cf_lines]` |

Dropped notebook features:
- sample weighting;
- log-scaled inputs;
- alternative error targets (C_f, δ-based, τ_w-scaled);
- the inner-region mode;
- hierarchical training and Sobolev regularisation;
- the aleatoric-only and PyTorch error models;
- the x/y coordinate filters;
- the LES a-posteriori mode (`use_y_apost_flag`);
- truth-based signs and gates.

Changed defaults (the old behaviour stays available as options):
- The SNGP learning rate decays per epoch (`DecayUnit = "step"` reproduces the shipped models).
- SNGP dropout is applied (`DropoutActive = false` reproduces the shipped models).

Notebook behaviour that was fixed:
- The always-corrected candidate is $|\mathrm{NN}(\pi) + \mu|$. The notebooks used $\big|\,|\mathrm{NN}(\pi)| + \mu\,\big|$,
  which differs where the raw baseline output is negative.
- The confidence uses the un-merged SNGP classes; post-hoc merging changes only labels and plots (it inflated $C$).
- The SNGP input normalisation is read from the model's metadata, and labels are the argmax over the trained classes.
- The OOD relabelling of plots has no side effects on the class lists.
- A point is laminar if `kt_km <= KtKmThreshold`.
- `downy` is honoured even when `upy` ≥ 1.
- Percent errors use the low-value blend on the true $\pi_{out}$ for every prediction, including the EQWM.

## Shipped models (`models/`, git-ignored)

| Model | Role |
|---|---|
| `cleaned_up_20251208_ymax_0_15_ymin_0_0025.pth` | turbulent baseline (both variants) |
| `NN_wm_CH0_G0_S0_tn560665_vn140167_fds0_lds0_customw1_inputs2_final_ep30000_tl0.00032144_vl0.00031670.pth` | laminar baseline |
| `sngp_model_Baseline_cleanedupNew_Add_ExpertExtFlow_ep4000_merged` | turbulent SNGP (default in both test scripts; the one in CharLES BFMv2) |
| `sngp_model_Baseline_cleanedupNew_Add_ExpertExtFlow_ep5000_merged` | turbulent SNGP of the turbulent-only notebook (not exported to CharLES) |
| `sngp_model_Baseline_selftrained_lam_Add_ExpertExtFlowUPY0p8_ep20000_merged` | laminar SNGP |
| `error_model_Baseline_cleanedupNew_Add_ExpertExtFlow_CustomLayers_N60xL15_wVal_ep3811` | turbulent error model (both variants) |
| `error_model_Baseline_selftrained_lam_Add_ExpertExtFlowUPY1_CustomLayers_N60xL6_wVal_ep4000` | laminar error model |

- The shipped models were trained without seeds, so their exact splits cannot be reproduced.
- The laminar error model was trained on the laminar A-airfoil files up to n1/δ = 1 (`down0p005_up1`, the
  "UPY1" in its name), which `error/error_lamturb.toml` uses. The laminar SNGP was trained on the narrower
  `down0p02_up0p8` files (`sngp/sngp_lamturb.toml`).
- The training TOMLs default to new names (`*_v2`) so that the shipped models are never overwritten.

## Baseline testing tools (upstream)

### Interactive testing

```bash
# Load model and test on selected datasets
python test_model.py --model_path model_name.pth
```
When prompted, enter a dataset name or 'P' to print available datasets.

### Generating test reports

```bash
# Test model and generate HTML report
python test_model_summary_table.py model_name.pth
```

This generates:
- CSV file with performance metrics
- HTML table visualization showing performance across datasets

## Changes relative to upstream

| File | Change |
|---|---|
| `src/data_handler.py` | Default `downy` is **0.0** (upstream 0.005) in both loaders. In near-wall mode (`near_wall_threshold > 0`) reversed-flow points are kept: there is no `u1 > 0` mask (upstream had one). The filter message of `load_external_dataset` also prints the applied `upy`. |
| `src/wall_model.py` | `load_data()` sets `model.flow_type`, the per-sample dataset names (upstream left it unset). Near-wall test plots use the exact inverse of the signed log1p. `train()` passes the split indices to the trainer. |
| `src/trainer.py` | Checkpoints also store `train_index` / `valid_index` (int64 tensors). |
| `src/README.md` | New section listing the changes to `src/`. |
| `train_wm_proposal.py` | Raises `RuntimeError` instead of a string. Optional `[general] Seed`. |
| `wall_model_cases.py` | All dataset keys of the BFM_WM fork (A-airfoil full 2D fields, hump regions, a-posteriori sets, ...), plus the three `aairfoil_2M_full2Dfield_down0p005_up1_laminar_*` keys. |
| `plot_utils.py` | No longer loads a personal matplotlib style file. |
| `.gitignore` | `inputfiles/` is tracked. |
| `data/` | See [Data](#data). |
| `README.md` | `test_model.py` takes `--model_path` (the old usage line was out of date). |

## To do

- **Input sign frame.** CharLES BFMv2 (GFIX_v3) gives the networks inputs in a local u2-positive frame:
  sign($\pi_1$) = sign($u_1 \cdot u_2$), $\pi_3 = |u_2|\, n_1/\nu$, and the pressure gradient is projected on the $u_2$
  direction. The training data (and therefore this repository's a-priori tests) use the datasets' global-x
  frame. On reversed-flow points the two frames give different inputs: the current networks are much less
  accurate in the CharLES frame (e.g. gaussian_2M_SEP, where $u_2 < 0$ on 72 % of the points: baseline mean
  |e| 7.5 % in the dataset frame vs 17.8 % in the CharLES frame).
  - Plan: regenerate the training data in the u2-positive frame (data-generation repository) and retrain all
    models.
  - Why this frame: a wall model in a real application does not always know the global flow direction, so
    reversal cannot be defined relative to it. Taking $u_2$ as the local reference decides whether the flow is
    locally reversed and which way the pressure gradient points. It also gives the desired symmetry: if $u_1$
    and $u_2$ point the same way, the prediction (a magnitude) must not depend on whether the local flow runs
    forward or backward relative to the global flow. The sign of $\tau_w$ is assigned after the prediction,
    from the global direction of $u_1$.

## Provenance

The scripts replace the notebooks of the BFM_WM error-model fork. Code comments refer to the notebooks by
these abbreviations:
- **TURB-NB**: `BFMWM_ErrorModel copy_gaussian2M_cleaned4context.ipynb` (turbulent-only);
- **LT-NB**: `BFMWM_ErrorModel copy_aairfoil2M-LamTurbModel-cleaned4context2.ipynb` (laminar-turbulent);
- **EVB**: `Error_vs_baseline_plotting.ipynb` (multi-dataset a-priori benchmark), with its variants
  **LOG** (`Error_vs_baseline_plotting_loglog.ipynb`, log-log scatter) and **AIAA**
  (`Error_vs_baseline_plotting_aiaa.ipynb`, always-corrected vs blended PDF).

`tools/apriori_benchmark.py` ports EVB, LOG and AIAA; the CharLES export scripts come from the same fork.
These notebooks are local, unpublished files of the BFM_WM fork (git-ignored or untracked there).
Notebook and cell references in code comments (e.g. `TURB-NB c64`, `LT-NB cell 78`) refer to their final
versions and are kept for traceability.
