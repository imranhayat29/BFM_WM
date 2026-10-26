# BFM_WM_v2

Building-block flow model (BFM) wall model with an **error model** (aleatoric uncertainty and a correction
of the baseline) and an **SNGP confidence model**, combined in a blended "expert" prediction.

- **Baseline:** a neural-network wall model that predicts the wall shear stress (as $u_\tau n_1/\nu$) from
  velocity and pressure-gradient inputs at a matching point.
- **Error model:** a second network that predicts the mean $\mu$ and the spread $\sigma$ (aleatoric
  uncertainty) of the baseline's error.
- **SNGP confidence model:** a spectral-normalised neural Gaussian process (SNGP) classifier of the flow type.
  The entropy of its class probabilities gives a confidence $C$ that the input lies in the training
  distribution; inputs with low $C$ are treated as out-of-distribution (OOD).
- **Blended prediction:** the corrected prediction where the error model is trusted, the baseline where it
  is not, and an equilibrium wall model (EQWM) or a laminar law where the confidence is low or the matching
  point is very close to the wall.

This repository is a fork of [yuenongling/BFM_WM](https://github.com/yuenongling/BFM_WM) (git remote
`upstream`). The baseline wall model, its data handling and its training/testing scripts are upstream's
(with the changes listed in [Changes relative to upstream](#changes-relative-to-upstream)). This repository
adds the error model, the SNGP confidence model, the blended prediction, their training and test scripts
with TOML configs, an a-priori test on all cases, and exporters to the CharLES BFMv2 solver (the
group's GPU LES code that runs this wall model a posteriori). The formulas match those of CharLES BFMv2. One
difference remains, the sign frame of the network inputs on reversed-flow points (see [To do](#to-do)).

Two variants are kept separate:

| Variant | Branches | Training | Testing |
|---|---|---|---|
| **Turbulent-only** | one baseline, SNGP and error model | `train_turb.py` | `test_turb.py` |
| **Laminar-turbulent** | a laminar and a turbulent branch, hard-switched by the $k_t/k_m$ sensor | `train_lamturb.py` | `test_lamturb.py` |

$k_t/k_m$ is the ratio of the fluctuating to the mean kinetic energy. A point is laminar if
$k_t/k_m \le$ 0.002 (`[switch] KtKmThreshold`). Both branches are evaluated on every point and the results are
switched per point. The value is read from the dataset's `kt_km` column; datasets without one are classified
by flow name (names containing `laminar` or starting with `fs_` are laminar, all others turbulent).

## Repository layout

```
train_wm_proposal.py     baseline wall-model training (upstream)
test_model.py, ...       baseline testing tools (upstream)
train_turb.py            SNGP and/or error-model training, turbulent-only
train_lamturb.py         SNGP and/or error-model training, laminar + turbulent branches
test_turb.py             a-priori test of the blended prediction, turbulent-only
test_lamturb.py          a-priori test of the blended prediction, laminar-turbulent
test_turb_allcases.py    a-priori test of the turbulent blended prediction on all cases (80 dataset keys)
inputfiles/
  input2_example.toml    commented example baseline config in upstream's format (see Workflow, step 1)
  baseline/              baseline configs of the reference checkpoints (baseline_turb, baseline_lam)
  sngp/                  SNGP training configs (sngp_turb, sngp_lamturb)
  error/                 error-model training configs (error_turb, error_lamturb)
  test/                  test configs (test_turb, test_lamturb, test_turb_allcases)
src/                     baseline package (upstream, see src/README.md)
src_err/                 error model, SNGP, blended correction, data, metrics, plots
tools/                   CharLES export scripts (see tools/README.md)
wall_model_cases.py      dataset registry (upstream + additional keys)
data/                    HDF5 datasets (git-lfs)
models/                  reference models (git-lfs): baseline/, sngp/, error/ (see Reference models)
docs/DEVELOPMENT.md      developer notes (see Development notes)
```

`src_err/` modules:

| Module | Content |
|---|---|
| `config.py` | Reads and validates the SNGP, error-model and test TOMLs (unknown keys are errors) |
| `data.py` | Baseline checkpoint and train/valid split, dataset loading with the n1/δ window, additional-data split, $k_t/k_m$ |
| `classes.py` | SNGP class bookkeeping by name: remove, merge, post-hoc groups, display names |
| `sngp.py` | SNGP model, training, saving/loading (with metadata), batched prediction, confidence |
| `error_model.py` | Error model build/training/saving/loading/prediction, normalisation statistics |
| `gating.py` | EQWM and laminar law, aleatoric metric, ramps, the 3-switch blend |
| `lamturb.py` | $k_t/k_m$ switch and composite class labels |
| `pipeline.py` | The shared inference pipeline of the three test scripts |
| `evaluation.py` | Percent errors and summary statistics |
| `training.py` | Orchestration of the training scripts |
| `reporting.py`, `plotting.py`, `selection.py` | Figures, `summary.json`, station selection for line plots |
| `cli.py` | Command-line helpers shared by the training and test scripts, including the check that every dataset and model file exists |
| `utils.py` | Seeds, devices, JSON, hashing and path helpers |

## Setup

- **Python 3.11** is required: the saved error models contain cloudpickled Python 3.11 bytecode, and the
  TensorFlow stack is pinned for legacy-Keras SavedModels. Create a virtual environment and install the pinned
  packages of `requirements.txt` (PyTorch, TensorFlow with tf_keras, TensorFlow Probability and tf-models-official,
  and the scientific Python stack):
  ```bash
  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- **On a cluster**, build the venv from a Python 3.11 that also exists on the compute nodes (e.g. one provided
  by an environment module): a venv only links to its base interpreter, so batch jobs fail at the first import
  if that interpreter is missing on the node. If environment modules put other Python packages on `PYTHONPATH`,
  run `unset PYTHONPATH` before creating, filling and activating the venv; those packages would shadow the
  venv's.
- **Getting the code, data and reference models.** The data and the reference models are stored with git-lfs, so
  install [git-lfs](https://git-lfs.com) first. Then:
  ```bash
  git clone -b v2 https://github.com/imranhayat29/BFM_WM.git BFM_WM_v2
  cd BFM_WM_v2
  git lfs install --local && git lfs pull     # about 1.1 GB to download, about 2.2 GB on disk
  ```
- Run all scripts from the repository root. Paths are relative to it: `./inputfiles`, `./models`, `./data`,
  `./results`.
- `TF_USE_LEGACY_KERAS=1` is set automatically by `src_err` before TensorFlow is imported.
- On a shared login node, limit the threads, because torch and TensorFlow otherwise see every core:
  `export OMP_NUM_THREADS=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1`.

## Data

Data are stored in `data/` directories. They cover data summarized in [Table](https://docs.google.com/spreadsheets/d/18hcHtwxN7-uUAkYSo2UL1Tny2SpPDvQiWsKEqjHOlh0/edit?usp=sharing) and the data generation repository [Repo](https://github.com/yuenongling/WM_data_creation?tab=readme-ov-file).

`data/` holds the **training data** of all reference models. Differences from upstream's data:

| File | Change | Why |
|---|---|---|
| `FS_ZPG_data.h5`, `FS_APG_data.h5`, `FS_FPG_data.h5` | replaced by larger Falkner-Skan sets (about 5x more points) | used by both reference baselines (laminar and turbulent), hence also part of the turbulent SNGP and error-model training data |
| `curved_TBL_APG_data.h5`, `curved_TBL_FPG_data.h5` | replaced (same numbers, case names `curved_BL_APG`/`curved_BL_FPG` instead of `curved_BL`) | the SNGP needs the two classes to be distinct |
| `aairfoil_2M_full2Dfield_down0p02_up0p8_laminar_{ZPG,APG,FPG}.h5` | added | additional training data of the laminar SNGP (n1/δ 0.02-0.8, with a `kt_km` column) |
| `aairfoil_2M_full2Dfield_down0p005_up1_laminar_{ZPG,APG,FPG}.h5` | added | additional training data of the laminar error model (n1/δ 0.005-1, with a `kt_km` column) |

- **Dataset lookup.** The TOMLs name datasets by key. A key is resolved to a file in this order: an explicit
  path in the TOML's `[data.Files]` table, `./data/<key>_data.h5`, then the entry in `wall_model_cases.py`.
- **Lam-turb test set.** `aairfoil_2M_full2Dfield_down0_up1_cfgauss_w30_tr3_xalltoall.h5`, the default test
  set of `test_lamturb.py`, is **not included** in this repository (contact the repository owner to obtain it). It is
  a full 2D-field extraction of the A-airfoil at Re = 2M, with the `kt_km` column. Place the file, or a link to
  it, in `data/` under exactly that name (it is git-ignored there), or give its path in the `[data.Files]` table
  of the test TOML.
- `wall_model_cases.py` also registers keys whose files are not included (the comment at the top of the file
  lists them). Place such files in `data/` or give their paths in `[data.Files]` to use them.

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
- `baseline_turb.toml` and `baseline_lam.toml` are the exact configurations embedded in the two reference
  checkpoints. Only `GpuNum`, `UseWandb`, `SaveDir` (`./models/baseline`) and `Seed` (an optional key added by
  this repository) differ.
- New checkpoints store their train/valid split indices. The SNGP and error-model training reuses them, so
  both see the baseline's actual train/valid split. Checkpoints without stored indices (such as the two
  reference baselines) get a re-split seeded by the training TOML's `Seed`.
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

- Either `--sngp` or `--error` may be given alone. `--gpu` overrides `[general] GpuNum` (`-1` = CPU).
- Before any data are loaded, the TOMLs are validated (a TOML of the wrong kind, e.g. an error-model TOML
  given with `--sngp`, is reported as such) and every `AddDatasets` file and baseline checkpoint of all
  selected branches is looked up, so a typo fails within seconds, also with `--smoke`.
- With both, the baseline checkpoint must be the same. The additional datasets and their 80/20 split are
  shared when their `[data]` settings and `Seed` agree.
- In the laminar-turbulent variant each branch is trained on its own data with its own baseline; $k_t/k_m$
  is not used in training. The switch between the branches is applied at test time (and in CharLES).
- Models are saved to `<SaveDir>/<ModelName>` (default `./models/sngp` or `./models/error`). An explicit `ModelName` is required, and
  existing models are only replaced with `Overwrite = true`.
- Each model directory contains the SavedModel and `assets/`:
  - SNGP: `metadata.json` (the keys read by the CharLES exporters, plus a `bfmwm_v2` block with the
    full config, class groups, datasets, seed and versions), `history.json` and `split_indices.npz`.
  - Error model: `error_norm_stats.json` (normalisation statistics, read by the exporters), `metadata.json`,
    `training_history.json`, `val_loss.csv` and `split_indices.npz`. Periodic snapshots go to
    `<SaveDir>/checkpoints/<ModelName>/`.
- Diagnostics (confusion matrices, class counts, training histories) go to `results/training/<ModelName>/`.
- The error model keeps the weights of its best validation epoch. The reference turbulent error model
  `…_ep3811` is the best epoch of a 10000-epoch run.

### 3. A-priori testing

```bash
python test_turb.py test/test_turb.toml [--gpu N] [--output-dir DIR] [--max-samples N]
python test_lamturb.py test/test_lamturb.toml [--gpu N] [--output-dir DIR] [--max-samples N]
```

- The options override `[general] GpuNum`, `[general] OutputDir` and `[data] MaxSamples` (a deterministic,
  evenly spaced sample cap for quick runs; 0 = all points).
- As in training, the TOML, every test dataset file and every model are checked before the models are loaded.

The output directory (`[general] OutputDir`) receives:
- spatial maps of the errors, σ, confidence and class labels (`map_*`), and raw-n1 maps of the laminar
  region for lam-turb;
- $|C_f|$ line plots at one n1 station per x (`cf_lines*`);
- error and confidence PDFs (`pdf_*`);
- smoothed tripcolor maps with the OOD and near-wall-fallback regions hatched (`tri_*`);
- `summary.json`, with error statistics overall and per dataset, blend diagnostics, class counts and the
  lam/turb switch counts;
- optionally (`[general] SaveArrays = true`) `arrays.npz`, with every per-point field.

In these outputs, "corrected" (`summary.json` errors `corrected`, `map_corrected`, `tri_corrected`, "Expert
Model" in the PDFs) is the blended prediction $y_{final}$, and "always corrected" is the always-corrected
candidate $y_{corr}$ everywhere (see [The blended prediction](#the-blended-prediction)).

The test truth is the dataset target. **No prediction uses any true-target information**: gates, fallbacks and
y+ use only quantities available at inference time, as in CharLES. All comparisons are on magnitudes.

### 4. A-priori test on all cases

`test_turb_allcases.py` evaluates the turbulent wall model a priori (on stored datasets of wall-normal profiles,
without running a flow solver) on all cases at once, 80 dataset keys by default, and compares, per case, the mean
error of the baseline with the mean error of the blended model. Every dataset goes through the pipeline of
`test_turb.py` (`src_err/pipeline.predict_turb`): baseline, error model, SNGP confidence, EQWM fallback, and the blend
of three switches that use inference-time information only. Laminar and transitional datasets also take this
turbulent path (there is no $k_t/k_m$ switch here). The models are loaded once; one pass per dataset gives the
blended, the always-corrected ($y_{corr}$ everywhere) and the EQWM predictions.

The configuration is `inputfiles/test/test_turb_allcases.toml`. It is self-contained (its own model paths and gate
thresholds, with the same defaults as `test/test_turb.toml`), and every key is commented there.

```bash
python test_turb_allcases.py test/test_turb_allcases.toml --gpu 0        # all 80 dataset keys
python test_turb_allcases.py test/test_turb_allcases.toml --dry-run      # only check the TOML and every dataset/model file
python test_turb_allcases.py test/test_turb_allcases.toml --gpu -1 \
    --datasets CH TBL_5 "Gaussian bump 2M" --max-samples 3000 --output-dir results/smoke/test_turb_allcases
python test_turb_allcases.py --replot results/smoke/test_turb_allcases --scale log   # figures only, no torch/TF
```

| Option | Meaning |
| --- | --- |
| `<toml>` | TOML under `./inputfiles/` or a path; optional with `--replot` |
| `--gpu N` | GPU index, -1 = CPU; overrides `[general].GpuNum` |
| `--output-dir DIR` | output directory; overrides `[general].OutputDir` (with `--replot`: where the figures go, default `RUN_DIR`) |
| `--max-samples N` | per-dataset deterministic sample cap, applied after the domain filter (0 = all); overrides `[data].MaxSamplesPerDataset` |
| `--datasets k1 k2 ...` | evaluate only these keys; a `[groups]` name expands to its members, and a group is pooled from the members selected |
| `--scale linear\|log\|both` | scatter axes; overrides `[plots].Scale` |
| `--replot RUN_DIR` | regenerate every figure from `RUN_DIR/cases.csv` + `points.npz`, without torch, TensorFlow or inference; plot options from the given TOML, else from `RUN_DIR/overall.json` |
| `--dry-run` | validate the TOML and resolve every dataset and model file, then exit (no models loaded) |

**Processing of each dataset**

1. **Load** with the `[data]` wall-distance window (default n1/δ in `[0.0025, 0.15]`, the training window of the
   turbulent baseline) and `Stride`.
2. **Domain filter** `[evaluation].DomainFilter`: `"retau"` keeps points with $|u_\tau\delta/\nu|$ > `RetauMin` (5),
   `"piout"` keeps $|\pi_{out,true}| \ge$ `PioutMin` (2), `"none"` keeps all; a zero minimum disables the filter.
3. **Sample cap** `MaxSamplesPerDataset` (deterministic, evenly spaced).
4. **Prediction** with the pipeline.
5. **Percent errors** with the low-value blend on the truth (`[metrics].LowValue`, 2 ± 1), as in the other test
   scripts, so that near-zero targets do not blow up the errors.
6. **Outlier mask** (joint): a point is dropped if any of the baseline, blended or EQWM |e%| exceeds
   `OutlierPercent` (10000), so all predictions of a case are compared on the same points.
7. **Error-change filter** (optional, off by default): `MinErrorChangePercent > 0` keeps only points where the
   blended error differs from the baseline error by more than that value. It biases the statistics towards points
   the correction changes.

The dataset truth (the target and the dataset $u_\tau$) only selects evaluation points and gives the percent errors;
it never enters a prediction. Missing dataset files and models are reported before the models are loaded; a dataset
that fails later (unreadable, or no points left after a filter) is recorded as skipped with its reason. Only the
datasets listed in `[data].Datasets` are evaluated.

**Cases, roles and statistics**

- **Cases.** Every dataset that is not in a group is one case. Every `[groups]` entry is one case whose members'
  points are pooled; members are not plotted individually. The default group `"Gaussian bump 2M"` pools the four
  `gaussian_2M_*` subsets (APG, FPG, SEP, MAPG) the baseline was trained on.
- **Roles.** A case is `training` (blue) if it is listed in `[data].TrainingDatasets`, otherwise `untrained`
  (maroon); a group is `training` if its name or any member is listed. The default list holds the training sets of
  the turbulent baseline plus the additional training sets of the error model and the SNGP. The JHU transitional
  boundary layer appears twice: `Transition_transition` (its transition region, which the error model and the SNGP
  were trained on, role `training`) and the full `Transition` case (role `untrained`).
- **Per case:** mean, median and P90 of |e%| for the baseline, the blended (final), the always-corrected and the
  EQWM predictions; improvement = mean baseline |e%| − mean final |e%|, and the maximum point-wise improvement; mean
  σ% and mean confidence $C$; the mean switch weights $\alpha_a$, $\alpha_c$, $\alpha_{nw}$; the fractions of points
  fully corrected (all three weights 1) and in full EQWM fallback; point counts after the window (`n_raw`), the
  domain filter (`n_domain`), the cap (`n_eval`) and the outlier and error-change filters (`n_used`).
- **Overall:** the unweighted mean over cases (each case counts once; the headline number), the point-weighted mean
  over all used points, and both per role.

**Outputs**, written to `[general].OutputDir` (default `results/test_turb_allcases`):

| File | Content |
| --- | --- |
| `cases.csv` | one row per dataset, group member, group and skipped dataset (column `kind`), with the statistics above and the fraction of points whose stored network input is inconsistent with $u_1 n_1/\nu$ (a data check) |
| `overall.json` | resolved configuration, command line, model paths, git commit, package versions, overall averages, skipped datasets, timings |
| `points.npz` | float32 per-point errors, σ%, $C$, switch weights and $\pi_{out}$ values; `dataset`/`case` indices into `dataset_keys`/`case_names`; the `training` and `used` masks |
| console | a summary table of the mean \|e%\| per case with the overall averages, and a table of the points removed by the domain filter per dataset |

Figures are named `<RunName>_<figure>[_log].<fmt>` (default `RunName` = `test_turb_allcases`; `[plots].Figures`
selects them, `Formats` sets the formats):

| Figure | Content |
| --- | --- |
| `train_test` | one marker per case at (mean baseline \|e%\|, mean final \|e%\|), coloured by role, with the y = x line ("Same Performance"), the legend and, on linear axes, the dash-dot zoom box. With `Scale = "log"` or `"both"` also `train_test_log` on log-log axes: cases with non-positive or non-finite errors are skipped with a warning, and a limit <= 0 is set automatically from the data |
| `train_test_zoom` | the same scatter within `ZoomLimits` (linear only) |
| `sigma` | the scatter coloured by the mean σ% of each case (jet colour map, `SigmaColorRange`) |
| `confidence` | the scatter coloured by the mean confidence $C$ of each case (`ConfidenceColorRange`) |
| `pdf`, `pdf_confidence` | PDFs of \|e%\| and of $C$, pooled over the used points; curves from `PdfCurves`, points from `PdfSubset`; kernel density estimates are fitted inside the plotted range, so the densities are conditional on that range |
| `pdf_always_vs_blended` | \|e%\| PDFs of the baseline, the always-corrected and the blended predictions (needs `CompareAlways = true`) |

### 5. Export to CharLES BFMv2

`tools/export_*` write the baseline, SNGP and error-model weights in the formats the CharLES BFMv2
`BFM_*_data` files use (see [tools/README.md](tools/README.md)).

### Running on a GPU node (Slurm)

The scripts use one GPU (`--gpu 0` inside a job). Example job script, submitted from the repository root with
`sbatch job.sh` (adapt the account, partition and resources to your cluster):
```bash
#!/bin/bash
#SBATCH -A <account> -p <gpu-partition> --gres=gpu:1 -c 8 --mem=48G -t 08:00:00
unset PYTHONPATH                      # if environment modules set it (see Setup)
source <path-to-venv>/bin/activate
export OMP_NUM_THREADS=8 TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2
python train_turb.py --sngp sngp/sngp_turb.toml --gpu 0
```
If the job fails at the first import (`ModuleNotFoundError: No module named 'numpy'`), the venv's base interpreter
does not exist on the compute node: `source <venv>/bin/activate` then silently runs another Python (see
[Setup](#setup)).

## The blended prediction

Inputs (InputScaling 2): $\pi_1 = u_1 n_1/\nu$, $\pi_2 = u_p n_1/\nu$, $\pi_3 = u_2 n_1/\nu$. Here $n_1$ is the wall
distance of the matching point, $u_1$ and $u_2$ are the wall-parallel velocities at $n_1$ and $3n_1$ (signed along a
fixed wall tangent in the datasets), and $u_p = \mathrm{sign}(dp/ds)\,(\nu|dp/ds|/\rho)^{1/3}$. Output:
$\pi_{out} = u_\tau n_1/\nu$, which is the $y^+$ of that $u_\tau$. $y$ denotes the dataset target $\pi_{out}$, and $\delta$
(in the $n_1/\delta$ windows of the TOMLs) the boundary-layer thickness.

1. **Baseline:** $\hat y_b = |\mathrm{NN}(\pi)|$. The network is trained on magnitudes, but its linear output
   layer can return small negative values (close to the wall and around separation).
2. **Error model:** a Gaussian negative-log-likelihood (NLL) network on z-scored inputs gives the error mean
   $\mu$ and the aleatoric $\sigma$ (in $\pi_{out}$ units; $\sigma$ = `SigmaFloor` + softplus). It is trained on
   $\epsilon = y - \mathrm{NN}(\pi)$, the error of the raw output, so the corrected candidate is
   $y_{corr} = |\mathrm{NN}(\pi) + \mu|$.
3. **SNGP confidence:** $p = \mathrm{softmax}\big(\mathrm{logits}/\sqrt{1+\tfrac{\pi}{8} v}\big)$, with
   $v = 1$ by default (as in CharLES) or the GP variance (`[confidence] Covariance = "gp"`).
   Then $C = 1 - H(q)/\ln(K-K_{tr}+1)$, with $q = [\sum_{c<K_{tr}} p_c,\ p_{K_{tr}}, \dots, p_{K-1}]$, where $K$ is the
   number of GP output units (`NumClasses`) and $K_{tr}$ the number of trained classes: the untrained units act
   as OOD sinks. Post-hoc class groups change only labels and plots, never $C$.
4. **Aleatoric metric** (default `Type = "percentage"`, with a low-value blend):
   $m = w\,\frac{100\sigma}{|\hat y_b|} + (1-w)\frac{100\sigma}{\sqrt{|\pi_1|}}$, with
   $w = r_\uparrow(|\hat y_b|; 2, 1)$ (`[gates] LowValue`). The blend keeps $m$ finite where $|\hat y_b|$ is
   tiny (near separation and reattachment). `Type = "ratio"` uses $m = \sigma/|\mu|$ instead.
5. **Blend:**
   - Ramps: $r_\uparrow(m;t,d) = \mathrm{clip}\big(\frac{m-(t-d)}{2d},0,1\big)$ and
     $r_\downarrow(m;t,d) = \mathrm{clip}\big(\frac{t+d-m}{2d},0,1\big)$, with threshold $t$ and half-width $d$
     (`Threshold`, `Delta` of the `[gates]` entries).
   - Weights: $\alpha_a = r_\downarrow(m)$ (`Aleatoric`), $\alpha_c = r_\uparrow(C)$ (`Confidence`),
     $\alpha_{nw} = r_\uparrow(y^+_{nw})$ (`Yplus`).
   - Prediction: $y_{final} = \alpha_{nw}\big[\alpha_c(\alpha_a y_{corr} + (1-\alpha_a)\hat y_b) + (1-\alpha_c)\,y_{fb}\big] + (1-\alpha_{nw})\,y_{fb}$.
6. **Fallback $y_{fb}$** (used by both the confidence and the near-wall switch):
   - **Turbulent points:** the EQWM. By default (`[eqwm] Law = "charles"`) this is the CharLES BFMv2
     two-layer law: $u^+ = y^+$ for $y^+ \le 11.0623$, else $u^+ = \ln(y^+)/0.41 + 5.2$, solved from
     $|\pi_1| = u^+ y^+$. Here $y^+_{nw} = \pi_{out,EQWM}$.
   - **Laminar points** (lam-turb, $k_t/k_m \le$ `KtKmThreshold` = 0.002): the laminar law
     $\tau_w = \mu|u_1|/y_1$ (no pressure-gradient term), i.e. $\pi_{out} = \sqrt{|\pi_1|}$, with its own
     $y^+_{nw} = \sqrt{|\pi_1|}$. Laminar points thus go through the same three switches as turbulent points,
     with the laminar branch's networks and the laminar law in place of the EQWM, as in CharLES BFMv2.
7. **Evaluation:** $e\% = w_t\,\frac{100(y-\hat y)}{y} + (1-w_t)\,\frac{100(y-\hat y)}{\sqrt{|\pi_1|}}$ for any prediction
   $\hat y$ ($\hat y_b$, $y_{corr}$, $y_{final}$ or the EQWM), with $w_t = r_\uparrow(y; 2, 1)$ on the true $\pi_{out}$
   (`[metrics] LowValue`).

## Configuration

All settings live in TOML files under `inputfiles/`; every key is explained by comments in the files. The
scripts take the path relative to `./inputfiles/` (or any path). The TOMLs are validated on loading:
defaults are filled in for missing keys, unknown keys are errors (with a "did you mean" suggestion), and a TOML given
to the wrong script or option is reported as such.

| TOML | Read by | Content |
|---|---|---|
| `baseline/baseline_turb.toml`, `baseline/baseline_lam.toml`, `input2_example.toml` | `train_wm_proposal.py` | baseline training (upstream format; every key is commented in `input2_example.toml`) |
| `sngp/sngp_turb.toml` | `train_turb.py --sngp` | turbulent SNGP |
| `error/error_turb.toml` | `train_turb.py --error` | turbulent error model |
| `sngp/sngp_lamturb.toml` | `train_lamturb.py --sngp` | laminar and turbulent SNGPs |
| `error/error_lamturb.toml` | `train_lamturb.py --error` | laminar and turbulent error models |
| `test/test_turb.toml` | `test_turb.py` | models, test datasets, switches, plots |
| `test/test_lamturb.toml` | `test_lamturb.py` | as `test_turb.toml`, per branch, plus the $k_t/k_m$ switch |
| `test/test_turb_allcases.toml` | `test_turb_allcases.py` | models, datasets and groups, switches, filters, figures (see [Workflow, step 4](#4-a-priori-test-on-all-cases)) |

### Training TOMLs (`sngp/`, `error/`)

The lam-turb TOMLs hold a `[lam]` and a `[turb]` table, each a complete turbulent-only config; values of a
shared top-level `[general]` table are inherited by the branches that do not set them.

| Section | Main keys (default) |
|---|---|
| `[general]` | `ModelName` (required), `SaveDir` (`"./models/sngp"` / `"./models/error"`), `Save` (`true`), `Overwrite` (`false`), `GpuNum` (`0`; `-1` = CPU), `Seed` (required) |
| `[baseline]` | `Checkpoint` (required): baseline `.pth`, as a path relative to `./models` (e.g. `baseline/<file>.pth`) or any path. Its embedded config defines the baseline training data (the first SNGP classes) and the input scaling. |
| `[data]` | `AddDatasets` (`[]`): additional datasets, split 80/20 by `TrainRatio` (`0.8`); `upy` (`0.15`) and `downy` (`0.0`): n1/δ window of the additional data (`upy` ≥ 1: no upper cut); `Files` (`{}`): explicit file paths |
| `[data]`, SNGP only | `DownsampleBaseline` (`1`): keep every N-th baseline point; `RecomputeNorm` (`false`): z-score statistics over the (strided) baseline + additional training data instead of the full baseline training data |
| `[classes]`, SNGP only | `NumClasses` (`50`): GP output units (CharLES expects 50 turbulent, 30 laminar); `RemoveClasses`, `[classes.MergeGroups]`, `[classes.DisplayNames]`, `[classes.PosthocGroups]`, all by class **name** |
| `[model]`, SNGP | `NumLayers` (`6`), `NumHidden` (`128`), `SpecNormBound` (`0.9`), `NumInducing` (`1024`), `GpKernelScale` (`1.0`), `Dropout` (`0.1`), `DropoutActive` (`true`) |
| `[training]`, SNGP | `Epochs` (required), `BatchSize` (`50000`), `LearningRate` (`5e-4`) linearly to `EndLearningRate` (`1e-4`), `DecayUnit` (`"epoch"`), `ClassWeights` (`"balanced"` or `"none"`) |
| `[model]`, error model | `HiddenLayers` (required, e.g. 15 × 60), `SigmaFloor` (`1e-4`) |
| `[training]`, error model | `Epochs` (required), `BatchSize` (`50000`), `LearningRate` (`1e-3`) linearly to `EndLearningRate` (`8e-5`) over the first `DecayFraction` (`2/3`) of the epochs; `[training.EarlyStopping]` `RestoreBestWeights` (`true`), `Patience` (`0` = never stop early), `MinDelta` (`0.0`); `[training.Snapshots]` `SaveInterval` (`1000`; `0` = off) |

Two SNGP options select between the current training and the one of the reference SNGPs:
- `DecayUnit = "epoch"` decays the learning rate over all epochs; `"step"` counts optimiser steps instead,
  so the end value is reached early in training (as for the reference SNGPs).
- `DropoutActive = true` applies the dropout in training; `false` never applies it (as for the reference SNGPs).

### Test TOMLs (`test/`)

`test_lamturb.toml` has `[lam.models]`/`[lam.classes]` and `[turb.models]`/`[turb.classes]` instead of
`[models]`/`[classes]`, and a `[switch]` table. Defaults that differ between the variants are given as
turbulent-only / lam-turb.

| Section | Main keys (default) |
|---|---|
| `[general]` | `OutputDir` (`results/test_turb_gaussian_2M` / `results/test_lamturb`), `Formats` (`["png", "pdf"]`), `GpuNum` (`0`), `SaveArrays` (`false`) |
| `[models]` | `Baseline`, `Sngp`, `Error`: path relative to `./models` (e.g. `sngp/<name>`) or any path (default: the reference models of the branch) |
| `[classes]` | `PosthocGroups`, `DisplayNames` (default: those stored in the SNGP metadata) |
| `[data]` | `Datasets`, `upy` (`0.15` / `1.0`), `downy` (`0.0`), `Stride` (`2`: every 2nd point), `MaxSamples` (`0` = all), `Files` |
| `[eqwm]` | `Law`: `"charles"` (default, CharLES BFMv2 two-layer law) or `"parent"` (upstream's `src/baseline_models.log_law_solve`: log law, re-solved with a cubic near-wall layer below $y^+ = 23$) |
| `[confidence]` | `Covariance`: `"none"` (default, $v = 1$ as in CharLES) or `"gp"` (per-point GP variance) |
| `[gates]` | `Aleatoric` (`Type = "percentage"`, threshold `10` / `15` ± `5`; `Type = "ratio"` defaults to `1.0` ± `0.5`), `Confidence` (`0.7` ± `0.05` / `0.85` ± `0.1`), `Yplus` (`4` ± `2`), `LowValue` (`2` ± `1`) |
| `[metrics]` | `LowValue` (`2` ± `1`): low-value blend of the evaluation percent errors |
| `[switch]` (lam-turb) | `KtKmThreshold` (`0.002`): a point is laminar if `kt_km` ≤ threshold |
| `[plots]` | `SpatialMaps`, `CfLines`, `Pdfs`, `Tripcolor` (all `true`), and per-figure tables `[plots.maps]`, `[plots.cf_lines]`, `[plots.pdfs]`, `[plots.tripcolor]` (axes, colour ranges, station selection) |

The gate entries are inline tables, e.g. `Confidence = { Threshold = 0.7, Delta = 0.05 }`.

## Reference models (`models/`, git-lfs)

The pretrained reference models are stored with git-lfs in `models/`, in one folder per kind of model:
`baseline/` (PyTorch checkpoints), `sngp/` and `error/` (TensorFlow SavedModels). `git lfs pull` fetches them
together with the data. The TOMLs refer to a model by its path relative to `./models` (e.g. `sngp/<name>`) or by
any other path.

| Model (in `models/`) | Role | Used by default in | In CharLES BFMv2 |
|---|---|---|---|
| `baseline/cleaned_up_20251208_ymax_0_15_ymin_0_0025.pth` | turbulent baseline (both variants) | all training TOMLs (turbulent), all three test TOMLs | yes |
| `baseline/NN_wm_CH0_G0_S0_tn560665_vn140167_fds0_lds0_customw1_inputs2_final_ep30000_tl0.00032144_vl0.00031670.pth` | laminar baseline | lam-turb training TOMLs (`[lam]`), `test_lamturb.toml` | yes |
| `sngp/sngp_model_Baseline_cleanedupNew_Add_ExpertExtFlow_ep4000_merged` | turbulent SNGP (4000 epochs) | all three test TOMLs | yes |
| `sngp/sngp_model_Baseline_cleanedupNew_Add_ExpertExtFlow_ep5000_merged` | alternative turbulent SNGP (5000 epochs) | none | no |
| `sngp/sngp_model_Baseline_selftrained_lam_Add_ExpertExtFlowUPY0p8_ep20000_merged` | laminar SNGP | `test_lamturb.toml` | yes |
| `error/error_model_Baseline_cleanedupNew_Add_ExpertExtFlow_CustomLayers_N60xL15_wVal_ep3811` | turbulent error model (both variants) | all three test TOMLs | yes |
| `error/error_model_Baseline_selftrained_lam_Add_ExpertExtFlowUPY1_CustomLayers_N60xL6_wVal_ep4000` | laminar error model | `test_lamturb.toml` | yes |

- The training TOMLs are the recipes of these models: `sngp_turb.toml` (5000 epochs) of the ep5000 SNGP,
  the `[turb]` branch of `sngp_lamturb.toml` (4000 epochs) of the ep4000 SNGP, `error_turb.toml` and the
  `[turb]` branch of `error_lamturb.toml` of the turbulent error model, and the `[lam]` branches of the
  laminar models. With the default `DecayUnit` and `DropoutActive` they train with the current SNGP settings
  (see [Configuration](#training-tomls-sngp-error)).
- The reference models were trained without seeds, so their exact splits cannot be reproduced.
- The laminar error model was trained on the laminar A-airfoil files up to n1/δ = 1 (`down0p005_up1`, the
  "UPY1" in its name), which `error/error_lamturb.toml` uses. The laminar SNGP was trained on the narrower
  `down0p02_up0p8` files (`sngp/sngp_lamturb.toml`).
- New models are saved next to the reference models of their kind: `train_turb.py`/`train_lamturb.py` to
  `models/sngp/<ModelName>` and `models/error/<ModelName>` (error-model snapshots in `models/error/checkpoints/`),
  and `train_wm_proposal.py` to a subfolder per run in `models/baseline/`. Models trained locally are git-ignored;
  only the reference models are tracked.
- The training TOMLs default to new names (`*_v2`) so that the reference models are never overwritten. Models
  trained with them perform like the reference models (comparison in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md),
  section 12).

## Baseline testing tools (upstream)

### Interactive testing

```bash
# Load model and test on selected datasets
python test_model.py --model_path baseline/<checkpoint>.pth   # path relative to ./models
```
When prompted, enter a dataset name or 'P' to print available datasets.

### Generating test reports

```bash
# Test model and generate HTML report
python test_model_summary_table.py baseline/<checkpoint>.pth   # path relative to ./models
```

This generates:
- CSV file with performance metrics
- HTML table visualization showing performance across datasets

The upstream tools write to `./results/`, `./paper_plots/`, `./dummy/`, `./testing_results/` and `./tables/`
(created when needed, git-ignored), and their plots use LaTeX text rendering (matplotlib `text.usetex`), so a LaTeX
installation is needed for them. The scripts of this repository do not need LaTeX.

## Changes relative to upstream

| File | Change |
|---|---|
| `src/data_handler.py` | Default `downy` is **0.0** (upstream 0.005) in both loaders. In near-wall mode (`near_wall_threshold > 0`) reversed-flow points are kept: there is no `u1 > 0` mask (upstream had one). The filter message of `load_external_dataset` also prints the applied `upy`. |
| `src/wall_model.py` | `load_data()` sets `model.flow_type`, the per-sample dataset names (upstream left it unset). Near-wall test plots use the exact inverse of the signed log1p. `train()` passes the split indices to the trainer. |
| `src/trainer.py` | Checkpoints also store `train_index` / `valid_index` (int64 tensors). |
| `src/README.md` | New section listing the changes to `src/`. |
| `train_wm_proposal.py` | Raises `RuntimeError` instead of a string. Optional `[general] Seed`. |
| `wall_model_cases.py` | Additional dataset keys (A-airfoil full 2D fields and their laminar sets, NASA hump regions, a-posteriori LES sets, ...). Keys whose files are not included are listed at the top of the file. |
| `plot_utils.py`, `src/plot_utils.py`, `src/visualization.py` | No longer load a personal matplotlib style file. |
| `test_model.py`, `test_model_summary_table.py`, `src/wall_model.py` | Create their output folders when needed. The summary table keeps dots in the checkpoint name and accepts a model subfolder (e.g. `baseline/<checkpoint>.pth`). |
| `requirements.txt` | New: the pinned Python 3.11 environment. |
| `.gitignore` | `inputfiles/` and the reference models in `models/` are tracked; models trained locally, `docs/dev_local/` (local developer files), a local copy of the lam-turb test set and the output folders of the upstream testing tools are ignored. |
| `models/` | Reference models tracked with git-lfs (`models/.gitattributes`), in `baseline/`, `sngp/` and `error/`; upstream keeps `models/` out of git. |
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

## Development notes

The history of the code, the design decisions, and the provenance of the data and the reference models are
documented for developers in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Code comments use abbreviations
defined there.
