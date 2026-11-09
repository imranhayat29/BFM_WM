"""
Command-line helpers shared by the training and test scripts

train_turb.py and train_lamturb.py share the argument parser, the TOML loading and pairing
checks, the GPU selection and the config summary printed before training; test_turb.py and
test_lamturb.py share their argument parser and the GPU selection. All four scripts (and
test_turb_allcases.py) check with `check_inputs` that every dataset and model of their
TOMLs can be found.

This module imports neither torch nor TensorFlow, so the scripts can validate their TOMLs
and report configuration errors and missing files before the devices are configured.
"""
import argparse
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src_err.config import BRANCHES, load_error_config, load_sngp_config
from src_err.utils import model_hint, resolve_dataset_file, resolve_model_path


# ---------------------------------------------------------------------------------------
# Training scripts
# ---------------------------------------------------------------------------------------


def parse_train_args(argv: Optional[Sequence[str]], description: str, lamturb: bool = False) -> argparse.Namespace:
    """
    Command line of train_turb.py (and, with lamturb=True, train_lamturb.py)

    Args:
        argv: Argument list (None: sys.argv)
        description: Help text
        lamturb: Add the --branch option

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    variant = "lamturb" if lamturb else "turb"
    parser.add_argument("--sngp", metavar="TOML", help=f"SNGP training TOML (e.g. sngp/sngp_{variant}.toml)")
    parser.add_argument("--error", metavar="TOML", help=f"error-model training TOML (e.g. error/error_{variant}.toml)")
    if lamturb:
        parser.add_argument("--branch", choices=["lam", "turb", "both"], default="both",
                            help="branch(es) to train (default: both)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index, overrides [general] GpuNum (-1 = CPU)")
    parser.add_argument("--smoke", action="store_true",
                        help="smoke test: 2 epochs, capped samples, save to ./results/smoke/models/")
    parser.add_argument("--plots-only", action="store_true",
                        help="train nothing: rebuild the training data from the TOML and write the input-space figures "
                             "of the saved model(s) <SaveDir>/<ModelName> to results/training/<ModelName>/input_space/")
    args = parser.parse_args(argv)
    if args.sngp is None and args.error is None:
        parser.error("at least one of --sngp and --error is required")
    return args


def load_configs(args: argparse.Namespace, lamturb: bool) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load and validate the training TOMLs and check that they belong to the calling script

    A TOML of another kind (e.g. an error-model TOML given with --sngp) or of the other
    variant raises one ValueError naming the right option or script (see config.check_kind).

    Args:
        args: Parsed arguments (parse_train_args)
        lamturb: True for train_lamturb.py (TOMLs must have [lam]/[turb] tables), False for train_turb.py

    Returns:
        (sngp_cfg or None, error_cfg or None)
    """
    sngp_cfg = None if args.sngp is None else load_sngp_config(args.sngp, lamturb=lamturb)
    error_cfg = None if args.error is None else load_error_config(args.error, lamturb=lamturb)
    return sngp_cfg, error_cfg


def check_pair(sngp_cfg: Optional[Dict[str, Any]], error_cfg: Optional[Dict[str, Any]], where: str = "") -> None:
    """
    Check that an SNGP and an error model trained together share the baseline checkpoint

    Different seeds are allowed but announced, because the baseline and additional-data
    splits are then not shared.

    Args:
        sngp_cfg: SNGP branch config (None: nothing to check)
        error_cfg: Error-model branch config (None: nothing to check)
        where: Message prefix naming the branch, e.g. '[lam] '
    """
    if sngp_cfg is None or error_cfg is None:
        return
    a, b = sngp_cfg["baseline"]["Checkpoint"], error_cfg["baseline"]["Checkpoint"]
    if a != b:
        raise ValueError(f"{where}[baseline] Checkpoint differs between the SNGP ({a}) and error-model ({b}) TOMLs")
    if sngp_cfg["general"]["Seed"] != error_cfg["general"]["Seed"]:
        print(f"WARNING: {where}Seed differs (SNGP {sngp_cfg['general']['Seed']}, error model "
              f"{error_cfg['general']['Seed']}): the baseline and additional-data splits are not shared")


def print_summary(sngp_cfg: Optional[Dict[str, Any]], error_cfg: Optional[Dict[str, Any]], gpu: int,
                  smoke: bool, title: str, plots_only: bool = False) -> None:
    """
    Print the configuration summary of one branch before training

    Args:
        sngp_cfg: SNGP branch config (None: not trained)
        error_cfg: Error-model branch config (None: not trained)
        gpu: Selected GPU index (-1 = CPU)
        smoke: Smoke-test mode (the output directory and epochs are overridden)
        title: First line of the summary
        plots_only: --plots-only (input-space figures of the saved models, no training)
    """
    print("=" * 80)
    print(title + ("  [SMOKE TEST]" if smoke else "") + ("  [PLOTS ONLY: no training]" if plots_only else ""))
    print(f"  GPU: {gpu}")
    for kind, cfg in (("SNGP", sngp_cfg), ("Error model", error_cfg)):
        if cfg is None:
            continue
        g, d, t = cfg["general"], cfg["data"], cfg["training"]
        if smoke:
            print(f"  {kind}: {g['ModelName']} -> ./results/smoke/models (smoke; TOML: {g['SaveDir']}), Seed {g['Seed']}")
        else:
            print(f"  {kind}: {g['ModelName']} -> {g['SaveDir']} (Save {g['Save']}, Overwrite {g['Overwrite']}, "
                  f"Seed {g['Seed']})")
        print(f"    baseline {cfg['baseline']['Checkpoint']}")
        print(f"    AddDatasets ({len(d['AddDatasets'])}): {d['AddDatasets']}")
        print(f"    upy {d['upy']}, downy {d['downy']}, TrainRatio {d['TrainRatio']}"
              + (f", DownsampleBaseline {d['DownsampleBaseline']}, RecomputeNorm {d['RecomputeNorm']}"
                 if "DownsampleBaseline" in d else ""))
        if "classes" in cfg:
            print(f"    NumClasses {cfg['classes']['NumClasses']}, RemoveClasses {cfg['classes']['RemoveClasses']}, "
                  f"MergeGroups {list(cfg['classes']['MergeGroups'])}")
        if "HiddenLayers" in cfg["model"]:
            print(f"    HiddenLayers {len(cfg['model']['HiddenLayers'])}x{cfg['model']['HiddenLayers'][0]}")
        print(f"    Epochs {t['Epochs']}{' (smoke: 2)' if smoke else ''}, BatchSize {t['BatchSize']}, LR {t['LearningRate']} -> {t['EndLearningRate']}")
    print("=" * 80)


# ---------------------------------------------------------------------------------------
# Test scripts
# ---------------------------------------------------------------------------------------


def parse_test_args(argv: Optional[Sequence[str]], description: str, example: str) -> argparse.Namespace:
    """
    Command line of test_turb.py and test_lamturb.py

    Args:
        argv: Argument list (None: sys.argv)
        description: Help text
        example: Example TOML shown in the help, e.g. 'test/test_turb.toml'

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help=f"test TOML under ./inputfiles/ (e.g. {example}) or a path")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (-1 = CPU); overrides [general].GpuNum")
    parser.add_argument("--output-dir", default=None, help="output directory; overrides [general].OutputDir")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="deterministic sample cap (0 = all); overrides [data].MaxSamples")
    parser.add_argument("--show-cf-stations", action="store_true",
                        help="mark the n1 stations of the C_f line plots (red) on the map_* and tri_* maps; "
                             "sets [plots.maps] and [plots.tripcolor] ShowSelected")
    return parser.parse_args(argv)


def apply_plot_flags(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    """
    Apply the plot flags of the test command line to a loaded test config (in place)

    Args:
        args: Parsed arguments of parse_test_args
        cfg: Test config; --show-cf-stations sets ShowSelected in [plots.maps] and [plots.tripcolor]
    """
    if args.show_cf_stations:
        for table in ("maps", "tripcolor"):
            cfg["plots"][table]["ShowSelected"] = True


# ---------------------------------------------------------------------------------------
# All scripts
# ---------------------------------------------------------------------------------------


def select_gpu(args: argparse.Namespace, cfgs: Iterable[Optional[Dict[str, Any]]]) -> int:
    """
    GPU index of a run: --gpu, else the [general] GpuNum of the given configs

    Args:
        args: Parsed arguments with a `gpu` attribute (None: not given)
        cfgs: Configs of the run (None entries are skipped); if their GpuNum values disagree,
              the smallest is used and a warning is printed

    Returns:
        GPU index (-1 = CPU)
    """
    if args.gpu is not None:
        return int(args.gpu)
    nums = sorted({int(c["general"]["GpuNum"]) for c in cfgs if c is not None})
    if len(nums) > 1:
        print(f"WARNING: the TOMLs set different GpuNum values {nums}; using {nums[0]}")
    return nums[0]


def missing_model(role: str, name: str) -> Optional[str]:
    """
    Problem message if a model cannot be found, resolved as its loader resolves it

    Args:
        role: 'Baseline' (checkpoint file, data.load_baseline_model), 'Sngp' (sngp.load_sngp) or
              'Error' (error_model.load_error_model); the last two are SavedModel directories
        name: Path relative to ./models (e.g. sngp/<name>) or any path

    Returns:
        The message, or None if the model exists
    """
    if role == "Baseline":
        if os.path.isfile(resolve_model_path(name)):
            return None
        return f"Baseline checkpoint not found: {model_hint(name)}"
    if os.path.isdir(resolve_model_path(name, directory=True)):
        return None
    what = "SNGP" if role == "Sngp" else "Error"
    return f"{what} model directory not found: {model_hint(name)}"


def check_inputs(cfgs: Iterable[Optional[Dict[str, Any]]]) -> None:
    """
    Check that every dataset file and model of the given configs exists, before anything is loaded

    Without this check a typo in a dataset key or path would only surface when that dataset is
    loaded, e.g. after hours of SNGP training or after all test models have been loaded. Files
    are resolved exactly as the loaders resolve them ([data.Files] first, then the dataset
    registry; models under ./models, else as paths).

    Args:
        cfgs: Validated configs (None entries are skipped):
              - training branch configs (load_sngp_config/load_error_config, one per branch):
                every [data] AddDatasets key and the [baseline] Checkpoint;
              - test and benchmark configs (load_test_config/load_benchmark_config): every
                [data] Datasets key and the models of [models] (lam-turb: [lam.models] and
                [turb.models])

    Raises:
        FileNotFoundError: Listing every missing file
    """
    problems: List[str] = []
    datasets, model_names = set(), set()
    for cfg in cfgs:
        if cfg is None:
            continue
        where = cfg.get("config_file", "<config>") + (f" [{cfg['branch']}]" if "branch" in cfg else "")
        data = cfg["data"]
        for key in data["AddDatasets"] if "AddDatasets" in data else data["Datasets"]:
            datasets.add(key)
            try:
                resolve_dataset_file(key, data["Files"])
            except FileNotFoundError as exc:
                problems.append(f"{where}: {exc}")
        models = []
        if "baseline" in cfg:
            models.append(("[baseline] Checkpoint", "Baseline", cfg["baseline"]["Checkpoint"]))
        tables = [("models", cfg["models"])] if "models" in cfg else \
            [(f"{b}.models", cfg[b]["models"]) for b in BRANCHES if "models" in cfg.get(b, {})]
        for table, entries in tables:
            models += [(f"[{table}] {role}", role, name) for role, name in entries.items()]
        for label, role, name in models:
            model_names.add(name)
            msg = missing_model(role, name)
            if msg is not None:
                problems.append(f"{where}: {label}: {msg}")
    if len(problems) == 1:
        raise FileNotFoundError(problems[0])
    if problems:
        raise FileNotFoundError(f"{len(problems)} input files not found:\n  - " + "\n  - ".join(problems))
    print(f"Input check: {len(datasets)} dataset(s) and {len(model_names)} model(s) found")
