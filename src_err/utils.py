"""
Utility functions shared by the src_err package and the training/testing scripts

Importing this module sets the TensorFlow environment variables, so it must be imported
before anything that imports TensorFlow.
"""
import os

# The SNGP and error models are tf_keras (Keras 2) SavedModels; Keras 3 cannot load them.
# Both variables only take effect if set before TensorFlow is first imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

import numpy as np

# torch is imported only where it is needed (set_seeds, configure_devices), so that the config,
# JSON and path helpers stay importable without it (e.g. `tools/apriori_benchmark.py --replot`)
if TYPE_CHECKING:
    import torch

# Device configuration of the current process (TF devices can only be set once)
_DEVICE_STATE: Dict[str, Any] = {"gpu": None, "device": None}


def lazy_tf():
    """
    Import TensorFlow after the environment set-up of this module

    Returns:
        The tensorflow module
    """
    import tensorflow as tf
    return tf


def set_seeds(seed: int) -> None:
    """
    Seed every random number generator used by the workflow

    Args:
        seed: Seed for python `random`, numpy, TensorFlow/Keras and torch (incl. CUDA)
    """
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        tf = lazy_tf()
        # Seeds python, numpy and the TF global generator in one call
        tf.keras.utils.set_random_seed(seed)
    except ImportError as exc:
        print(f"WARNING: TensorFlow not available, TF seed not set ({exc})")
    print(f"Random seeds set to {seed} (random, numpy, torch, tensorflow)")


def configure_devices(gpu: int) -> "torch.device":
    """
    Select the device for torch and TensorFlow

    TensorFlow devices can be configured only once per process (before TF initialises its
    runtime), so repeated calls with the same index are no-ops and a different index raises.

    Args:
        gpu: GPU index used for both frameworks; < 0 selects the CPU (TF GPUs are hidden)

    Returns:
        torch.device for the baseline model
    """
    gpu = int(gpu)
    if _DEVICE_STATE["gpu"] is not None:
        if _DEVICE_STATE["gpu"] != gpu:
            raise RuntimeError(f"Devices already configured for gpu={_DEVICE_STATE['gpu']}; "
                               f"cannot switch to gpu={gpu} in the same process")
        return _DEVICE_STATE["device"]

    import torch

    tf = lazy_tf()
    tf_gpus = tf.config.list_physical_devices("GPU")
    use_gpu = gpu >= 0 and torch.cuda.is_available() and len(tf_gpus) > 0
    if gpu >= 0 and not use_gpu:
        print(f"WARNING: GPU {gpu} requested but no GPU is visible to torch/TF; using the CPU")

    try:
        if use_gpu:
            if gpu >= torch.cuda.device_count():
                raise ValueError(f"PyTorch GPU index {gpu} unavailable; found {torch.cuda.device_count()} device(s)")
            if gpu >= len(tf_gpus):
                raise ValueError(f"TensorFlow GPU index {gpu} unavailable; found {len(tf_gpus)} device(s)")
            tf.config.set_visible_devices(tf_gpus[gpu], "GPU")
            # Torch shares the GPU, so TF must not reserve all of its memory up front
            tf.config.experimental.set_memory_growth(tf_gpus[gpu], True)
            device = torch.device(f"cuda:{gpu}")
        else:
            if tf_gpus:
                tf.config.set_visible_devices([], "GPU")
            device = torch.device("cpu")
    except RuntimeError as exc:
        # Raised by TF if its runtime was initialised before this call
        print(f"WARNING: TensorFlow devices could not be configured ({exc})")
        device = torch.device(f"cuda:{gpu}") if use_gpu else torch.device("cpu")

    _DEVICE_STATE.update(gpu=gpu, device=device)
    print(f"Using device {device} for PyTorch and "
          f"{'GPU ' + str(gpu) if use_gpu else 'CPU'} for TensorFlow")
    return device


def to_jsonable(obj: Any) -> Any:
    """
    Deep-convert an object to JSON-serialisable python types

    Mappings become dicts with str keys, tuples and sets become lists, numpy arrays/scalars
    (and torch tensors, if torch is loaded) become python values. NaN and inf become None,
    because strict JSON has no such values. Any other object is written as its str().

    Args:
        obj: Nested mappings/sequences with possibly numpy values

    Returns:
        A copy made of dict, list, str, int, float, bool and None
    """
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    # Only check for tensors when torch is already loaded: importing it here would defeat the
    # torch-free import of this module
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


def save_json(path: str, obj: Any) -> None:
    """
    Write an object as indented JSON (converted with `to_jsonable`)

    Used for the model metadata and histories. The file has no trailing newline, as the
    files of the shipped models.

    Args:
        path: Output file path (parent directories are created)
        obj: Object to write
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def sha256_file(path: Optional[str]) -> Optional[str]:
    """
    SHA-256 of a file (stored in the model metadata to identify the baseline checkpoint)

    Args:
        path: File path

    Returns:
        The hex digest, or None if `path` is empty or not a file
    """
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_model_dir(path: str, overwrite: bool, what: str) -> None:
    """
    Make room for a new SavedModel directory

    Refuses to overwrite unless asked. When overwriting, the old SavedModel is removed first so
    that no stale assets survive; only directories that look like SavedModels (or are empty)
    are removed.

    Args:
        path: Model directory
        overwrite: [general] Overwrite of the training TOML
        what: Model kind for the messages, e.g. 'SNGP model' or 'error model'

    Raises:
        FileExistsError: `path` exists and may not or cannot be replaced
    """
    if not os.path.exists(path):
        return
    if not overwrite:
        raise FileExistsError(f"{what[:1].upper() + what[1:]} directory already exists: {path} "
                              "(set [general] Overwrite = true to replace it)")
    if not os.path.isdir(path):
        raise FileExistsError(f"Refusing to overwrite {path}: it is a file")
    if os.listdir(path) and not os.path.exists(os.path.join(path, "saved_model.pb")):
        raise FileExistsError(f"Refusing to overwrite {path}: it is not empty and not a SavedModel directory")
    shutil.rmtree(path)
    print(f"Overwriting existing {what} at {path}")


def git_commit_hash(repo_root: str = ".") -> Optional[str]:
    """
    Return the current git commit of the repository

    Args:
        repo_root: Directory inside the repository

    Returns:
        The commit hash (with a '-dirty' suffix if tracked files are modified), or None
    """
    # GIT_OPTIONAL_LOCKS=0: 'git status' must not rewrite the index of the repository
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                                text=True, check=True, timeout=30, env=env).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root,
                                capture_output=True, text=True, check=True, timeout=60, env=env).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{commit}-dirty" if status else commit


def software_versions() -> Dict[str, Optional[str]]:
    """
    Versions of the main packages (read from the package metadata, no TF import)

    Returns:
        Dict package -> version (None if not installed)
    """
    versions: Dict[str, Optional[str]] = {"python": sys.version.split()[0]}
    packages = {"numpy": "numpy", "pandas": "pandas", "torch": "torch", "tensorflow": "tensorflow",
                "tf_keras": "tf-keras", "tfp": "tensorflow-probability"}
    for name, dist in packages.items():
        try:
            versions[name] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def resolve_input_file(arg: str, subdir: Optional[str] = None) -> str:
    """
    Resolve a TOML argument like the parent scripts (paths relative to the repo root)

    Args:
        arg: File name relative to ./inputfiles/ (e.g. 'sngp/sngp_turb.toml') or a path
        subdir: Optional inputfiles subfolder tried as well (e.g. 'sngp' for 'sngp_turb.toml')

    Returns:
        Path of the existing file
    """
    candidates = [os.path.join(".", "inputfiles", arg)]
    if subdir:
        candidates.append(os.path.join(".", "inputfiles", subdir, arg))
    candidates.append(arg)
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Input file '{arg}' not found. Tried: " + ", ".join(candidates))


def resolve_model_path(name_or_path: str, directory: bool = False) -> str:
    """
    Resolve a model name under ./models, falling back to the given path

    Args:
        name_or_path: Model file/directory name in ./models, or a path
        directory: Only a directory under ./models matches (SavedModels); a file of that name
                   is ignored

    Returns:
        './models/<name>' if it exists, else `name_or_path` unchanged
    """
    candidate = os.path.join(".", "models", name_or_path)
    found = os.path.isdir(candidate) if directory else os.path.exists(candidate)
    return candidate if found else name_or_path


def resolve_dataset_file(key: str, files_override: Optional[Mapping[str, str]] = None) -> str:
    """
    h5 file of a dataset key

    Order: explicit override, then the parent `load_external_dataset` lookup
    (./data/<key>_data.h5, registry entry, TBL_<key> for numeric keys, first registry key
    containing `key`).

    Args:
        key: Dataset key
        files_override: Optional {key: path} ([data.Files] of the TOMLs)

    Returns:
        Existing file path
    """
    # Imported here so that importing this module does not depend on the parent registry
    from wall_model_cases import INPUT_TURB_FILES

    tried: List[str] = []
    if files_override and key in files_override:
        path = files_override[key]
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f"Dataset '{key}': file from [data.Files] does not exist: {path}")

    tried.append(f"./data/{key}_data.h5")
    if key in INPUT_TURB_FILES:
        tried.append(INPUT_TURB_FILES[key].replace(".pkl", ".h5"))
    elif key.lstrip("-").isdigit():
        tried.append(INPUT_TURB_FILES.get(f"TBL_{key}", "").replace(".pkl", ".h5"))
    else:
        for reg_key, path in INPUT_TURB_FILES.items():
            if key in reg_key:
                tried.append(path.replace(".pkl", ".h5"))
                break
    for path in tried:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Dataset '{key}' not found. Tried: {', '.join(p for p in tried if p)}"
                            " (set an explicit path in [data.Files])")


def snapshot_dir(cfg: Mapping[str, Any]) -> str:
    """
    Snapshot directory of an error-model training config (the notebooks' models/checkpoints/<name>)

    Args:
        cfg: Error-model branch config ([general] ModelName, SaveDir)

    Returns:
        <SaveDir>/checkpoints/<ModelName>
    """
    general = cfg.get("general", {})
    return os.path.join(general.get("SaveDir", "./models"), "checkpoints", general["ModelName"])


def make_output_dir(path: str) -> str:
    """
    Create an output directory if needed

    Args:
        path: Directory path

    Returns:
        The same path
    """
    os.makedirs(path, exist_ok=True)
    return path
