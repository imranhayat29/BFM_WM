"""
Error model: a dual-output Gaussian NLL regressor of the baseline wall-model error.

The model maps the InputScaling-2 inputs X = [pi1, pi2, pi3] (z-scored) to a Normal
distribution over the z-scored baseline error eps = pi_out,true - pi_out,baseline:
the mean is the epistemic correction and the standard deviation the aleatoric
uncertainty. Ported from BFM_WM/src_err/error_model_functs.py and the notebook cells
(TURB-NB c32-35, LT-NB c38-42/c46); weighting, log inputs, alternative targets,
inner-region mode, hierarchical training and Sobolev regularisation were dropped.

Compatibility contract (CharLES BFMv2 and tools/export_error_model_*):
    - plain Sequential Dense stack (ReLU hidden layers, linear Dense(2)), so the
      exporter can read the raw SavedModel variables;
    - sigma = SigmaFloor + softplus(o1) with SigmaFloor = 1e-4 hard-coded in CharLES;
    - assets/error_norm_stats.json in the legacy format
      {"x_mean": [[...]], "x_std": [[...]], "y_mean": [v], "y_std": [v]}.
"""

import os

# BFM_WM_v2: tf_keras (legacy Keras 2) is required to reload the shipped SavedModels;
# the variable must be set before TensorFlow is first imported (src_err.utils does the same).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import datetime
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src_err import utils

tfd = tfp.distributions

# Values hard-coded in CharLES BFMv2 (wm/BfmWM_error.hpp)
CHARLES_SIGMA_FLOOR = 1e-4
CHARLES_HIDDEN_WIDTH = 60
CHARLES_HIDDEN_DEPTHS = {'turbulent': 15, 'laminar': 6}

NORM_STATS_FILE = 'error_norm_stats.json'
HISTORY_FILE = 'training_history.json'
VAL_LOSS_FILE = 'val_loss.csv'
METADATA_FILE = 'metadata.json'
SPLIT_FILE = 'split_indices.npz'
METADATA_FORMAT_VERSION = 1


@dataclass
class ErrorBundle:
    """A loaded error model with its normalisation statistics.

    Attributes:
        model: Keras model returning a tfd.Normal over the z-scored error
        x_mean: Input means, shape (1, D)
        x_std: Input standard deviations, shape (1, D)
        y_mean: Error mean (scalar)
        y_std: Error standard deviation (scalar)
        metadata: Contents of assets/metadata.json ({} for legacy models)
        path: Model directory
    """
    model: tf.keras.Model
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    path: str = ''


#******************************************************************************************************************************#
# Model definition
#******************************************************************************************************************************#

def nll_loss(y_true, y_pred):
    """Negative log-likelihood of the targets under the predicted distribution.

    The name must stay 'nll_loss': the shipped SavedModels reference it as a custom object.

    Args:
        y_true: Targets, shape (B, 1)
        y_pred: tfd.Normal returned by the model

    Returns:
        Per-sample NLL
    """
    return -y_pred.log_prob(y_true)


def custom_objects() -> Dict[str, Any]:
    """Custom objects needed by tf.keras.models.load_model (same keys as the notebooks)."""
    return {'DistributionLambda': tfp.layers.DistributionLambda,
            'nll_loss': nll_loss,
            'Normal': tfd.Normal}


def check_charles_compatibility(hidden_layers: Sequence[int], sigma_floor: float = CHARLES_SIGMA_FLOOR) -> List[str]:
    """Return warnings for architectures that the CharLES exporter/solver cannot use as is.

    Args:
        hidden_layers: Hidden-layer widths
        sigma_floor: Additive floor of the predicted standard deviation

    Returns:
        List of warning messages (empty if compatible)
    """
    msgs = []
    widths = [int(u) for u in hidden_layers]
    if len(widths) == 0:
        msgs.append("no hidden layers: tools/export_error_model_to_cuda_datafiles.py needs at least one")
    elif len(set(widths)) > 1:
        msgs.append(f"unequal hidden widths {widths}: tools/export_error_model_to_cuda_datafiles.py "
                    f"requires all hidden layers to have the same width")
    if widths and (widths[0] != CHARLES_HIDDEN_WIDTH or len(widths) not in CHARLES_HIDDEN_DEPTHS.values()):
        msgs.append(f"HiddenLayers {len(widths)}x{widths[0]} differ from CharLES BFMv2's hard-coded "
                    f"{CHARLES_HIDDEN_WIDTH} neurons x {CHARLES_HIDDEN_DEPTHS['turbulent']} (turbulent) / "
                    f"{CHARLES_HIDDEN_DEPTHS['laminar']} (laminar) hidden layers")
    if not np.isclose(sigma_floor, CHARLES_SIGMA_FLOOR, rtol=0.0, atol=1e-12):
        msgs.append(f"SigmaFloor {sigma_floor:g} differs from CharLES BFMv2's hard-coded {CHARLES_SIGMA_FLOOR:g}")
    return msgs


def build_error_model(input_dim: int, hidden_layers: Sequence[int], sigma_floor: float = CHARLES_SIGMA_FLOOR) -> tf.keras.Model:
    """Build the dual-output distributional error model (uncompiled).

    Architecture: InputLayer(D) -> Dense(u, relu) per hidden layer -> Dense(2) ->
    Normal(loc=o0, scale=sigma_floor + softplus(o1)).

    Args:
        input_dim: Number of input features D
        hidden_layers: Widths of the hidden layers
        sigma_floor: Additive floor of the standard deviation

    Returns:
        tf.keras.Sequential whose output is a tfd.Normal of batch shape (B, 1)
    """
    for msg in check_charles_compatibility(hidden_layers, sigma_floor):
        print(f"WARNING (error model): {msg}")

    floor = float(sigma_floor)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(int(input_dim),)))
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(int(units), activation='relu'))
    # Linear head: [mu, raw sigma parameter]
    model.add(tf.keras.layers.Dense(2))
    # A lambda (not a registered class) keeps the SavedModel loadable with the legacy custom_objects;
    # the floor is bound as a default argument so the serialised closure is self-contained.
    model.add(
        tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t, floor=floor: tfd.Normal(
                loc=t[..., :1],
                scale=floor + tf.math.softplus(t[..., 1:])
            )
        )
    )
    return model


#******************************************************************************************************************************#
# Normalisation and targets
#******************************************************************************************************************************#

def error_targets(y_true: np.ndarray, y_baseline_raw: np.ndarray) -> np.ndarray:
    """Error-model targets eps = y_true - y_baseline_raw (pi_out units).

    Args:
        y_true: Dataset targets pi_out
        y_baseline_raw: Raw baseline NN outputs (not abs'ed), as in the notebooks

    Returns:
        Flat array of errors
    """
    return np.asarray(y_true, dtype=np.float64).reshape(-1) - np.asarray(y_baseline_raw, dtype=np.float64).reshape(-1)


def compute_norm_stats(X_train: np.ndarray, e_train: np.ndarray) -> Dict[str, np.ndarray]:
    """Z-score statistics in the legacy error_norm_stats.json shapes.

    Args:
        X_train: Error-model training inputs, shape (N, D)
        e_train: Training errors, shape (N,)

    Returns:
        dict with x_mean (1, D), x_std (1, D), y_mean (1,), y_std (1,)
    """
    X = np.asarray(X_train, dtype=np.float64)
    e = np.asarray(e_train, dtype=np.float64).reshape(-1)
    return {'x_mean': np.mean(X, axis=0, keepdims=True),
            'x_std': np.std(X, axis=0, keepdims=True),
            'y_mean': np.mean(e, axis=0, keepdims=True),
            'y_std': np.std(e, axis=0, keepdims=True)}


def normalize_inputs(X: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    """(X - x_mean) / x_std as float32."""
    return ((np.asarray(X, dtype=np.float64) - np.asarray(x_mean).reshape(1, -1))
            / np.asarray(x_std).reshape(1, -1)).astype(np.float32)


def normalize_targets(e: np.ndarray, y_mean, y_std) -> np.ndarray:
    """(e - y_mean) / y_std as float32, flat."""
    return ((np.asarray(e, dtype=np.float64).reshape(-1) - float(np.asarray(y_mean).reshape(-1)[0]))
            / float(np.asarray(y_std).reshape(-1)[0])).astype(np.float32)


#******************************************************************************************************************************#
# Training
#******************************************************************************************************************************#

class SnapshotCallback(tf.keras.callbacks.Callback):
    """Save a full SavedModel every `interval` epochs to <ckpt_dir>/ckpt_XXXX.

    Fixes relative to the old SaveEveryNEpochs: the directory is created only when the first
    snapshot is written, interval <= 0 disables it, and (if given) the normalisation stats are
    written into each snapshot's assets so snapshots are directly exportable/loadable.
    """

    def __init__(self, interval: int, ckpt_dir: Optional[str], norm_stats: Optional[Dict[str, Any]] = None):
        """
        Args:
            interval: Snapshot period in epochs (<= 0 disables)
            ckpt_dir: Directory receiving ckpt_XXXX snapshots
            norm_stats: Optional stats written as assets/error_norm_stats.json in each snapshot
        """
        super().__init__()
        self.interval = int(interval)
        self.ckpt_dir = ckpt_dir
        self.norm_stats = norm_stats
        self.saved: List[str] = []

    @property
    def enabled(self) -> bool:
        return self.interval > 0 and bool(self.ckpt_dir)

    def on_epoch_end(self, epoch, logs=None):
        if not self.enabled:
            return
        e = int(epoch) + 1
        if e % self.interval != 0:
            return
        filepath = os.path.join(self.ckpt_dir, f"ckpt_{e:04d}")
        try:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.model.save(filepath)
            if self.norm_stats is not None:
                _write_norm_stats(filepath, self.norm_stats)
            self.saved.append(filepath)
            print(f"[Snapshot] Saved epoch {e} to {filepath}")
        except Exception as ex:
            # A failed snapshot must not kill a long training run
            print(f"[Snapshot] Failed to save epoch {e} to {filepath}: {ex}")


class SmoothedBestWeights(tf.keras.callbacks.Callback):
    """Keep the weights of the epoch where an exponential moving average of the monitored loss is lowest.

    The validation NLL of the error model is spiky, so its raw minimum is close to a chance pick;
    the EMA (span `span` epochs, alpha = 2/(span+1)) prefers epochs after which the loss stayed low.
    Non-finite losses are skipped. The selected weights are restored at the end of training.
    """

    def __init__(self, monitor: str, span: int):
        """
        Args:
            monitor: Logged quantity ('val_loss' or 'loss')
            span: EMA span in epochs (> 0)
        """
        super().__init__()
        self.monitor = monitor
        self.alpha = 2.0 / (float(span) + 1.0)
        self.ema = None
        self.best = np.inf
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        value = float((logs or {}).get(self.monitor, np.nan))
        if not np.isfinite(value):
            return
        self.ema = value if self.ema is None else self.alpha * value + (1.0 - self.alpha) * self.ema
        if self.ema < self.best:
            self.best, self.best_epoch = self.ema, int(epoch) + 1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def lr_schedule(cfg_training: Dict[str, Any]) -> tf.keras.optimizers.schedules.PolynomialDecay:
    """Per-epoch learning-rate schedule.

    Linear decay LearningRate -> EndLearningRate over Epochs*DecayFraction epochs, then constant
    (the notebooks' PolynomialDecay(1e-3, epochs/1.5, 8e-5) evaluated per epoch).

    Args:
        cfg_training: The [training] table

    Returns:
        PolynomialDecay evaluated with the 0-based epoch index
    """
    epochs = int(cfg_training['Epochs'])
    frac = float(cfg_training.get('DecayFraction', 2.0 / 3.0))
    return tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=float(cfg_training.get('LearningRate', 1e-3)),
        decay_steps=max(epochs * frac, 1.0),
        end_learning_rate=float(cfg_training.get('EndLearningRate', 8e-5)),
        power=1.0,
        cycle=False,
    )


def train_error_model(Xtr_n: np.ndarray, etr_n: np.ndarray,
                      Xva_n: Optional[np.ndarray], eva_n: Optional[np.ndarray],
                      cfg: Dict[str, Any], snapshot_dir: Optional[str] = None,
                      norm_stats: Optional[Dict[str, Any]] = None,
                      verbose: int = 2) -> Tuple[tf.keras.Model, Dict[str, List[float]], int]:
    """Build and train the error model on normalised data.

    Callbacks: per-epoch LearningRateScheduler, EarlyStopping on val_loss (restores the best
    weights at the end of training even without an early stop in tf_keras 2.19; this is how
    the shipped 'ep3811' model was produced) and optional SavedModel snapshots.

    Args:
        Xtr_n: Normalised training inputs, shape (N, D)
        etr_n: Normalised training errors, shape (N,)
        Xva_n: Normalised validation inputs (None: monitor the training loss)
        eva_n: Normalised validation errors
        cfg: Validated error-model config ([model], [training] incl. EarlyStopping/Snapshots)
        snapshot_dir: Snapshot directory (None: <SaveDir>/checkpoints/<ModelName>, or disabled
            if [general] Save is false)
        norm_stats: Optional stats copied into each snapshot's assets
        verbose: Keras fit verbosity (2 = one line per epoch)

    Returns:
        (model with the restored weights, history dict with loss/val_loss/lr, best_epoch (1-based))
    """
    model_cfg = cfg.get('model', {})
    tr = cfg['training']
    es_cfg = tr.get('EarlyStopping', {})
    snap_cfg = tr.get('Snapshots', {})

    epochs = int(tr['Epochs'])
    batch_size = int(tr.get('BatchSize', 50000))
    hidden = [int(u) for u in model_cfg['HiddenLayers']]
    sigma_floor = float(model_cfg.get('SigmaFloor', CHARLES_SIGMA_FLOOR))
    has_valid = Xva_n is not None and eva_n is not None and len(Xva_n) > 0

    Xtr = np.asarray(Xtr_n, dtype=np.float32)
    # Explicit (N, 1) targets: a flat (N,) target would broadcast against the (N, 1) Normal
    ytr = np.asarray(etr_n, dtype=np.float32).reshape(-1, 1)
    validation_data = None
    if has_valid:
        validation_data = (np.asarray(Xva_n, dtype=np.float32), np.asarray(eva_n, dtype=np.float32).reshape(-1, 1))
    monitor = 'val_loss' if has_valid else 'loss'

    schedule = lr_schedule(tr)
    patience = int(es_cfg.get('Patience', 0))
    patience = patience if patience > 0 else epochs
    restore = bool(es_cfg.get('RestoreBestWeights', True))
    smooth = int(es_cfg.get('SmoothEpochs', 0)) if restore else 0  # > 0: select on the EMA of the monitored loss

    if snapshot_dir is None and cfg.get('general', {}).get('Save', True) and 'ModelName' in cfg.get('general', {}):
        snapshot_dir = utils.snapshot_dir(cfg)
    snapshot = SnapshotCallback(int(snap_cfg.get('SaveInterval', 0)), snapshot_dir, norm_stats)

    print('=' * 80)
    print("Training error model")
    print(f"  HiddenLayers: {len(hidden)}x{hidden[0] if hidden else 0} {hidden}, SigmaFloor: {sigma_floor:g}")
    print(f"  Train: {len(Xtr)} samples, valid: {len(validation_data[0]) if has_valid else 0} samples")
    print(f"  Epochs: {epochs}, BatchSize: {batch_size}")
    print(f"  LR: {tr.get('LearningRate', 1e-3):g} -> {tr.get('EndLearningRate', 8e-5):g} "
          f"over {float(schedule.decay_steps):g} epochs (DecayFraction {tr.get('DecayFraction', 2.0 / 3.0):g}), then constant")
    print(f"  EarlyStopping: monitor={monitor}, patience={patience}, min_delta={float(es_cfg.get('MinDelta', 0.0)):g}, "
          f"restore_best_weights={restore}" + (f", selection on the {smooth}-epoch EMA of {monitor}" if smooth else ""))
    print(f"  Snapshots: " + (f"every {snapshot.interval} epochs to {snapshot_dir}" if snapshot.enabled else "off"))
    print('=' * 80)

    model = build_error_model(Xtr.shape[1], hidden, sigma_floor)
    model.compile(optimizer=tf.keras.optimizers.Adam(float(tr.get('LearningRate', 1e-3))), loss=nll_loss)

    early = tf.keras.callbacks.EarlyStopping(
        monitor=monitor, mode='min', patience=patience,
        min_delta=float(es_cfg.get('MinDelta', 0.0)),
        restore_best_weights=restore and not smooth, verbose=1,
    )
    callbacks = [tf.keras.callbacks.LearningRateScheduler(lambda e, lr: float(schedule(e))), early]
    smoothed = SmoothedBestWeights(monitor, smooth) if smooth else None
    if smoothed is not None:
        callbacks.append(smoothed)
    if snapshot.enabled:
        callbacks.append(snapshot)

    hist = model.fit(Xtr, ytr, batch_size=batch_size, epochs=epochs, validation_data=validation_data,
                     shuffle=True, callbacks=callbacks, verbose=verbose)

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    epochs_run = len(history.get('loss', []))
    if 'lr' not in history:
        history['lr'] = [float(schedule(e)) for e in range(epochs_run)]
    # EarlyStopping.best_epoch is the 0-based index of the best monitored epoch
    best_epoch = smoothed.best_epoch if smoothed is not None else int(early.best_epoch) + 1
    best_val = history.get(monitor, [float('nan')])[best_epoch - 1] if epochs_run else float('nan')
    print(f"Error model: {epochs_run} epochs run, " + (f"lowest {smooth}-epoch EMA of {monitor} {smoothed.best:.6g} "
          f"({monitor} {best_val:.6g}) at epoch {best_epoch}" if smoothed is not None
          else f"best {monitor} {best_val:.6g} at epoch {best_epoch}")
          + (" (weights restored)" if restore else " (last-epoch weights kept)"))
    return model, history, best_epoch


#******************************************************************************************************************************#
# Save / load
#******************************************************************************************************************************#

def _write_norm_stats(model_dir: str, norm_stats: Dict[str, Any]) -> str:
    """Write assets/error_norm_stats.json in exactly the legacy format.

    x_mean/x_std are nested [[...]] (shape (1, D)); y_mean/y_std are one-element lists. The
    exporters, CharLES tooling and the expert script parse this file.
    """
    assets = os.path.join(model_dir, 'assets')
    os.makedirs(assets, exist_ok=True)
    meta = {
        'x_mean': np.asarray(norm_stats['x_mean'], dtype=np.float64).reshape(1, -1).tolist(),
        'x_std': np.asarray(norm_stats['x_std'], dtype=np.float64).reshape(1, -1).tolist(),
        'y_mean': np.asarray(norm_stats['y_mean'], dtype=np.float64).reshape(-1).tolist(),
        'y_std': np.asarray(norm_stats['y_std'], dtype=np.float64).reshape(-1).tolist(),
    }
    if len(meta['y_mean']) != 1 or len(meta['y_std']) != 1:
        raise ValueError("y_mean/y_std must be scalars (the CharLES exporter requires it)")
    path = os.path.join(assets, NORM_STATS_FILE)
    with open(path, 'w') as f:
        # Compact, as the legacy save_error_model_metadata wrote it
        json.dump(meta, f)
    return path


def save_error_model(model: tf.keras.Model, path: str, norm_stats: Dict[str, Any],
                     history: Optional[Dict[str, List[float]]] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     split_indices: Optional[Dict[str, Optional[np.ndarray]]] = None,
                     overwrite: bool = False) -> str:
    """Save the error model as a SavedModel plus its assets.

    Writes, under <path>/assets/: error_norm_stats.json (legacy format), training_history.json
    (loss, val_loss, lr), val_loss.csv (epoch,val_loss; 1-based), metadata.json and
    split_indices.npz (None entries are skipped).

    Args:
        model: Trained Keras model
        path: Output directory
        norm_stats: dict with x_mean, x_std, y_mean, y_std
        history: Training history (from train_error_model)
        metadata: Model metadata (see error_metadata())
        split_indices: Arrays for split_indices.npz
        overwrite: Replace an existing directory ([general] Overwrite)

    Returns:
        The model directory
    """
    utils.prepare_model_dir(path, overwrite, 'error model')
    print('=' * 80)
    model.save(path)
    assets = os.path.join(path, 'assets')
    os.makedirs(assets, exist_ok=True)
    _write_norm_stats(path, norm_stats)

    if history:
        hist = {k: [float(v) for v in vals] for k, vals in history.items()}
        utils.save_json(os.path.join(assets, HISTORY_FILE), hist)
        if 'val_loss' in hist:
            with open(os.path.join(assets, VAL_LOSS_FILE), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['epoch', 'val_loss'])
                for i, v in enumerate(hist['val_loss']):
                    w.writerow([i + 1, v])

    if metadata is not None:
        meta = dict(metadata)
        meta.setdefault('format_version', METADATA_FORMAT_VERSION)
        utils.save_json(os.path.join(assets, METADATA_FILE), meta)

    if split_indices:
        arrays = {k: np.asarray(v, dtype=np.int64) for k, v in split_indices.items() if v is not None}
        np.savez(os.path.join(assets, SPLIT_FILE), **arrays)

    print(f"Saved error model to {path} (assets: {', '.join(sorted(os.listdir(assets)))})")
    print('=' * 80)
    return path


def load_norm_stats(path: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Read assets/error_norm_stats.json.

    Args:
        path: Model directory

    Returns:
        (x_mean (1, D), x_std (1, D), y_mean, y_std)
    """
    stats_path = os.path.join(path, 'assets', NORM_STATS_FILE)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Normalisation stats not found: {stats_path} "
                                f"(models without them cannot be used: their stats are not reproducible)")
    with open(stats_path) as f:
        meta = json.load(f)
    x_mean = np.asarray(meta['x_mean'], dtype=np.float64).reshape(1, -1)
    x_std = np.asarray(meta['x_std'], dtype=np.float64).reshape(1, -1)
    y_mean = np.asarray(meta['y_mean'], dtype=np.float64).reshape(-1)
    y_std = np.asarray(meta['y_std'], dtype=np.float64).reshape(-1)
    if y_mean.size != 1 or y_std.size != 1:
        raise ValueError(f"Expected scalar y_mean/y_std in {stats_path}")
    return x_mean, x_std, float(y_mean[0]), float(y_std[0])


def load_error_model(path: str) -> ErrorBundle:
    """Load an error model (new or legacy) with its normalisation stats.

    Uses compile=False and the legacy custom objects; the DistributionLambda's function is
    cloudpickled Python 3.11 bytecode, so the same Python minor version is required.

    Args:
        path: Model directory: a path relative to ./models (e.g. error/<name>) or any path

    Returns:
        ErrorBundle
    """
    path = utils.resolve_model_path(path, directory=True)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Error model directory not found: {path}")
    model = tf.keras.models.load_model(path, custom_objects=custom_objects(), compile=False)
    x_mean, x_std, y_mean, y_std = load_norm_stats(path)
    meta_path = os.path.join(path, 'assets', METADATA_FILE)
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    hidden = [layer.units for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)][:-1]
    print(f"Loaded error model {path}: hidden {len(hidden)}x{hidden[0] if hidden else 0}, "
          f"x_mean {np.array2string(x_mean.ravel(), precision=4)}, y_mean {y_mean:.6g}, y_std {y_std:.6g}"
          + ("" if metadata else " (legacy: no metadata.json)"))
    return ErrorBundle(model=model, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                       metadata=metadata, path=path)


#******************************************************************************************************************************#
# Prediction
#******************************************************************************************************************************#

def predict_error(bundle: ErrorBundle, X_raw: np.ndarray, batch_size: int = 65536) -> Tuple[np.ndarray, np.ndarray]:
    """Predicted error mean and standard deviation in pi_out units.

    Args:
        bundle: Loaded error model
        X_raw: Raw inputs [pi1, pi2, pi3], shape (N, D)
        batch_size: Rows per forward pass

    Returns:
        (mu, sigma), each shape (N,): mu = mean_n*y_std + y_mean, sigma = std_n*y_std
    """
    X_raw = np.asarray(X_raw)
    n = X_raw.shape[0]
    mu = np.empty(n, dtype=np.float64)
    sigma = np.empty(n, dtype=np.float64)
    bs = max(int(batch_size), 1)
    for s in range(0, n, bs):
        Xn = normalize_inputs(X_raw[s:s + bs], bundle.x_mean, bundle.x_std)
        dist = bundle.model(Xn, training=False)
        mu[s:s + bs] = dist.mean().numpy().reshape(-1)
        sigma[s:s + bs] = dist.stddev().numpy().reshape(-1)
    return mu * bundle.y_std + bundle.y_mean, sigma * bundle.y_std


#******************************************************************************************************************************#
# Metadata (assets/metadata.json)
#******************************************************************************************************************************#

def error_metadata(cfg: Dict[str, Any], *, baseline_checkpoint: str, add_datasets: Sequence[str],
                   window: Dict[str, float], seed: int, split_source: str, n_train: int, n_valid: int,
                   best_epoch: int, epochs_run: int, versions: Optional[Dict[str, Any]] = None,
                   git_commit: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble assets/metadata.json for a trained error model.

    Args:
        cfg: Validated error-model config
        baseline_checkpoint: Path of the baseline checkpoint (basename and sha256 are stored)
        add_datasets: Additional dataset keys
        window: {'upy': ..., 'downy': ...} of the additional data
        seed: Seed of the splits/weights
        split_source: 'checkpoint' | 'seeded_resplit' (baseline split origin)
        n_train: Training samples
        n_valid: Validation samples
        best_epoch: Best epoch (1-based)
        epochs_run: Epochs actually run
        versions: Software versions (src_err.utils.software_versions())
        git_commit: Repository commit (src_err.utils.git_commit_hash())
        config: JSON-serialisable config copy (default: cfg itself)

    Returns:
        Metadata dict
    """
    tr = cfg.get('training', {})
    es = tr.get('EarlyStopping', {})
    model_cfg = cfg.get('model', {})
    return {
        'format_version': METADATA_FORMAT_VERSION,
        'config': config if config is not None else cfg,
        'baseline_checkpoint': os.path.basename(baseline_checkpoint),
        'baseline_checkpoint_sha256': utils.sha256_file(baseline_checkpoint),
        'add_datasets': list(add_datasets),
        'window': dict(window),
        'seed': int(seed),
        'split_source': split_source,
        'n_train': int(n_train),
        'n_valid': int(n_valid),
        'hidden_layers': [int(u) for u in model_cfg.get('HiddenLayers', [])],
        'sigma_floor': float(model_cfg.get('SigmaFloor', CHARLES_SIGMA_FLOOR)),
        'training': {
            'Epochs': int(tr.get('Epochs', 0)),
            'BatchSize': int(tr.get('BatchSize', 50000)),
            'LearningRate': float(tr.get('LearningRate', 1e-3)),
            'EndLearningRate': float(tr.get('EndLearningRate', 8e-5)),
            'DecayFraction': float(tr.get('DecayFraction', 2.0 / 3.0)),
            'RestoreBestWeights': bool(es.get('RestoreBestWeights', True)),
            'Patience': int(es.get('Patience', 0)),
            'MinDelta': float(es.get('MinDelta', 0.0)),
            'SmoothEpochs': int(es.get('SmoothEpochs', 0)),
            'SaveInterval': int(tr.get('Snapshots', {}).get('SaveInterval', 0)),
        },
        'best_epoch': int(best_epoch),
        'epochs_run': int(epochs_run),
        'versions': versions or {},
        'git_commit': git_commit,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }
