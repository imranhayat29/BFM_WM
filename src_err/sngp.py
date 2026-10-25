"""
SNGP flow-type classifier: model, training, save/load and the one-class confidence.

The SNGP (spectral-normalised residual network + random-feature Gaussian-process head) classifies
the InputScaling-2 inputs X = [pi1, pi2, pi3] into the flow types it was trained on. Its
normalised-entropy "one-class" confidence is the CONFIDENCE switch of the BFM wall model.
Ported from BFM_WM/src_err/SNGP.py and the notebook cells (TURB-NB c19-24, LT-NB c24-29).

Changes relative to BFM_WM:
    - pinned layer names that reproduce the Keras auto-names of a first-built model, so the
      CharLES exporter (tools/export_sngp_to_npz.py) also works when several SNGPs are built
      in one process;
    - dropout is applied in training when DropoutActive = true (upstream never applied it,
      because `training` was not propagated); DropoutActive = false keeps the legacy behaviour;
    - architecture hyperparameters, class weights and the LR decay unit are configurable
      (DecayUnit 'epoch' by default; 'step' reproduces the shipped models);
    - prediction always z-scores with the statistics stored in the model metadata;
    - the confidence always uses the un-merged K and K_tr (post-hoc merging never changes it);
    - labels are the argmax over the trained classes only;
    - optional GP-covariance confidence (per-sample predictive variance, diagonal only).

Compatibility contract (CharLES BFMv2 and tools/export_sngp_*):
    - variables input_layer/{kernel,bias}, spectral_normalization[_i]/{kernel,bias},
      random_feature_gaussian_process/{kernel (2x), bias, gp_output_bias, gp_precision_matrix};
    - 3 -> 128 linear (frozen) input layer, 6 residual blocks h + relu(W h + b), 1024 cos random
      features scaled by sqrt(2/1024), no input LayerNorm, kernel scale 1;
    - metadata keys num_classes, num_classes_train, mean_X_train, std_X_train ([[...]]).
"""

import os

# BFM_WM_v2: tf_keras (legacy Keras 2) is required by the tf-models SNGP layers and to reload the
# shipped SavedModels; it must be selected before TensorFlow is first imported (as src_err.utils).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import datetime
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import official.nlp.modeling.layers as nlp_layers
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src_err import utils
from src_err.classes import ClassMap

# Values hard-coded in CharLES BFMv2 (cti_solver_ig_gpu.cu asserts, wm/BfmWM_confidence.hpp)
CHARLES_ARCHITECTURE = {
    'NumLayers': 6,
    'NumHidden': 128,
    'SpecNormBound': 0.9,
    'NumInducing': 1024,
    'GpKernelScale': 1.0,
}
DEFAULT_MODEL_CONFIG = {**CHARLES_ARCHITECTURE, 'Dropout': 0.1, 'DropoutActive': True}
DEFAULT_TRAINING_CONFIG = {
    'BatchSize': 50000,
    'LearningRate': 5e-4,
    'EndLearningRate': 1e-4,
    'DecayUnit': 'epoch',
    'ClassWeights': 'balanced',
}
MEAN_FIELD_FACTOR = math.pi / 8.0   # lambda of the mean-field softmax (also hard-coded in the exporter/CharLES)

METADATA_FILE = 'metadata.json'
HISTORY_FILE = 'history.json'
SPLIT_FILE = 'split_indices.npz'
METADATA_FORMAT_VERSION = 1
LEGACY_METADATA_KEYS = ('num_classes', 'num_classes_train', 'flow_to_label_combined', 'flow_to_label_filt',
                        'labels_to_remove', 'mean_X_train', 'std_X_train')

# Keras auto-names of the first SNGP built in a fresh process; the exporter looks variables up by them
INPUT_LAYER_NAME = 'input_layer'
GP_LAYER_NAME = 'random_feature_gaussian_process'


def spectral_layer_name(i: int) -> str:
    """Name of the i-th residual (spectral-normalised) layer: 'spectral_normalization', then '_1', '_2', ..."""
    return 'spectral_normalization' if i == 0 else f'spectral_normalization_{i}'


#******************************************************************************************************************************#
# Architecture
#******************************************************************************************************************************#

class DeepResNet(tf.keras.Model):
    """Residual MLP: frozen linear input projection, then `num_layers` blocks h = h + Dropout(Dense(h))."""

    def __init__(self, num_classes: int, num_layers: int = 6, num_hidden: int = 128,
                 dropout_rate: float = 0.1, dropout_active: bool = True, **classifier_kwargs):
        """
        Args:
            num_classes: Number of output units
            num_layers: Number of residual blocks
            num_hidden: Hidden width
            dropout_rate: Dropout rate of the residual branches
            dropout_active: Apply dropout in training; False reproduces the shipped models (never applied)
            **classifier_kwargs: Extra keyword arguments of the output layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.dropout_active = dropout_active
        self.classifier_kwargs = classifier_kwargs
        # Fixed random projection (not trained), as in the TF SNGP tutorial
        self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False, name=INPUT_LAYER_NAME)
        self.dense_layers = [self.make_dense_layer(i) for i in range(num_layers)]
        self.dropout_layers = ([tf.keras.layers.Dropout(dropout_rate, name=f'residual_dropout_{i}')
                                for i in range(num_layers)] if dropout_active and dropout_rate > 0 else [])
        self.classifier = self.make_output_layer(num_classes)

    def make_dense_layer(self, i: int) -> tf.keras.layers.Layer:
        """Residual Dense layer of block i."""
        return tf.keras.layers.Dense(self.num_hidden, activation='relu', name=f'residual_dense_{i}')

    def make_output_layer(self, num_classes: int) -> tf.keras.layers.Layer:
        """Output layer."""
        return tf.keras.layers.Dense(num_classes, **self.classifier_kwargs)

    def hidden(self, inputs, training=False):
        """Penultimate activations (input to the output layer)."""
        h = self.input_layer(inputs)
        for i in range(self.num_layers):
            resid = self.dense_layers[i](h, training=training)
            if self.dropout_layers:
                resid = self.dropout_layers[i](resid, training=training)
            h = h + resid
        return h

    def call(self, inputs, training=False):
        return self.classifier(self.hidden(inputs, training=training))


class DeepResNetSNGP(DeepResNet):
    """DeepResNet with spectral-normalised residual layers and a random-feature GP output layer."""

    def __init__(self, spec_norm_bound: float = 0.9, num_inducing: int = 1024, gp_kernel_scale: float = 1.0, **kwargs):
        """
        Args:
            spec_norm_bound: Spectral-norm bound (norm_multiplier) of the residual layers
            num_inducing: Number of random Fourier features of the GP layer
            gp_kernel_scale: RBF length scale of the GP layer (inputs are scaled by 1/sqrt)
            **kwargs: DeepResNet arguments
        """
        self.spec_norm_bound = spec_norm_bound
        self.num_inducing = num_inducing
        self.gp_kernel_scale = gp_kernel_scale
        super().__init__(**kwargs)

    def make_dense_layer(self, i: int) -> tf.keras.layers.Layer:
        dense_layer = super().make_dense_layer(i)
        # Wrapper variables are named after the wrapper, e.g. 'spectral_normalization_3/kernel:0'
        return nlp_layers.SpectralNormalization(dense_layer, norm_multiplier=self.spec_norm_bound,
                                                name=spectral_layer_name(i))

    def make_output_layer(self, num_classes: int) -> tf.keras.layers.Layer:
        # gp_cov_momentum=-1: exact precision matrix (sum over the epoch; reset by ResetCovarianceCallback)
        return nlp_layers.RandomFeatureGaussianProcess(
            units=num_classes,
            num_inducing=self.num_inducing,
            gp_cov_momentum=-1,
            normalize_input=False,
            scale_random_features=True,
            gp_kernel_scale=self.gp_kernel_scale,
            name=GP_LAYER_NAME,
            **self.classifier_kwargs)

    def call(self, inputs, training=False, return_covmat=False):
        # BFM_WM_v2: `training` reaches the residual stack, so dropout is applied in fit when active
        # (upstream: super().call(inputs) always ran the stack with training=False).
        logits, covmat = self.classifier(self.hidden(inputs, training=training), training=training)
        if not training and return_covmat:
            return logits, covmat
        return logits


class ResetCovarianceCallback(tf.keras.callbacks.Callback):
    """Reset the GP precision matrix at the start of every epoch (after the first), so that the saved
    matrix is the exact sum over the last training epoch."""

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0:
            self.model.classifier.reset_covariance_matrix()


class DeepResNetSNGPWithCovReset(DeepResNetSNGP):
    """DeepResNetSNGP whose fit() always adds ResetCovarianceCallback."""

    def fit(self, *args, **kwargs):
        callbacks = list(kwargs.get('callbacks') or [])
        callbacks.append(ResetCovarianceCallback())
        kwargs['callbacks'] = callbacks
        return super().fit(*args, **kwargs)


def check_charles_compatibility(model_cfg: Dict[str, Any]) -> List[str]:
    """Compare the [model] hyperparameters with the values hard-coded in CharLES BFMv2.

    Args:
        model_cfg: [model] table (missing keys take the defaults)

    Returns:
        List of warning messages (empty if compatible); each one is also printed
    """
    warnings = []
    for key, ref in CHARLES_ARCHITECTURE.items():
        val = model_cfg.get(key, ref)
        if float(val) != float(ref):
            warnings.append(f"WARNING: [model] {key} = {val} differs from the CharLES-compatible value {ref} "
                            f"(CharLES BFMv2 hard-codes these; the exported model would not run in CharLES)")
    for w in warnings:
        print(w)
    return warnings


def build_sngp(num_classes: int, model_cfg: Optional[Dict[str, Any]] = None, input_dim: int = 3) -> DeepResNetSNGPWithCovReset:
    """Build an SNGP from the [model] table.

    Args:
        num_classes: Number of GP output units ([classes] NumClasses)
        model_cfg: [model] table (NumLayers, NumHidden, SpecNormBound, NumInducing, GpKernelScale, Dropout, DropoutActive)
        input_dim: Number of input features

    Returns:
        Built (uncompiled) model
    """
    cfg = {**DEFAULT_MODEL_CONFIG, **(model_cfg or {})}
    check_charles_compatibility(cfg)
    model = DeepResNetSNGPWithCovReset(
        num_classes=int(num_classes),
        num_layers=int(cfg['NumLayers']),
        num_hidden=int(cfg['NumHidden']),
        dropout_rate=float(cfg['Dropout']),
        dropout_active=bool(cfg['DropoutActive']),
        spec_norm_bound=float(cfg['SpecNormBound']),
        num_inducing=int(cfg['NumInducing']),
        gp_kernel_scale=float(cfg['GpKernelScale']))
    model.build((None, int(input_dim)))
    return model


#******************************************************************************************************************************#
# Training
#******************************************************************************************************************************#

def compute_norm_stats(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column z-score statistics (population std, as the notebooks), shapes (1, D)."""
    X_train = np.asarray(X_train, dtype=np.float64)
    return np.mean(X_train, axis=0, keepdims=True), np.std(X_train, axis=0, keepdims=True)


def normalize_inputs(X: np.ndarray, mean_X: np.ndarray, std_X: np.ndarray) -> np.ndarray:
    """z-score X with the given statistics (float64 arithmetic, float32 result)."""
    mean_X = np.asarray(mean_X, dtype=np.float64).reshape(1, -1)
    std_X = np.asarray(std_X, dtype=np.float64).reshape(1, -1)
    return ((np.asarray(X, dtype=np.float64) - mean_X) / std_X).astype(np.float32)


def class_weights(labels: np.ndarray, mode: str = 'balanced') -> Optional[Dict[int, float]]:
    """Keras class_weight dict for the training labels.

    Args:
        labels: Integer training labels
        mode: 'balanced' (sklearn n / (K n_c), keyed by label value) or 'none'

    Returns:
        dict label -> weight, or None for 'none'
    """
    if mode == 'none':
        return None
    if mode != 'balanced':
        raise ValueError(f"ClassWeights must be 'balanced' or 'none', got {mode!r}")
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    # Keyed by label value (upstream used enumerate, which assumed contiguous labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def lr_schedule(cfg_training: Dict[str, Any], n_train: int) -> tf.keras.optimizers.schedules.PolynomialDecay:
    """Linear decay LearningRate -> EndLearningRate.

    DecayUnit 'epoch' spreads the decay over all Epochs (decay_steps = Epochs * steps per epoch);
    'step' reproduces the shipped models (decay_steps = Epochs optimiser steps, so the floor is
    reached after Epochs / steps-per-epoch epochs).
    """
    tr = {**DEFAULT_TRAINING_CONFIG, **cfg_training}
    epochs = int(tr['Epochs'])
    steps_per_epoch = int(math.ceil(n_train / int(tr['BatchSize'])))
    if tr['DecayUnit'] == 'epoch':
        decay_steps = epochs * steps_per_epoch
    elif tr['DecayUnit'] == 'step':
        decay_steps = epochs
    else:
        raise ValueError(f"DecayUnit must be 'epoch' or 'step', got {tr['DecayUnit']!r}")
    return tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=float(tr['LearningRate']),
        decay_steps=max(int(decay_steps), 1),
        end_learning_rate=float(tr['EndLearningRate']),
        power=1.0,
        cycle=False)


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Record the learning rate in effect at the start of each epoch as history['lr']."""

    def __init__(self, schedule: tf.keras.optimizers.schedules.LearningRateSchedule):
        super().__init__()
        self.schedule = schedule
        self._lr = float('nan')

    def on_epoch_begin(self, epoch, logs=None):
        self._lr = float(self.schedule(self.model.optimizer.iterations))

    def on_epoch_end(self, epoch, logs=None):
        # Runs before the History callback, which then stores logs['lr']
        if logs is not None:
            logs['lr'] = self._lr


def train_sngp(Xtr: np.ndarray, ytr: np.ndarray, Xva: Optional[np.ndarray], yva: Optional[np.ndarray],
               cfg: Dict[str, Any], verbose: int = 2) -> Tuple[DeepResNetSNGPWithCovReset, Dict[str, List[float]]]:
    """Build and train an SNGP.

    Loss SCCE(from_logits), metric SparseCategoricalAccuracy, Adam with a linear PolynomialDecay. The
    validation data are used for monitoring only.

    Args:
        Xtr: z-scored training inputs, shape (N, D) (the caller stores the statistics in the metadata)
        ytr: Integer training labels in [0, NumClasses)
        Xva: z-scored validation inputs (or None)
        yva: Validation labels (or None)
        cfg: Validated SNGP config ([general] Seed, [classes] NumClasses, [model], [training])
        verbose: Keras fit verbosity (2 = one line per epoch)

    Returns:
        (model, history) with history = Keras history dict plus 'lr' (rate at the start of each epoch)
    """
    tr = {**DEFAULT_TRAINING_CONFIG, **cfg['training']}
    num_classes = int(cfg['classes']['NumClasses'])
    Xtr = np.asarray(Xtr, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)
    n_trained = int(len(np.unique(ytr)))
    if ytr.min() < 0 or ytr.max() >= num_classes:
        raise ValueError(f"Training labels must lie in [0, NumClasses={num_classes}); got [{ytr.min()}, {ytr.max()}]")
    if num_classes <= n_trained:
        # The untrained output units are the "OOD sinks" of the one-class confidence
        raise ValueError(f"NumClasses ({num_classes}) must exceed the number of trained classes ({n_trained})")

    seed = cfg.get('general', {}).get('Seed')
    if seed is not None:
        # TF global seed only (weights, shuffling, dropout); numpy state is left to the caller
        tf.random.set_seed(int(seed))

    model = build_sngp(num_classes, cfg.get('model', {}), input_dim=Xtr.shape[1])
    schedule = lr_schedule(tr, len(Xtr))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=schedule))

    weights = class_weights(ytr, tr['ClassWeights'])
    validation = None
    if Xva is not None and yva is not None and len(Xva) > 0:
        validation = (np.asarray(Xva, dtype=np.float32), np.asarray(yva, dtype=np.int64))

    print('=' * 80)
    print(f"Training SNGP: {len(Xtr)} train / {0 if validation is None else len(validation[0])} valid samples, "
          f"{n_trained} trained classes, NumClasses = {num_classes}")
    print(f"  Epochs = {tr['Epochs']}, BatchSize = {tr['BatchSize']}, LR {tr['LearningRate']} -> {tr['EndLearningRate']} "
          f"(DecayUnit = {tr['DecayUnit']}, decay_steps = {schedule.decay_steps}), ClassWeights = {tr['ClassWeights']}, "
          f"Dropout = {model.dropout_rate if model.dropout_layers else 'off'}")
    print('=' * 80)
    hist = model.fit(Xtr, ytr, batch_size=int(tr['BatchSize']), epochs=int(tr['Epochs']), class_weight=weights,
                     validation_data=validation, shuffle=True, verbose=verbose, callbacks=[LearningRateLogger(schedule)])
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return model, history


#******************************************************************************************************************************#
# Save / load
#******************************************************************************************************************************#

def _name_index(class_map: Any) -> Dict[str, int]:
    """dict name -> label from a dict or a ClassMap-like object (with .index)."""
    index = class_map.index if hasattr(class_map, 'index') and isinstance(class_map.index, dict) else class_map
    return {str(k): int(v) for k, v in dict(index).items()}


def sngp_metadata(*, num_classes: int, class_map_combined: Any, labels_to_remove: Sequence[int],
                  class_map_trained: Any, mean_X: np.ndarray, std_X: np.ndarray,
                  include_extra_data: bool, merge_groups: bool, recompute_norm: bool, downsample: int,
                  cfg: Optional[Dict[str, Any]] = None, baseline_checkpoint: Optional[str] = None,
                  add_datasets: Sequence[str] = (), window: Optional[Dict[str, float]] = None,
                  remove_classes: Sequence[str] = (), merge_groups_def: Optional[Dict[str, List[str]]] = None,
                  display_names: Optional[Dict[str, str]] = None, posthoc_groups: Optional[Dict[str, List[str]]] = None,
                  seed: Optional[int] = None, split_source: Optional[str] = None,
                  n_train: Optional[int] = None, n_valid: Optional[int] = None,
                  class_counts_train: Optional[Dict[str, int]] = None, class_counts_valid: Optional[Dict[str, int]] = None,
                  versions: Optional[Dict[str, Any]] = None, git_commit: Optional[str] = None,
                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble assets/metadata.json: the legacy keys plus the 'bfmwm_v2' block.

    Args:
        num_classes: GP output units
        class_map_combined: name -> label over all combined classes (dict or ClassMap)
        labels_to_remove: Removed labels in the combined numbering
        class_map_trained: name -> label of the trained classes after remove + merge (dict or ClassMap)
        mean_X: SNGP input means, shape (1, D)
        std_X: SNGP input standard deviations, shape (1, D)
        include_extra_data: Additional datasets were used
        merge_groups: Pre-training merge groups were applied
        recompute_norm: Statistics recomputed over (strided baseline + additional) train data
        downsample: Stride of the baseline part
        cfg: Validated SNGP config (training/architecture entries are taken from it)
        baseline_checkpoint: Path of the baseline checkpoint (basename and sha256 are stored)
        add_datasets: Additional dataset keys
        window: {'upy': ..., 'downy': ...} of the additional data
        remove_classes: Removed class names
        merge_groups_def: Pre-training merge groups (name -> member names)
        display_names: Display renames
        posthoc_groups: Post-hoc groups (labels/plots only)
        seed: Seed
        split_source: 'checkpoint' | 'seeded_resplit'
        n_train: Training samples
        n_valid: Validation samples
        class_counts_train: Training samples per trained class
        class_counts_valid: Validation samples per trained class
        versions: Software versions (src_err.utils.software_versions())
        git_commit: Repository commit (src_err.utils.git_commit_hash())
        config: JSON-serialisable config copy (default: cfg)

    Returns:
        Metadata dict
    """
    trained = _name_index(class_map_trained)
    cfg = cfg or {}
    tr = {**DEFAULT_TRAINING_CONFIG, **cfg.get('training', {})}
    arch = {**DEFAULT_MODEL_CONFIG, **cfg.get('model', {})}
    return {
        # Legacy keys (exporters, CharLES tooling and the old loaders)
        'num_classes': int(num_classes),
        'num_classes_train': len(trained),
        'flow_to_label_combined': _name_index(class_map_combined),
        'flow_to_label_filt': trained,
        'labels_to_remove': [int(v) for v in labels_to_remove],
        'mean_X_train': np.asarray(mean_X, dtype=np.float64).reshape(1, -1).tolist(),
        'std_X_train': np.asarray(std_X, dtype=np.float64).reshape(1, -1).tolist(),
        'include_extra_data_flag': int(bool(include_extra_data)),
        'merge_groups_flag': int(bool(merge_groups)),
        'recompute_norm_with_extra_flag': int(bool(recompute_norm)),
        'downsample_sngp_data_factor': int(downsample),
        'bfmwm_v2': {
            'format_version': METADATA_FORMAT_VERSION,
            'config': config if config is not None else cfg,
            'baseline_checkpoint': os.path.basename(baseline_checkpoint) if baseline_checkpoint else None,
            'baseline_checkpoint_sha256': utils.sha256_file(baseline_checkpoint) if baseline_checkpoint else None,
            'add_datasets': list(add_datasets),
            'window': dict(window or {}),
            'remove_classes': list(remove_classes),
            'merge_groups': dict(merge_groups_def or {}),
            'display_names': dict(display_names or {}),
            'posthoc_groups': dict(posthoc_groups or {}),
            'seed': None if seed is None else int(seed),
            'split_source': split_source,
            'n_train': None if n_train is None else int(n_train),
            'n_valid': None if n_valid is None else int(n_valid),
            'class_counts_train': dict(class_counts_train or {}),
            'class_counts_valid': dict(class_counts_valid or {}),
            'training': {
                'Epochs': int(tr.get('Epochs', 0)),
                'BatchSize': int(tr['BatchSize']),
                'LearningRate': float(tr['LearningRate']),
                'EndLearningRate': float(tr['EndLearningRate']),
                'DecayUnit': tr['DecayUnit'],
                'ClassWeights': tr['ClassWeights'],
                'Dropout': float(arch['Dropout']),
                'DropoutActive': bool(arch['DropoutActive']),
            },
            'architecture': {k: arch[k] for k in CHARLES_ARCHITECTURE},
            'versions': versions or {},
            'git_commit': git_commit,
            'created': datetime.datetime.now().isoformat(timespec='seconds'),
        },
    }


def save_sngp(model: tf.keras.Model, path: str, metadata: Dict[str, Any],
              history: Optional[Dict[str, List[float]]] = None,
              split_indices: Optional[Dict[str, Any]] = None, overwrite: bool = False) -> str:
    """Save the SNGP as a SavedModel plus assets/metadata.json, history.json and split_indices.npz.

    Args:
        model: Trained SNGP
        path: Output directory
        metadata: Metadata (see sngp_metadata()); the legacy keys are required
        history: Training history
        split_indices: Arrays for split_indices.npz (None entries are skipped; int64)
        overwrite: Replace an existing directory ([general] Overwrite)

    Returns:
        The model directory
    """
    missing = [k for k in LEGACY_METADATA_KEYS if k not in metadata]
    if missing:
        raise ValueError(f"SNGP metadata lacks the legacy keys {missing} (needed by the exporters and loaders)")
    if int(metadata['num_classes_train']) != len(metadata['flow_to_label_filt']):
        raise ValueError("metadata num_classes_train != len(flow_to_label_filt)")
    utils.prepare_model_dir(path, overwrite, 'SNGP model')
    print('=' * 80)
    model.save(path, save_format='tf')
    assets = os.path.join(path, 'assets')
    os.makedirs(assets, exist_ok=True)
    utils.save_json(os.path.join(assets, METADATA_FILE), metadata)
    if history:
        utils.save_json(os.path.join(assets, HISTORY_FILE), {k: [float(v) for v in vals] for k, vals in history.items()})
    if split_indices:
        arrays = {k: np.asarray(v, dtype=np.int64) for k, v in split_indices.items() if v is not None}
        np.savez(os.path.join(assets, SPLIT_FILE), **arrays)
    print(f"Saved SNGP model to {path} (assets: {', '.join(sorted(os.listdir(assets)))})")
    print('=' * 80)
    return path


@dataclass
class SngpBundle:
    """A loaded SNGP with its metadata.

    Attributes:
        model: Keras model (revived SavedModel); model(Xn, training=False) returns logits (N, num_classes)
        num_classes: GP output units K
        num_classes_train: Trained classes K_tr (columns 0..K_tr-1)
        class_map_trained: ClassMap of the trained classes (from flow_to_label_filt, label order)
        mean_X: SNGP input means, shape (1, D)
        std_X: SNGP input standard deviations, shape (1, D)
        display_names: Display renames (metadata 'bfmwm_v2' block; {} for legacy models)
        posthoc_groups: Post-hoc groups (metadata 'bfmwm_v2' block; {} for legacy models)
        metadata: The full metadata dict
        path: Model directory
    """
    model: Any
    num_classes: int
    num_classes_train: int
    class_map_trained: ClassMap
    mean_X: np.ndarray
    std_X: np.ndarray
    display_names: Dict[str, str] = field(default_factory=dict)
    posthoc_groups: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    path: str = ''

    @property
    def class_names(self) -> List[str]:
        """Trained class names in label order."""
        return list(self.class_map_trained.names)


def load_sngp(path: str) -> SngpBundle:
    """Load an SNGP SavedModel (legacy or v2) and its metadata.

    Args:
        path: Model directory, or a name under ./models

    Returns:
        SngpBundle
    """
    path = utils.resolve_model_path(path, directory=True)
    meta_path = os.path.join(path, 'assets', METADATA_FILE)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"SNGP metadata not found: {meta_path}")
    with open(meta_path) as f:
        meta = json.load(f)
    missing = [k for k in ('num_classes', 'num_classes_train', 'flow_to_label_filt', 'mean_X_train', 'std_X_train') if k not in meta]
    if missing:
        raise ValueError(f"{meta_path} lacks {missing}")

    # compile=False: the optimizer/loss are not needed for inference; no custom_objects, so Keras
    # revives the saved call functions (works for the shipped models without our classes)
    model = tf.keras.models.load_model(path, compile=False)

    trained = sorted(((str(k), int(v)) for k, v in meta['flow_to_label_filt'].items()), key=lambda kv: kv[1])
    num_classes = int(meta['num_classes'])
    num_classes_train = int(meta['num_classes_train'])
    if [v for _, v in trained] != list(range(num_classes_train)):
        raise ValueError(f"{meta_path}: flow_to_label_filt labels {[v for _, v in trained]} are not 0..{num_classes_train - 1}")
    if num_classes_train >= num_classes:
        raise ValueError(f"{meta_path}: num_classes_train ({num_classes_train}) must be < num_classes ({num_classes})")
    v2 = meta.get('bfmwm_v2', {}) or {}
    bundle = SngpBundle(
        model=model,
        num_classes=num_classes,
        num_classes_train=num_classes_train,
        class_map_trained=ClassMap(names=[k for k, _ in trained]),
        mean_X=np.asarray(meta['mean_X_train'], dtype=np.float64).reshape(1, -1),
        std_X=np.asarray(meta['std_X_train'], dtype=np.float64).reshape(1, -1),
        display_names=dict(v2.get('display_names', {}) or {}),
        posthoc_groups={k: list(v) for k, v in (v2.get('posthoc_groups', {}) or {}).items()},
        metadata=meta,
        path=path)
    print(f"Loaded SNGP {path}: K = {num_classes}, K_tr = {num_classes_train} "
          f"({'v2 metadata' if v2 else 'legacy metadata'}); classes: {bundle.class_names}")
    return bundle


#******************************************************************************************************************************#
# Prediction and confidence
#******************************************************************************************************************************#

def predict_logits(bundle: SngpBundle, X_raw: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """SNGP logits for raw inputs.

    The inputs are z-scored with the metadata statistics (bug fix: upstream sometimes used the
    session's baseline statistics). Small batches keep the GP layer's internal B x B covariance cheap.

    Args:
        bundle: Loaded SNGP
        X_raw: Raw inputs (N, D)
        batch_size: Batch size

    Returns:
        Logits (N, num_classes), float32
    """
    Xn = normalize_inputs(X_raw, bundle.mean_X, bundle.std_X)
    out = np.empty((len(Xn), bundle.num_classes), dtype=np.float32)
    for s in range(0, len(Xn), batch_size):
        out[s:s + batch_size] = bundle.model(Xn[s:s + batch_size], training=False).numpy()
    return out


def _hidden(model: Any, x: np.ndarray) -> tf.Tensor:
    """Penultimate activations at inference (works for our classes and revived SavedModels)."""
    h = model.input_layer(x)
    for layer in model.dense_layers:
        h = h + layer(h, training=False)
    return h


def gp_variance(bundle: SngpBundle, X_raw: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Per-sample GP predictive variance v = s * phi^T P^-1 phi (diagonal of the GP covariance).

    P is the trained gp_precision_matrix (ridge s included in its initial value) and phi the random
    features of the penultimate activations. Uses the GP layer's own covariance computation on small
    batches and keeps only the diagonal (never the B x B covariance of a large batch).

    Args:
        bundle: Loaded SNGP
        X_raw: Raw inputs (N, D)
        batch_size: Batch size

    Returns:
        Variances (N,), float64
    """
    Xn = normalize_inputs(X_raw, bundle.mean_X, bundle.std_X)
    var = np.empty(len(Xn), dtype=np.float64)
    for s in range(0, len(Xn), batch_size):
        _, cov = bundle.model.classifier(_hidden(bundle.model, Xn[s:s + batch_size]), training=False)
        var[s:s + batch_size] = np.diag(cov.numpy())
    return var


def probabilities(logits: np.ndarray, variance: Optional[np.ndarray] = None, lambda_: float = MEAN_FIELD_FACTOR) -> np.ndarray:
    """Mean-field posterior probabilities softmax(logits / sqrt(1 + lambda * v)).

    With variance=None, v = 1 (a fixed temperature sqrt(1 + pi/8)); this is what the notebooks and
    CharLES compute.

    Args:
        logits: (N, K)
        variance: Per-sample GP variance (N,) from gp_variance(), or None
        lambda_: Mean-field factor

    Returns:
        Probabilities (N, K), float32
    """
    logits = np.asarray(logits, dtype=np.float32)
    v = np.float32(1.0) if variance is None else np.asarray(variance, dtype=np.float32).reshape(-1, 1)
    z = logits / np.sqrt(np.float32(1.0) + np.float32(lambda_) * v)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def oneclass_confidence(probs: np.ndarray, num_classes_train: int, num_classes: int) -> np.ndarray:
    """Normalised-entropy one-class confidence C = 1 - H(q) / ln(K - K_tr + 1).

    q = [sum_{c < K_tr} p_c, p_{K_tr}, ..., p_{K-1}] and H = -sum q log(q + 1e-8) (float32, as upstream
    and CharLES). Always pass the un-merged probabilities and the model's own K and K_tr
    (bug fix: upstream post-hoc merging changed the normaliser and inflated C).

    Args:
        probs: Un-merged probabilities (N, K)
        num_classes_train: K_tr
        num_classes: K

    Returns:
        Confidence (N,)
    """
    probs = np.asarray(probs, dtype=np.float32)
    if probs.shape[1] != num_classes:
        raise ValueError(f"probs has {probs.shape[1]} columns but num_classes = {num_classes}: "
                         f"the confidence must use the un-merged probabilities")
    q = np.concatenate([probs[:, :num_classes_train].sum(axis=1, keepdims=True), probs[:, num_classes_train:]], axis=1)
    entropy = -np.sum(q * np.log(q + np.float32(1e-8)), axis=1)
    return (np.float32(1.0) - entropy / np.log(np.float32(num_classes - num_classes_train + 1))).astype(np.float32)


def predict_labels(probs: np.ndarray, num_classes_train: int) -> np.ndarray:
    """Predicted labels = argmax over the trained classes only (bug fix: upstream used all K columns,
    so an untrained column could win and collide with other label spaces)."""
    return np.argmax(np.asarray(probs)[:, :num_classes_train], axis=1)


def predict(bundle: SngpBundle, X_raw: np.ndarray, covariance: str = 'none', batch_size: int = 512) -> Dict[str, np.ndarray]:
    """Convenience: logits, probabilities, confidence and labels for raw inputs.

    Args:
        bundle: Loaded SNGP
        X_raw: Raw inputs (N, D)
        covariance: 'none' (= CharLES) or 'gp' (per-sample GP variance in the mean-field softmax)
        batch_size: Batch size

    Returns:
        dict with 'logits', 'probs', 'confidence', 'labels' and, for 'gp', 'variance'
    """
    if covariance not in ('none', 'gp'):
        raise ValueError(f"covariance must be 'none' or 'gp', got {covariance!r}")
    out = {'logits': predict_logits(bundle, X_raw, batch_size)}
    variance = None
    if covariance == 'gp':
        variance = out['variance'] = gp_variance(bundle, X_raw, batch_size)
    out['probs'] = probabilities(out['logits'], variance)
    out['confidence'] = oneclass_confidence(out['probs'], bundle.num_classes_train, bundle.num_classes)
    out['labels'] = predict_labels(out['probs'], bundle.num_classes_train)
    return out


def confusion_matrix_known(y_true: np.ndarray, y_pred: np.ndarray, num_classes_train: int,
                           normalize: Optional[str] = None) -> Tuple[np.ndarray, float]:
    """Confusion matrix over the trained classes.

    Points whose true or predicted label is >= K_tr are dropped (notebook behaviour; with
    predict_labels() only the true labels can be out of range).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes_train: K_tr
        normalize: sklearn normalize option (None, 'true', 'pred', 'all')

    Returns:
        (matrix (K_tr, K_tr), ignored fraction)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    keep = (y_true >= 0) & (y_true < num_classes_train) & (y_pred >= 0) & (y_pred < num_classes_train)
    ignored = 1.0 - float(keep.sum()) / max(len(y_true), 1)
    cm = confusion_matrix(y_true[keep], y_pred[keep], labels=np.arange(num_classes_train), normalize=normalize)
    print(f"Confusion matrix: {100 * ignored:.6f}% of the points ignored (label outside the trained classes)")
    return cm, ignored
