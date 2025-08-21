import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# HIERARCHICAL ERROR MODEL FUNCTIONS

def train_error_model(X_train, y_true, y_pred, X_valid, y_true_valid, y_pred_valid, config, device_name, epochs, error_lr_schedule, sample_weights=None, save_path=None):
    """
    Train a dual-output NLL error model on the residuals, with validation.
    Returns the trained model and normalization stats.
    """

    # Compute error/residual
    error = y_true.flatten() - y_pred.flatten()
    error_valid = y_true_valid.flatten() - y_pred_valid.flatten()

    mean_X = np.mean(X_train, axis=0, keepdims=True)
    std_X = np.std(X_train, axis=0, keepdims=True)
    mean_error = np.mean(error, axis=0, keepdims=True)
    std_error = np.std(error, axis=0, keepdims=True)

    X_norm = (X_train - mean_X) / std_X
    error_norm = (error - mean_error) / std_error
    X_valid_norm = (X_valid - mean_X) / std_X
    error_valid_norm = (error_valid - mean_error) / std_error

    # Build model
    def build_dual_output_nll_model(input_dim, config):
        hidden_layers = config['model'].get('HiddenLayers', [64, 64])
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
        for units in hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(2))
        model.add(tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Normal(
                loc=t[..., :1],
                scale=1e-3 + tf.nn.softplus(t[..., 1:])
            )
        ))
        return model

    def nll_loss(y_true, y_pred):
        return -y_pred.log_prob(y_true)

    with tf.device(device_name):
        error_model = build_dual_output_nll_model(X_train.shape[1], config)
        error_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=nll_loss)
        fit_args = dict(
            epochs=epochs,
            batch_size=50000,
            callbacks=[tf.keras.callbacks.LearningRateScheduler(error_lr_schedule)],
            verbose=1,
            validation_data=(X_valid_norm, error_valid_norm)
        )
        if sample_weights is not None:
            fit_args['sample_weight'] = sample_weights
        error_model.fit(X_norm, error_norm, **fit_args)
        if save_path:
            error_model.save(save_path)
    return error_model, mean_X, std_X, mean_error, std_error

def apply_error_model(error_model, X, mean_X, std_X, mean_error, std_error):
    """
    Apply error model to inputs and return rescaled mean (epistemic) and std (aleatoric) predictions.
    """
    X_norm = (X - mean_X) / std_X
    error_pred_norm = error_model(X_norm)
    error_pred_mean = error_pred_norm.mean().numpy().flatten()
    error_pred_std  = error_pred_norm.stddev().numpy().flatten()
    error_pred_mean = error_pred_mean * std_error + mean_error  # Rescale to original error scale
    error_pred_std  = error_pred_std  * std_error               # Rescale to original
    return error_pred_mean, error_pred_std

#  Function to update sample_weights after each stage
def update_sample_weights(residuals, base_weights=None, threshold=10, low_weight=0.1):
    """
    Update sample weights based on new residuals.
    Assign low_weight to points with abs(residual) < threshold.
    """
    if base_weights is None:
        base_weights = np.ones_like(residuals)
    new_weights = base_weights.copy()
    new_weights[np.abs(residuals) < threshold] *= low_weight
    return new_weights

def hierarchical_error_correction(X_train, y_true, y_pred_init, X_valid, y_true_valid, y_pred_valid_init, config, device_name, epochs, error_lr_schedule, n_stages=2, sample_weights=None):
    """
    Perform n-stage hierarchical error correction with validation.
    Returns list of error models, normalization stats, and final corrected predictions.
    """
    y_pred = y_pred_init.copy()
    y_pred_valid = y_pred_valid_init.copy()
    error_models = []
    norm_stats = []
    for stage in range(n_stages):
        print(f"\n--- Training Error Model Stage {stage+1} ---")
        error_model, mean_X, std_X, mean_error, std_error = train_error_model(
            X_train, y_true, y_pred, X_valid, y_true_valid, y_pred_valid,
            config, device_name, epochs, error_lr_schedule, sample_weights
        )
        error_models.append(error_model)
        norm_stats.append((mean_X, std_X, mean_error, std_error))
        # Apply correction to train and valid
        error_pred, error_pred_std = apply_error_model(error_model, X_train, mean_X, std_X, mean_error, std_error)
        error_pred_valid, _        = apply_error_model(error_model, X_valid, mean_X, std_X, mean_error, std_error)
        y_pred = y_pred + error_pred
        y_pred_valid = y_pred_valid + error_pred_valid
        # Update sample_weights based on new residuals (base_weights=None to purge previous weights and reset to 1)
        residuals = y_true - y_pred
        if sample_weights is not None:
            sample_weights = update_sample_weights(residuals, base_weights=None, threshold=10, low_weight=0.1)
    return error_models, norm_stats, y_pred, error_pred, error_pred_std

# Modified hierarchical_error_inference to return only nth intermediate predictions
def hierarchical_error_inference(error_models, norm_stats, X_test, y_pred_test_init, n):
    """
    Passes X_test and initial predictions through all n error models for correction.
    Returns only the nth intermediate corrected predictions.
    """
    y_pred_test = y_pred_test_init.copy()
    for i, (error_model, (mean_X, std_X, mean_error, std_error)) in enumerate(zip(error_models, norm_stats)):
        error_pred_test, error_pred_std_test = apply_error_model(error_model, X_test, mean_X, std_X, mean_error, std_error)
        y_pred_test = y_pred_test + error_pred_test
        if i == n - 1:
            return y_pred_test, error_pred_test, error_pred_std_test
    return y_pred_test, error_pred_test, error_pred_std_test