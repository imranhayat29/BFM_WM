"""
Error-model, SNGP-confidence and testing package of BFM_WM_v2.

The parent package `src/` holds the baseline wall model (WallModel). This package adds
the components trained on top of a frozen baseline checkpoint and the shared test pipeline:

- utils.py:       environment, seeds, devices, json and path helpers
- cli.py:         command-line helpers shared by the training and test scripts (incl. input-file checks)
- config.py:      TOML loading and schema validation (SNGP, error model, tests, benchmark)
- data.py:        baseline train/valid split and additional/test dataset loading
- classes.py:     name-based SNGP class bookkeeping (remove, merge, post-hoc groups)
- sngp.py:        SNGP classifier (confidence model)
- error_model.py: probabilistic error model (aleatoric uncertainty)
- training.py:    SNGP and error-model training of one branch (train_turb.py, train_lamturb.py)
- gating.py, lamturb.py, pipeline.py, evaluation.py, selection.py, plotting.py:
                  inference-time blending, laminar/turbulent switch, evaluation and figures
- reporting.py:   test-script summaries, output files and figure sets

utils.py, cli.py and config.py import neither torch nor TensorFlow.

Import `src_err.utils` before anything that imports TensorFlow (it sets TF_USE_LEGACY_KERAS=1).
"""
