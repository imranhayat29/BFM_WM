"""
Train the SNGP confidence model and/or the error model of the turbulent-only BFM wall model

The baseline wall model itself is trained with train_wm_proposal.py; this script takes its checkpoint
([baseline] Checkpoint) as input.

Usage (from the repository root):
    python train_turb.py --sngp sngp/sngp_turb.toml --error error/error_turb.toml [--gpu N] [--smoke]

    --sngp / --error   TOML files resolved under ./inputfiles/ (or ./inputfiles/sngp|error/, or as a path);
                       at least one is required. With both, the baseline checkpoints must be equal; the
                       additional datasets and their split are shared when [data] and Seed agree.
    --gpu N            GPU index for torch and TensorFlow (overrides [general] GpuNum; -1 = CPU)
    --smoke            login-node smoke test: 2 epochs, capped samples, models saved to
                       ./results/smoke/models/ (Overwrite allowed), diagnostics to ./results/smoke/training/

Outputs: <SaveDir>/<ModelName> (SavedModel + assets/), diagnostics in results/training/<ModelName>/.
"""
# Provenance: replaces the training cells of TURB-NB (SNGP: cells 14-24, error model: cells 30-35);
# see docs/DEVELOPMENT.md.
import sys

from src_err import utils  # sets the TensorFlow environment before TF is imported

from src_err.cli import check_inputs, check_pair, load_configs, parse_train_args, print_summary, select_gpu


def main(argv=None) -> int:
    args = parse_train_args(argv, description=__doc__)
    try:
        sngp_cfg, error_cfg = load_configs(args, lamturb=False)
        check_pair(sngp_cfg, error_cfg)
        # Missing datasets/checkpoint: fail now, not after the data loading or the SNGP training
        check_inputs([sngp_cfg, error_cfg])
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not args.smoke:
        from src_err.training import check_outputs
        try:
            check_outputs(sngp_cfg, "sngp")
            check_outputs(error_cfg, "error")
        except FileExistsError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3
    gpu = select_gpu(args, (sngp_cfg, error_cfg))
    print_summary(sngp_cfg, error_cfg, gpu, args.smoke, "train_turb.py: turbulent-only SNGP / error model")

    device = utils.configure_devices(gpu)
    from src_err.training import run_branch  # imports TensorFlow (after the device set-up)

    try:
        run_branch(sngp_cfg, error_cfg, device, smoke=args.smoke, branch_label="turb")
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
