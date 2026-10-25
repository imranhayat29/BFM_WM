"""
Train the SNGP confidence models and/or the error models of the laminar-turbulent BFM wall model

Replaces the training cells of the notebook
`BFMWM_ErrorModel copy_aairfoil2M-LamTurbModel-cleaned4context2.ipynb` (cells 14-42), which had
to be run twice by hand (once per pipeline). Here the lam-turb TOMLs hold a [lam] and a [turb]
table, each a complete turbulent-only-style config (shared [general] values are inherited), and
each selected branch runs the same pipeline as train_turb.py with its own baseline checkpoint:

    lam : laminar baseline (FS_ZPG/APG/FPG), laminar A-airfoil additional data, n1/delta <= 0.8
          (SNGP) / no upper cut (error model), DownsampleBaseline 10, RecomputeNorm, 30 SNGP units;
    turb: as train_turb.py (SNGP 4000 epochs in the shipped lam-turb set).

Kt/Km is not used in training: each branch is trained on its own data only, and the Kt/Km
hard switch between the branches is applied at test time (test_lamturb.py) and in CharLES.

Usage (from the repository root):
    python train_lamturb.py --sngp sngp/sngp_lamturb.toml --error error/error_lamturb.toml \\
        [--branch lam|turb|both] [--gpu N] [--smoke]

    --branch   branch(es) to train (default both; a branch missing from a TOML is skipped for that model)
    --gpu N    GPU index for torch and TensorFlow (overrides [general] GpuNum; -1 = CPU)
    --smoke    login-node smoke test (2 epochs, capped samples, ./results/smoke/)
"""
import sys

from src_err import utils  # sets the TensorFlow environment before TF is imported

from src_err.config import branch_configs
from src_err.cli import check_inputs, check_pair, load_configs, parse_train_args, print_summary, select_gpu


def main(argv=None) -> int:
    args = parse_train_args(argv, description=__doc__, lamturb=True)
    branches = ["lam", "turb"] if args.branch == "both" else [args.branch]
    try:
        sngp_all, error_all = load_configs(args, lamturb=True)
        sngp_b = branch_configs(sngp_all) if sngp_all is not None else {}
        error_b = branch_configs(error_all) if error_all is not None else {}
        jobs = []
        for b in branches:
            s, e = sngp_b.get(b), error_b.get(b)
            if s is None and e is None:
                raise ValueError(f"branch '{b}' is in none of the given TOMLs")
            for kind, given, cfg in (("SNGP", sngp_all, s), ("error-model", error_all, e)):
                if given is not None and cfg is None:
                    print(f"NOTE: the {kind} TOML has no [{b}] table; no {kind} trained for this branch")
            check_pair(s, e, where=f"[{b}] ")
            jobs.append((b, s, e))
        # Missing datasets/checkpoints of any branch: fail now, not after an earlier branch has trained
        check_inputs([c for _, s, e in jobs for c in (s, e)])
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.smoke:
        from src_err.training import check_outputs
        try:
            # Fail before any branch is trained if a later one would refuse to save
            for _, s, e in jobs:
                check_outputs(s, "sngp")
                check_outputs(e, "error")
        except FileExistsError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3

    gpu = select_gpu(args, [c for _, s, e in jobs for c in (s, e)])
    for b, s, e in jobs:
        print_summary(s, e, gpu, args.smoke, f"train_lamturb.py: branch '{b}'")

    device = utils.configure_devices(gpu)
    from src_err.training import run_branch  # imports TensorFlow (after the device set-up)

    results = {}
    for i, (b, s, e) in enumerate(jobs):
        try:
            results[b] = run_branch(s, e, device, smoke=args.smoke, branch_label=b)
        except FileExistsError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3
        if i < len(jobs) - 1:
            # Models are saved; free the TF graph before the next branch (layer names are pinned,
            # so the exporters still find the variables of models built after the reset)
            utils.lazy_tf().keras.backend.clear_session()

    print("=" * 80)
    for b, r in results.items():
        for kind in ("sngp", "error"):
            if r[kind] is not None:
                print(f"[{b}] {kind}: {r[kind]['path'] or '(not saved)'}"
                      + (f", best_epoch {r[kind]['best_epoch']}" if kind == "error" else ""))
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
