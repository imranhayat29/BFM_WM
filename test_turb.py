"""
A-priori test of the turbulent BFM wall model with error model, SNGP confidence and EQWM fallback

    python test_turb.py test/test_turb.toml [--gpu N] [--output-dir DIR] [--max-samples N] [--show-cf-stations]

Runs the baseline, the error model and the SNGP on the test datasets, blends the corrected
prediction with the baseline and the EQWM fallback (CharLES BFMv2 3-switch blend, inference
quantities only) and evaluates everything against the dataset targets. Writes the figures
of [plots], summary.json and (optionally) arrays.npz to the output directory.
Run from the repository root (TOMLs are resolved under ./inputfiles/, models under ./models/).
"""
import sys
import time

import src_err.utils as utils  # sets the TensorFlow environment before TF is imported

from src_err.cli import apply_plot_flags, check_inputs, parse_test_args, select_gpu

import matplotlib

matplotlib.use("Agg")


def main(argv=None) -> None:
    args = parse_test_args(argv, description=__doc__.split("\n\n")[0], example="test/test_turb.toml")
    from src_err.config import load_test_config
    try:
        cfg = load_test_config(args.config, "turb")
        # Missing datasets/models: fail before the models are loaded
        check_inputs([cfg])
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"ERROR: {exc}")
    apply_plot_flags(args, cfg)
    from src_err.data import load_datasets, to_input_frame
    from src_err.pipeline import class_names_for_plot, load_branch_models, predict_turb
    from src_err.reporting import build_test_summary, make_test_figures, print_test_config, save_test_outputs

    gpu = select_gpu(args, [cfg])
    out_dir = cfg["general"]["OutputDir"] if args.output_dir is None else args.output_dir
    max_samples = cfg["data"]["MaxSamples"] if args.max_samples is None else args.max_samples
    print_test_config(cfg, out_dir, gpu, max_samples)
    timings = {}
    t0 = time.time()

    device = utils.configure_devices(gpu)
    models = load_branch_models(cfg["models"], cfg["classes"], device, branch="turb")
    timings["load_models"] = time.time() - t0

    t = time.time()
    d = cfg["data"]
    print("=" * 80)
    arrays = load_datasets(d["Datasets"], models.baseline.config, d["upy"], d["downy"],
                           files_override=d["Files"], stride=d["Stride"], max_samples=max_samples)
    arrays = to_input_frame(arrays, d["InputFrame"])
    timings["load_data"] = time.time() - t

    t = time.time()
    print("=" * 80)
    out = predict_turb(models, arrays, cfg)
    timings["predict"] = time.time() - t

    utils.make_output_dir(out_dir)
    class_names = class_names_for_plot(models)
    labels = out["labels_posthoc"]
    t = time.time()
    figures = make_test_figures(out, arrays, class_names, labels, cfg, out_dir)
    timings["figures"] = time.time() - t
    timings["total"] = time.time() - t0

    summary = build_test_summary(out, arrays, cfg, {"turb": models}, class_names, labels,
                                 cli={"config": args.config, "gpu": gpu, "output_dir": out_dir,
                                      "max_samples": max_samples, "show_cf_stations": args.show_cf_stations},
                                 figures=figures, timings=timings)
    save_test_outputs(out, arrays, summary, class_names, cfg, out_dir)
    print("=" * 80)
    print(f"Done in {timings['total']:.1f} s; outputs in {out_dir}")


if __name__ == "__main__":
    main()
