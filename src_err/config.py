"""
TOML configuration loading and validation for the src_err workflow

Every TOML is read with the parent `parsing_wm.parse_toml_config` (which stores
`config['config_file']`) and validated against the schemas below (`sngp_schema`,
`error_schema`, `test_schema`, `benchmark_schema`):
defaults are filled in, unknown keys raise a ValueError with a "did you mean" suggestion,
missing required keys raise, basic types and enum values are checked. Before that, each
loader checks with `check_kind` that the TOML is of the kind (and variant) it expects, so that
a TOML given to the wrong script or option gets one message naming the right one.

Lam-turb training TOMLs hold a `[lam]` and/or `[turb]` table, each a complete
turbulent-only-style config, plus an optional shared `[general]` table whose values are
inherited by the branches that don't set them (see `branch_configs`).
"""
import copy
import difflib
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import toml

from parsing_wm import parse_toml_config
from src_err.utils import resolve_input_file, to_jsonable

# This module must stay free of torch/TensorFlow imports (src_err.utils is), so that
# `test_turb_allcases.py --replot` can validate a TOML without loading either.

# ---------------------------------------------------------------------------------------
# Schema building blocks
# ---------------------------------------------------------------------------------------

REQUIRED = object()  # sentinel: key has no default


class Key:
    """
    Schema entry of a leaf value

    Args:
        kind: one of 'int', 'float', 'bool', 'str', 'list[str]', 'list[int]', 'list[float]',
              'map[str]' (name -> str) or 'map[list[str]]' (name -> list of names)
        default: default value, or REQUIRED
        enum: allowed values (for 'str', or for the items of 'list[str]')
        check: optional callable(value) -> error message or None
        nullable: if True, None (the default) means "not set"
    """

    def __init__(self, kind: str, default: Any = REQUIRED, enum: Optional[List[Any]] = None,
                 check: Optional[Callable[[Any], Optional[str]]] = None, nullable: bool = False):
        self.kind = kind
        self.default = default
        self.enum = enum
        self.check = check
        self.nullable = nullable


def _positive(v: Any) -> Optional[str]:
    return None if v > 0 else "must be > 0"


def _non_negative(v: Any) -> Optional[str]:
    return None if v >= 0 else "must be >= 0"


def _fraction(v: Any) -> Optional[str]:
    return None if 0.0 < v <= 1.0 else "must be in (0, 1]"


def _pair(v: List[float]) -> Optional[str]:
    return None if len(v) == 2 else "must have 2 entries [min, max]"


def _pair_or_empty(v: List[float]) -> Optional[str]:
    return None if len(v) in (0, 2) else "must be [] or [min, max]"


def _positive_items(v: List[int]) -> Optional[str]:
    if len(v) == 0:
        return "must not be empty"
    return None if all(x > 0 for x in v) else "entries must be > 0"


def _ramp(threshold: float, delta: float, enum_type: Optional[List[str]] = None,
          type_default: Optional[str] = None) -> Dict[str, Any]:
    """Schema of a smooth-ramp gate {Threshold, Delta} (inline table in the TOMLs)."""
    table = {"Threshold": Key("float", threshold),
             "Delta": Key("float", delta, check=_non_negative)}
    if enum_type is not None:
        table = {"Type": Key("str", type_default, enum=enum_type), **table}
    return table


# Aleatoric gate defaults of the 'ratio' metric sigma/|mu| (notebooks: 1.0 +- 0.5);
# used when Type = "ratio" is given without Threshold/Delta.
ALEATORIC_RATIO_DEFAULTS = {"Threshold": 1.0, "Delta": 0.5}

# Reference models, relative to ./models (baseline/, sngp/ and error/ hold the three kinds of model)
SHIPPED_TURB_BASELINE = "baseline/cleaned_up_20251208_ymax_0_15_ymin_0_0025.pth"
SHIPPED_LAM_BASELINE = ("baseline/NN_wm_CH0_G0_S0_tn560665_vn140167_fds0_lds0_customw1_inputs2_final_"
                        "ep30000_tl0.00032144_vl0.00031670.pth")
SHIPPED_TURB_SNGP = "sngp/sngp_model_Baseline_cleanedupNew_Add_ExpertExtFlow_ep4000_merged"
SHIPPED_LAM_SNGP = "sngp/sngp_model_Baseline_selftrained_lam_Add_ExpertExtFlowUPY0p8_ep20000_merged"
SHIPPED_TURB_ERROR = "error/error_model_Baseline_cleanedupNew_Add_ExpertExtFlow_CustomLayers_N60xL15_wVal_ep3811"
SHIPPED_LAM_ERROR = "error/error_model_Baseline_selftrained_lam_Add_ExpertExtFlowUPY1_CustomLayers_N60xL6_wVal_ep4000"

FIGURE_FORMATS = ["png", "pdf", "svg", "jpg", "eps"]
# PDF estimators of src_err/plotting.py ('hist' is an alias of 'histogram')
PDF_METHODS = ["kde", "histogram", "hist", "both"]

# ---------------------------------------------------------------------------------------
# Training schemas (sngp/*.toml, error/*.toml)
# ---------------------------------------------------------------------------------------


def _training_general_schema(kind: str) -> Dict[str, Any]:
    # Trained models go next to the reference models of their kind: ./models/sngp or ./models/error
    return {
        "ModelName": Key("str"),
        "SaveDir": Key("str", f"./models/{kind}"),
        "Save": Key("bool", True),
        "Overwrite": Key("bool", False),
        "GpuNum": Key("int", 0),
        "Seed": Key("int"),
    }


def _training_data_schema() -> Dict[str, Any]:
    return {
        "AddDatasets": Key("list[str]", []),
        "TrainRatio": Key("float", 0.8, check=_fraction),
        "upy": Key("float", 0.15, check=_positive),
        "downy": Key("float", 0.0, check=_non_negative),
        "Files": Key("map[str]", {}),
    }


def sngp_schema() -> Dict[str, Any]:
    """Schema of a turbulent-only SNGP TOML (and of each branch of sngp_lamturb.toml)."""
    data = _training_data_schema()
    data.update({
        "DownsampleBaseline": Key("int", 1, check=_positive),
        "RecomputeNorm": Key("bool", False),
    })
    return {
        "general": _training_general_schema("sngp"),
        "baseline": {"Checkpoint": Key("str")},
        "data": data,
        "classes": {
            "NumClasses": Key("int", 50, check=_positive),
            "RemoveClasses": Key("list[str]", []),
            "MergeGroups": Key("map[list[str]]", {}),
            "DisplayNames": Key("map[str]", {}),
            "PosthocGroups": Key("map[list[str]]", {}),
        },
        "model": {
            "NumLayers": Key("int", 6, check=_positive),
            "NumHidden": Key("int", 128, check=_positive),
            "SpecNormBound": Key("float", 0.9, check=_positive),
            "NumInducing": Key("int", 1024, check=_positive),
            "GpKernelScale": Key("float", 1.0, check=_positive),
            "Dropout": Key("float", 0.1, check=lambda v: None if 0.0 <= v < 1.0 else "must be in [0, 1)"),
            "DropoutActive": Key("bool", True),
        },
        "training": {
            "Epochs": Key("int", check=_positive),
            "BatchSize": Key("int", 50000, check=_positive),
            "LearningRate": Key("float", 5e-4, check=_positive),
            "EndLearningRate": Key("float", 1e-4, check=_non_negative),
            "DecayUnit": Key("str", "epoch", enum=["epoch", "step"]),
            "ClassWeights": Key("str", "balanced", enum=["balanced", "none"]),
        },
    }


def error_schema() -> Dict[str, Any]:
    """Schema of a turbulent-only error-model TOML (and of each branch of error_lamturb.toml)."""
    return {
        "general": _training_general_schema("error"),
        "baseline": {"Checkpoint": Key("str")},
        "data": _training_data_schema(),
        "model": {
            "HiddenLayers": Key("list[int]", check=_positive_items),
            "SigmaFloor": Key("float", 1.0e-4, check=_positive),
        },
        "training": {
            "Epochs": Key("int", check=_positive),
            "BatchSize": Key("int", 50000, check=_positive),
            "LearningRate": Key("float", 1e-3, check=_positive),
            "EndLearningRate": Key("float", 8e-5, check=_non_negative),
            "DecayFraction": Key("float", 2.0 / 3.0, check=_fraction),
            "EarlyStopping": {
                "RestoreBestWeights": Key("bool", True),
                "Patience": Key("int", 0, check=_non_negative),
                "MinDelta": Key("float", 0.0, check=_non_negative),
            },
            "Snapshots": {
                "SaveInterval": Key("int", 1000),
            },
        },
    }


# ---------------------------------------------------------------------------------------
# Test schemas (test/test_turb.toml, test/test_lamturb.toml) and the all-cases test schema (test/test_turb_allcases.toml)
# ---------------------------------------------------------------------------------------


def _gates_schema(variant: str) -> Dict[str, Any]:
    lamturb = variant == "lamturb"
    return {
        "Aleatoric": _ramp(15.0 if lamturb else 10.0, 5.0, enum_type=["percentage", "ratio"],
                           type_default="percentage"),
        "Confidence": _ramp(0.85 if lamturb else 0.7, 0.1 if lamturb else 0.05),
        "Yplus": _ramp(4.0, 2.0),
        "LowValue": _ramp(2.0, 1.0),
    }


def _models_schema(branch: str) -> Dict[str, Any]:
    if branch == "lam":
        return {"Baseline": Key("str", SHIPPED_LAM_BASELINE), "Sngp": Key("str", SHIPPED_LAM_SNGP),
                "Error": Key("str", SHIPPED_LAM_ERROR)}
    return {"Baseline": Key("str", SHIPPED_TURB_BASELINE), "Sngp": Key("str", SHIPPED_TURB_SNGP),
            "Error": Key("str", SHIPPED_TURB_ERROR)}


def _test_classes_schema() -> Dict[str, Any]:
    # None = take the post-hoc groups / display names from the SNGP metadata
    return {"PosthocGroups": Key("map[list[str]]", None, nullable=True),
            "DisplayNames": Key("map[str]", None, nullable=True)}


def test_schema(variant: str) -> Dict[str, Any]:
    """
    Schema of test/test_turb.toml or test/test_lamturb.toml

    Args:
        variant: 'turb' or 'lamturb' (the defaults follow TURB-NB resp. LT-NB)

    Returns:
        Schema dict
    """
    if variant not in ("turb", "lamturb"):
        raise ValueError(f"variant must be 'turb' or 'lamturb', got '{variant}'")
    lamturb = variant == "lamturb"
    schema: Dict[str, Any] = {
        "general": {
            "OutputDir": Key("str", "results/test_lamturb" if lamturb else "results/test_turb_gaussian_2M"),
            "Formats": Key("list[str]", ["png", "pdf"], enum=FIGURE_FORMATS),
            "GpuNum": Key("int", 0),
            "SaveArrays": Key("bool", False),
        },
        "data": {
            "Datasets": Key("list[str]",
                            ["aairfoil_2M_full2Dfield_down0_up1_cfgauss_w30_tr3_xalltoall"] if lamturb else
                            ["gaussian_2M_APG", "gaussian_2M_FPG", "gaussian_2M_SEP", "gaussian_2M_MAPG"],
                            check=lambda v: None if v else "must not be empty"),
            "upy": Key("float", 1.0 if lamturb else 0.15, check=_positive),
            "downy": Key("float", 0.0, check=_non_negative),
            "Stride": Key("int", 2, check=_positive),
            "MaxSamples": Key("int", 0, check=_non_negative),
            "Files": Key("map[str]", {}),
        },
        "eqwm": {"Law": Key("str", "charles", enum=["charles", "parent"])},
        "confidence": {"Covariance": Key("str", "none", enum=["none", "gp"])},
        "gates": _gates_schema(variant),
        "metrics": {"LowValue": _ramp(2.0, 1.0)},
        "plots": {
            "SpatialMaps": Key("bool", True),
            "CfLines": Key("bool", True),
            "Pdfs": Key("bool", True),
            "Tripcolor": Key("bool", True),
            "maps": {
                "YAxis": Key("str", "n1_over_delta", enum=["n1_over_delta", "n1"]),
                "YLim": Key("list[float]", [0.0, 0.3] if lamturb else [0.0, 0.2], check=_pair),
                "RawN1YLim": Key("list[float]", [0.0, 0.005], check=_pair),
                "VMin": Key("float", 1.0, check=_positive),
                "VMax": Key("float", 100.0, check=_positive),
                "MarkerSize": Key("float", 10.0, check=_positive),
                "ShowSelected": Key("bool", True),
                "XLim": Key("list[float]", [], check=_pair_or_empty),  # [] = data range
            },
            "cf_lines": {
                "Selection": Key("str", "raw_piecewise" if lamturb else "raw",
                                 enum=["raw", "normalized", "yplus", "raw_piecewise"]),
                "Value": Key("float", 0.001, check=_positive),
                "Upstream": Key("float", 0.0002, check=_positive),
                "Downstream": Key("float", 0.001, check=_positive),
                "XTransition": Key("float", 0.05),
                "RampWidth": Key("float", 0.1, check=_non_negative),
                "MovingAverage": Key("int", 20 if lamturb else 50, check=_non_negative),
                "XLim": Key("list[float]", [-0.01, 1.0] if lamturb else [-0.7, 0.41], check=_pair_or_empty),
                "YLim": Key("list[float]", [], check=_pair_or_empty),  # [] = bottom at 0, top automatic
                "TruthMarkers": Key("int", 40, check=_non_negative),
                "TruthMarkerDx": Key("float", 0.0, check=_non_negative),  # > 0: one marker every dx in x
            },
            "pdfs": {
                "Method": Key("str", "kde", enum=PDF_METHODS),
                "XRange": Key("list[float]", [0.0, 100.0] if lamturb else [0.0, 40.0], check=_pair),
                "ConfRange": Key("list[float]", [0.7, 1.0], check=_pair),
                "Bins": Key("int", 50, check=_positive),
            },
            "tripcolor": {
                "YLim": Key("list[float]", [0.01, 0.2], check=_pair),
                "GridNx": Key("int", 600, check=_positive),
                "GridNy": Key("int", 150, check=_positive),
                "Sigma": Key("float", 3.0, check=_non_negative),
                "XLim": Key("list[float]", [0.0, 1.0] if lamturb else [], check=_pair_or_empty),  # [] = data range
                "VMin": Key("float", 1.0, check=_positive),
                "VMax": Key("float", 100.0, check=_positive),
                "ShowSelected": Key("bool", True),
            },
        },
    }
    if lamturb:
        schema["switch"] = {"KtKmThreshold": Key("float", 0.002)}
        schema["lam"] = {"models": _models_schema("lam"), "classes": _test_classes_schema()}
        schema["turb"] = {"models": _models_schema("turb"), "classes": _test_classes_schema()}
    else:
        schema["models"] = _models_schema("turb")
        schema["classes"] = _test_classes_schema()
    return schema


BENCHMARK_FIGURES = ["train_test", "train_test_zoom", "sigma", "confidence", "pdf", "pdf_always_vs_blended"]


def benchmark_schema() -> Dict[str, Any]:
    """Schema of test/test_turb_allcases.toml (test_turb_allcases.py; see the README)."""
    return {
        "general": {
            "RunName": Key("str", "test_turb_allcases"),
            "OutputDir": Key("str", "results/test_turb_allcases"),
            "Formats": Key("list[str]", ["pdf", "png"], enum=FIGURE_FORMATS),
            "GpuNum": Key("int", 0),
        },
        "models": _models_schema("turb"),
        "data": {
            # The 75 EVB keys live in the shipped TOML; no implicit default list
            "Datasets": Key("list[str]", check=lambda v: None if v else "must not be empty"),
            "TrainingDatasets": Key("list[str]", []),
            "upy": Key("float", 0.15, check=_positive),
            "downy": Key("float", 0.0025, check=_non_negative),
            "Stride": Key("int", 1, check=_positive),
            "MaxSamplesPerDataset": Key("int", 0, check=_non_negative),
            "Files": Key("map[str]", {}),
        },
        "groups": Key("map[list[str]]", {}),
        "display_names": Key("map[str]", {}),
        "eqwm": {"Law": Key("str", "charles", enum=["charles", "parent"])},
        "confidence": {"Covariance": Key("str", "none", enum=["none", "gp"])},
        "gates": _gates_schema("turb"),
        "metrics": {"LowValue": _ramp(2.0, 1.0)},
        "evaluation": {
            "DomainFilter": Key("str", "retau", enum=["retau", "piout", "none"]),
            "RetauMin": Key("float", 5.0, check=_non_negative),
            "PioutMin": Key("float", 2.0, check=_non_negative),
            "OutlierPercent": Key("float", 10000.0, check=_non_negative),
            "MinErrorChangePercent": Key("float", 0.0, check=_non_negative),
            "CompareAlways": Key("bool", True),
        },
        "plots": {
            "Scale": Key("str", "both", enum=["linear", "log", "both"]),
            "LinearLimits": Key("list[float]", [-0.5, 50.0], check=_pair),
            "ZoomLimits": Key("list[float]", [0.0, 10.0], check=_pair),
            "LogLimits": Key("list[float]", [0.1, 100.0], check=_pair),
            "LogYLimits": Key("list[float]", [], check=_pair_or_empty),
            "YLabel": Key("str", "Expert Model Error (%)"),
            "Legend": Key("bool", True),
            "SigmaColorRange": Key("list[float]", [0.0, 30.0], check=_pair),
            "ConfidenceColorRange": Key("list[float]", [0.8, 1.0], check=_pair),
            "Figures": Key("list[str]", list(BENCHMARK_FIGURES), enum=BENCHMARK_FIGURES),
            "PdfMethod": Key("str", "kde", enum=PDF_METHODS),
            "PdfErrorRange": Key("list[float]", [0.0, 40.0], check=_pair),
            "PdfConfRange": Key("list[float]", [0.7, 1.0], check=_pair),
            "PdfBins": Key("int", 50, check=_positive),
            "PdfCurves": Key("list[str]", ["baseline", "final"], enum=["baseline", "final", "eqwm", "always"]),
            "PdfSubset": Key("str", "all", enum=["all", "train", "test"]),
        },
    }


# ---------------------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------------------

_KIND_NAMES = {"int": "an integer", "float": "a number", "bool": "a boolean (true/false)", "str": "a string",
               "list[str]": "a list of strings", "list[int]": "a list of integers",
               "list[float]": "a list of numbers", "map[str]": "a table of name = \"string\"",
               "map[list[str]]": "a table of name = [\"name\", ...]"}


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return (_is_int(v) or isinstance(v, float)) and math.isfinite(float(v))


def _convert(value: Any, key: Key) -> Any:
    """Type-check and normalise a leaf value; raises TypeError with a short reason."""
    kind = key.kind
    if kind == "int":
        if not _is_int(value):
            raise TypeError
        return value
    if kind == "float":
        if not _is_number(value):
            raise TypeError
        return float(value)
    if kind == "bool":
        if not isinstance(value, bool):
            raise TypeError
        return value
    if kind == "str":
        if not isinstance(value, str):
            raise TypeError
        return value
    if kind.startswith("list["):
        if not isinstance(value, list):
            raise TypeError
        item = kind[5:-1]
        if item == "str" and all(isinstance(v, str) for v in value):
            return list(value)
        if item == "int" and all(_is_int(v) for v in value):
            return list(value)
        if item == "float" and all(_is_number(v) for v in value):
            return [float(v) for v in value]
        raise TypeError
    if kind == "map[str]":
        if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
            raise TypeError
        return {str(k): v for k, v in value.items()}
    if kind == "map[list[str]]":
        if not isinstance(value, dict):
            raise TypeError
        for v in value.values():
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise TypeError
        return {str(k): list(v) for k, v in value.items()}
    raise ValueError(f"Unknown schema kind '{kind}'")


def _suggest(name: str, candidates: List[str]) -> str:
    """'did you mean' hint (case-insensitive fuzzy match)."""
    lower = {c.lower(): c for c in candidates}
    match = difflib.get_close_matches(name.lower(), list(lower), n=1, cutoff=0.6)
    if match:
        return f" Did you mean '{lower[match[0]]}'?"
    return f" Valid keys: {', '.join(sorted(candidates))}."


def _validate_table(raw: Any, schema: Dict[str, Any], path: str, errors: List[str]) -> Dict[str, Any]:
    """Recursively validate a table and fill defaults; problems are appended to `errors`."""
    if not isinstance(raw, dict):
        errors.append(f"[{path}] must be a table, got {type(raw).__name__}")
        raw = {}
    out: Dict[str, Any] = {}
    for name in raw:
        if name not in schema:
            where = f"[{path}]" if path else "top level"
            errors.append(f"Unknown key '{name}' in {where}.{_suggest(name, list(schema))}")
    for name, spec in schema.items():
        full = f"{path}.{name}" if path else name
        if isinstance(spec, dict):
            out[name] = _validate_table(raw.get(name, {}), spec, full, errors)
            continue
        if name not in raw:
            if spec.default is REQUIRED:
                errors.append(f"Missing required key '{full}' ({_KIND_NAMES[spec.kind]})")
            else:
                out[name] = copy.deepcopy(spec.default)
            continue
        try:
            value = _convert(raw[name], spec)
        except TypeError:
            errors.append(f"'{full}' must be {_KIND_NAMES[spec.kind]}, got {raw[name]!r}")
            continue
        if spec.enum is not None:
            items = value if isinstance(value, list) else [value]
            bad = [v for v in items if v not in spec.enum]
            if bad:
                errors.append(f"'{full}' has invalid value(s) {bad}; allowed: {spec.enum}")
                continue
        if spec.check is not None:
            msg = spec.check(value)
            if msg:
                errors.append(f"'{full}' = {value!r} {msg}")
                continue
        out[name] = value
    return out


def _raise_if(errors: List[str], config_file: str) -> None:
    if errors:
        raise ValueError(f"Invalid configuration '{config_file}':\n  - " + "\n  - ".join(errors))


def validate(raw: Dict[str, Any], schema: Dict[str, Any], config_file: str = "<dict>") -> Dict[str, Any]:
    """
    Validate a raw config dict against a schema

    Args:
        raw: Parsed TOML (the 'config_file' entry, if any, is kept)
        schema: Schema dict (nested tables of Key entries)
        config_file: Name used in error messages

    Returns:
        A new dict with defaults filled in
    """
    raw = dict(raw)
    cfg_file = raw.pop("config_file", None)
    errors: List[str] = []
    cfg = _validate_table(raw, schema, "", errors)
    _raise_if(errors, config_file)
    if cfg_file is not None:
        cfg["config_file"] = cfg_file
    return cfg


def _load_raw(arg: str, subdir: str) -> Dict[str, Any]:
    path = resolve_input_file(arg, subdir=subdir)
    print(f"Loading configuration from {path}")
    try:
        return parse_toml_config(path)
    except toml.TomlDecodeError as exc:
        # e.g. mixed int/float arrays, which the 'toml' package rejects: write [0.0, 1.0]
        raise ValueError(f"TOML syntax error in '{path}': {exc}") from exc


# ---------------------------------------------------------------------------------------
# Semantic checks (after schema validation)
# ---------------------------------------------------------------------------------------


def _check_groups(groups: Dict[str, List[str]], where: str, errors: List[str]) -> None:
    """Groups must be non-empty and disjoint (name validity is checked against the classes later)."""
    seen: Dict[str, str] = {}
    for gname, members in groups.items():
        if not members:
            errors.append(f"{where}: group '{gname}' is empty")
        for m in members:
            if m in seen:
                errors.append(f"{where}: class '{m}' appears in groups '{seen[m]}' and '{gname}'")
            seen[m] = gname


def _check_sngp_branch(cfg: Dict[str, Any], where: str, errors: List[str]) -> None:
    classes = cfg["classes"]
    _check_groups(classes["MergeGroups"], f"{where}classes.MergeGroups", errors)
    _check_groups(classes["PosthocGroups"], f"{where}classes.PosthocGroups", errors)
    overlap = set(classes["RemoveClasses"]) & {m for v in classes["MergeGroups"].values() for m in v}
    if overlap:
        errors.append(f"{where}classes: {sorted(overlap)} are both removed and merged")
    if len(set(classes["RemoveClasses"])) != len(classes["RemoveClasses"]):
        errors.append(f"{where}classes.RemoveClasses has duplicates")
    if cfg["training"]["EndLearningRate"] > cfg["training"]["LearningRate"]:
        errors.append(f"{where}training: EndLearningRate > LearningRate")


def _check_error_branch(cfg: Dict[str, Any], where: str, errors: List[str]) -> None:
    if cfg["training"]["EndLearningRate"] > cfg["training"]["LearningRate"]:
        errors.append(f"{where}training: EndLearningRate > LearningRate")


def _apply_aleatoric_type_defaults(raw_gates: Any) -> None:
    """With Type = 'ratio' and no explicit Threshold/Delta use the ratio defaults (not the % ones)."""
    if not isinstance(raw_gates, dict):
        return
    ale = raw_gates.get("Aleatoric")
    if isinstance(ale, dict) and ale.get("Type") == "ratio":
        for k, v in ALEATORIC_RATIO_DEFAULTS.items():
            ale.setdefault(k, v)


def _check_gates(cfg: Dict[str, Any], errors: List[str]) -> None:
    conf = cfg["gates"]["Confidence"]
    if not 0.0 <= conf["Threshold"] <= 1.0:
        errors.append("gates.Confidence.Threshold must be in [0, 1]")


# ---------------------------------------------------------------------------------------
# TOML kind detection
# ---------------------------------------------------------------------------------------

BRANCHES = ("lam", "turb")

# (article, name) of each TOML kind in messages
_KIND_LABELS = {"sngp": ("an", "SNGP"), "error": ("an", "error-model"), "test": ("a", "test"),
                "benchmark": ("an", "all-cases test"), "baseline": ("a", "baseline wall-model")}


def is_lamturb(cfg: Dict[str, Any]) -> bool:
    """True if a training config holds [lam]/[turb] branch tables."""
    return any(b in cfg for b in BRANCHES)


def _subtable(table: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = table.get(name)
    return value if isinstance(value, dict) else {}


def _test_variant(raw: Dict[str, Any]) -> Optional[bool]:
    """True: lam-turb test TOML ([lam]/[turb]/[switch]), False: turbulent-only ([models]/[classes]), None: unclear."""
    lam = is_lamturb(raw) or "switch" in raw
    turb = bool({"models", "classes"} & set(raw))
    return lam if lam != turb else None


def sniff_kind(raw: Dict[str, Any]) -> Tuple[Optional[str], Optional[bool]]:
    """
    Guess the kind of a parsed TOML from tables and keys that only that kind has

    Markers:
        'benchmark': [evaluation], [groups], [display_names], general.RunName,
                     data.TrainingDatasets or data.MaxSamplesPerDataset
        'test':      [switch], [lam.models]/[turb.models], general.SaveArrays, data.MaxSamples or
                     the [plots] subtables maps/cf_lines/pdfs/tripcolor (test and benchmark TOMLs
                     share [models], [gates], [eqwm], [confidence] and [metrics], which therefore
                     do not count)
        'baseline':  model.inputs/outputs, training.optimizer/scheduler or data.partition
                     (train_wm_proposal.py TOMLs)
        'error':     model.HiddenLayers, model.SigmaFloor or training.EarlyStopping, unless a
                     baseline marker is present (baseline TOMLs have model.HiddenLayers too)
        'sngp':      model.NumLayers or classes.NumClasses/RemoveClasses/MergeGroups (test TOMLs
                     have a [classes] table too, but only with PosthocGroups/DisplayNames)
    The training markers are also looked for inside the [lam]/[turb] tables of lam-turb TOMLs.

    Args:
        raw: Parsed TOML

    Returns:
        (kind, lamturb). kind is None if no marker or markers of several kinds are found
        (ambiguous). lamturb: for 'sngp'/'error' whether the file has [lam]/[turb] tables, for
        'test' see _test_variant, None for the other kinds.
    """
    raw = {k: v for k, v in raw.items() if k != "config_file"}
    general, data, plots = (_subtable(raw, n) for n in ("general", "data", "plots"))
    found = set()
    if {"evaluation", "groups", "display_names"} & set(raw) or "RunName" in general \
            or {"TrainingDatasets", "MaxSamplesPerDataset"} & set(data):
        found.add("benchmark")
    if "switch" in raw or any("models" in _subtable(raw, b) for b in BRANCHES) or "SaveArrays" in general \
            or "MaxSamples" in data or {"maps", "cf_lines", "pdfs", "tripcolor"} & set(plots):
        found.add("test")
    for table in [raw] + [_subtable(raw, b) for b in BRANCHES]:
        model, training = _subtable(table, "model"), _subtable(table, "training")
        if {"inputs", "outputs"} & set(model) or {"optimizer", "scheduler"} & set(training) \
                or "partition" in _subtable(table, "data"):
            found.add("baseline")
        elif {"HiddenLayers", "SigmaFloor"} & set(model) or "EarlyStopping" in training:
            found.add("error")
        if "NumLayers" in model or {"NumClasses", "RemoveClasses", "MergeGroups"} & set(_subtable(table, "classes")):
            found.add("sngp")
    if len(found) != 1:
        return None, None
    kind = found.pop()
    if kind in ("sngp", "error"):
        return kind, is_lamturb(raw)
    if kind == "test":
        return kind, _test_variant(raw)
    return kind, None


def _variant_text(kind: str, lamturb: Optional[bool]) -> str:
    """'an SNGP', 'a lam-turb test', 'a turbulent-only error-model', ... (+ ' TOML' by the caller)."""
    article, name = _KIND_LABELS[kind]
    if lamturb is None:
        return f"{article} {name}"
    return f"{'a lam-turb' if lamturb else 'a turbulent-only'} {name}"


def _usage_text(kind: str, lamturb: Optional[bool], same_script: bool) -> str:
    """How a TOML of this kind is used, e.g. 'pass it with --error' or 'use test_turb.py'."""
    if kind in ("sngp", "error"):
        script = "train_lamturb.py" if lamturb else "train_turb.py"
        return f"pass it with --{kind}" if same_script else f"pass it to {script} with --{kind}"
    if kind == "test":
        if lamturb is None:
            return "use test_turb.py or test_lamturb.py"
        return "use test_lamturb.py" if lamturb else "use test_turb.py"
    if kind == "benchmark":
        return "use test_turb_allcases.py"
    return "train the baseline with train_wm_proposal.py"


def check_kind(raw: Dict[str, Any], config_file: str, expected: str, lamturb: Optional[bool] = None) -> None:
    """
    Raise one clear error if a TOML is of another kind or variant than its loader expects

    Without this, e.g. an error-model TOML given with --sngp would be reported as a list of
    unknown keys with misleading 'did you mean' hints. If neither the kind nor the variant can
    be told (see sniff_kind), nothing is raised and the schema validation reports the problems.

    Args:
        raw: Parsed TOML
        config_file: TOML path for the message
        expected: 'sngp', 'error', 'test' or 'benchmark'
        lamturb: Expected variant (True: lam-turb, False: turbulent-only, None: either)

    Raises:
        ValueError: The TOML is of another kind, or of the other variant
    """
    kind, variant = sniff_kind(raw)
    training = expected in ("sngp", "error")
    if kind is not None and kind != expected:
        same_script = training and kind in ("sngp", "error") and variant == lamturb
        # Within the right script only the option is wrong: the variant needs no mention
        raise ValueError(f"'{config_file}' looks like {_variant_text(kind, None if same_script else variant)} "
                         f"TOML; {_usage_text(kind, variant, same_script)}")
    # The variant can be told from the [lam]/[turb] tables even when the kind cannot
    if training:
        variant = is_lamturb(raw)
    elif expected == "test":
        variant = _test_variant(raw)
    if lamturb is None or variant is None or variant == lamturb:
        return
    if training:
        reason = "it has [lam]/[turb] tables" if variant else "it has no [lam]/[turb] tables"
    else:
        reason = "it has [lam]/[turb]/[switch] tables" if variant else "it has a top-level [models]/[classes]"
    if kind is None:
        # Kind unknown: name only the variant and the script
        what = "a lam-turb" if variant else "a turbulent-only"
        if training:
            usage = "use train_lamturb.py" if variant else "use train_turb.py"
        else:
            usage = _usage_text("test", variant, False)
    else:
        what, usage = _variant_text(kind, variant), _usage_text(kind, variant, False)
    raise ValueError(f"'{config_file}' is {what} TOML ({reason}); {usage}")


# ---------------------------------------------------------------------------------------
# Training configs (turbulent-only or lam-turb with [lam]/[turb])
# ---------------------------------------------------------------------------------------


def _load_training_config(arg: str, subdir: str, schema_fn: Callable[[], Dict[str, Any]],
                          check_fn: Callable[[Dict[str, Any], str, List[str]], None],
                          lamturb: Optional[bool]) -> Dict[str, Any]:
    raw = _load_raw(arg, subdir)
    config_file = raw.pop("config_file")
    check_kind(raw, config_file, subdir, lamturb)  # the inputfiles subfolder names the kind: 'sngp' or 'error'
    schema = schema_fn()
    errors: List[str] = []

    if not is_lamturb(raw):
        cfg = _validate_table(raw, schema, "", errors)
        _raise_if(errors, config_file)
        check_fn(cfg, "", errors)
        _raise_if(errors, config_file)
        cfg["config_file"] = config_file
        return cfg

    # Lam-turb: top level may only hold the shared [general] and the branch tables
    for name in raw:
        if name not in ("general",) + BRANCHES:
            hint = (" Section tables must be inside [lam]/[turb] in a lam-turb TOML."
                    if name in schema else _suggest(name, ["general", *BRANCHES]))
            errors.append(f"Unknown top-level key '{name}'.{hint}")
    shared = raw.get("general", {})
    if not isinstance(shared, dict):
        errors.append("[general] must be a table")
        shared = {}
    for name in shared:
        if name not in schema["general"]:
            errors.append(f"Unknown key '{name}' in [general].{_suggest(name, list(schema['general']))}")
    if "ModelName" in shared:
        errors.append("ModelName must be set per branch ([lam.general]/[turb.general]), not in the shared [general]")
        shared = {k: v for k, v in shared.items() if k != "ModelName"}

    cfg: Dict[str, Any] = {"general": dict(shared)}
    for branch in BRANCHES:
        if branch not in raw:
            continue
        braw = copy.deepcopy(raw[branch])
        if not isinstance(braw, dict):
            errors.append(f"[{branch}] must be a table")
            continue
        # Shared [general] values are inherited where the branch doesn't set them
        bgen = braw.setdefault("general", {})
        if isinstance(bgen, dict):
            for k, v in shared.items():
                bgen.setdefault(k, copy.deepcopy(v))
        n_err = len(errors)
        bcfg = _validate_table(braw, schema, branch, errors)
        if len(errors) == n_err:
            check_fn(bcfg, f"{branch}.", errors)
        bcfg["config_file"] = config_file
        bcfg["branch"] = branch
        cfg[branch] = bcfg
    _raise_if(errors, config_file)
    names = [cfg[b]["general"]["ModelName"] for b in BRANCHES if b in cfg]
    if len(set(names)) != len(names):
        raise ValueError(f"Invalid configuration '{config_file}': the branches share ModelName '{names[0]}'")
    cfg["config_file"] = config_file
    return cfg


def load_sngp_config(arg: str, lamturb: Optional[bool] = None) -> Dict[str, Any]:
    """
    Load and validate an SNGP training TOML (sngp_turb.toml or sngp_lamturb.toml)

    Args:
        arg: TOML name under ./inputfiles/ (or ./inputfiles/sngp/), or a path
        lamturb: Expected variant (True: lam-turb, False: turbulent-only, None: either)

    Returns:
        Validated config (lam-turb: {'general', 'lam', 'turb', 'config_file'}, see branch_configs)
    """
    return _load_training_config(arg, "sngp", sngp_schema, _check_sngp_branch, lamturb)


def load_error_config(arg: str, lamturb: Optional[bool] = None) -> Dict[str, Any]:
    """
    Load and validate an error-model training TOML (error_turb.toml or error_lamturb.toml)

    Args:
        arg: TOML name under ./inputfiles/ (or ./inputfiles/error/), or a path
        lamturb: Expected variant (True: lam-turb, False: turbulent-only, None: either)

    Returns:
        Validated config (lam-turb: {'general', 'lam', 'turb', 'config_file'}, see branch_configs)
    """
    return _load_training_config(arg, "error", error_schema, _check_error_branch, lamturb)


def branch_configs(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Per-branch configs of a training config

    Each branch config is complete (same sections as a turbulent-only TOML, shared [general]
    values already inherited) and carries 'branch' and 'config_file'.

    Args:
        cfg: Output of load_sngp_config/load_error_config

    Returns:
        {'lam': cfg_lam, 'turb': cfg_turb} for lam-turb TOMLs (only the branches present);
        {'turb': cfg} for a turbulent-only TOML
    """
    if is_lamturb(cfg):
        return {b: cfg[b] for b in BRANCHES if b in cfg}
    return {"turb": cfg}


# ---------------------------------------------------------------------------------------
# Test and benchmark configs
# ---------------------------------------------------------------------------------------


def load_test_config(arg: str, variant: str) -> Dict[str, Any]:
    """
    Load and validate a test TOML

    Args:
        arg: TOML name under ./inputfiles/ (or ./inputfiles/test/), or a path
        variant: 'turb' (test_turb.py) or 'lamturb' (test_lamturb.py)

    Returns:
        Validated config with defaults filled in
    """
    schema = test_schema(variant)
    raw = _load_raw(arg, "test")
    config_file = raw.get("config_file", arg)
    check_kind(raw, config_file, "test", lamturb=variant == "lamturb")
    errors: List[str] = []
    if variant == "lamturb":
        for name in ("models", "classes"):
            if name in raw:
                errors.append(f"Top-level [{name}] is not allowed in a lam-turb test TOML; "
                              f"use [lam.{name}] and [turb.{name}]")
                raw.pop(name)
    else:
        for name in ("lam", "turb", "switch"):
            if name in raw:
                errors.append(f"[{name}] is only allowed in a lam-turb test TOML (test_lamturb.py)")
                raw.pop(name)
    _raise_if(errors, config_file)
    _apply_aleatoric_type_defaults(raw.get("gates"))
    cfg = validate(raw, schema, config_file)
    _check_gates(cfg, errors)
    for b in (BRANCHES if variant == "lamturb" else ("",)):
        classes = cfg[b]["classes"] if b else cfg["classes"]
        if classes["PosthocGroups"]:
            _check_groups(classes["PosthocGroups"], f"{b + '.' if b else ''}classes.PosthocGroups", errors)
    _raise_if(errors, config_file)
    cfg["variant"] = variant
    return cfg


def load_benchmark_config(arg: str) -> Dict[str, Any]:
    """
    Load and validate the all-cases test TOML (test_turb_allcases.py)

    Args:
        arg: TOML name under ./inputfiles/ (or ./inputfiles/test/), or a path

    Returns:
        Validated config with defaults filled in
    """
    raw = _load_raw(arg, "test")
    config_file = raw.get("config_file", arg)
    check_kind(raw, config_file, "benchmark")
    _apply_aleatoric_type_defaults(raw.get("gates"))
    cfg = validate(raw, benchmark_schema(), config_file)
    errors: List[str] = []
    _check_gates(cfg, errors)
    _check_groups(cfg["groups"], "groups", errors)
    datasets = set(cfg["data"]["Datasets"])
    for gname, members in cfg["groups"].items():
        missing = [m for m in members if m not in datasets]
        if missing:
            errors.append(f"groups.'{gname}': members {missing} are not in data.Datasets")
    unknown_train = [k for k in cfg["data"]["TrainingDatasets"] if k not in datasets and k not in cfg["groups"]]
    if unknown_train:
        errors.append(f"data.TrainingDatasets {unknown_train} are neither in data.Datasets nor group names")
    _raise_if(errors, config_file)
    return cfg


def config_to_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON-serialisable deep copy of a validated config (stored in the model assets)

    Args:
        cfg: Validated config

    Returns:
        Plain-python copy
    """
    return to_jsonable(cfg)
