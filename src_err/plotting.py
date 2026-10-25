"""Plotting functions for the BFM_WM_v2 test, training and benchmark scripts.

Every public figure function takes explicit numpy arrays plus a plot-options
dict (the keys of the matching ``[plots.*]`` TOML table, CamelCase) and returns
the figure(s). When ``out_dir`` is given, figures are written with
:func:`save_figure` under unique names per figure family and then closed.

Figure families (``<prefix>_`` is prepended when ``prefix`` is non-empty):

- ``map_<field>`` / ``map_rawn1_<field>``: scatter maps over (x, n1/delta) or (x, n1);
- ``tri_<field>``: smoothed tripcolor maps (port of LT-NB cell 78);
- ``cf_lines`` / ``cf_lines_ma``: |C_f| at the selected stations, raw and moving-averaged;
- ``pdf_errors`` / ``pdf_confidence``: error and confidence PDFs;
- ``confusion_matrix``, ``sngp_history``, ``error_history``: training diagnostics;
- ``<RunName>_<fig>[_log]``: a-priori benchmark figures (see tools/README.md).

Conventions:

- Everything is plotted as magnitudes; mathtext only (``text.usetex`` is forced
  off inside every public function, because the parent ``plot_utils`` style may
  enable it).
- The backend is chosen by the calling script (Agg in all v2 scripts).
- Truth enters these functions only as evaluation data (errors, C_f markers).
"""

import functools
import os
import re
import warnings
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib import colors
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext

# ---------------------------------------------------------------------------
# Colours and names
# ---------------------------------------------------------------------------

# The notebooks' 19-colour class map (TURB-NB c26 / LT-NB c31), in label order.
DEFAULT_CMAP = colors.ListedColormap(
    ["#377eb8", "#e41a1c", "#800080", "#ff7f00", "#4daf4a", "#a65628",
     "#f781bf", "#999999", "#a6cee3", "#b2df8a", "#f27775", "#ebbd81",
     "#1449e7", "#cab2d6", "#db76d6", "#f6ff00", "#6dcbb8", "#aabb54", "#84b2cf"],
    name="bfmwm_classes",
)
OOD_NAME = "OOD"
# Fixed dark grey, distinct from DEFAULT_CMAP's light grey #999999.
OOD_COLOR = "#4d4d4d"

# Per-field presentation: (colourbar label, scale). 'log' fields use
# LogNorm(VMin, VMax); 'unit' fields use Normalize(0, 1).
FIELD_SPECS: Dict[str, Tuple[str, str]] = {
    "baseline": (r"$|\varepsilon_{\mathrm{base}}|\ (\%)$", "log"),
    "corrected": (r"$|\varepsilon_{\mathrm{corr}}|\ (\%)$", "log"),
    "always": (r"$|\varepsilon_{\mathrm{always}}|\ (\%)$", "log"),
    "eqwm": (r"$|\varepsilon_{\mathrm{eqwm}}|\ (\%)$", "log"),
    "mu_pct": (r"$|\mu_{\varepsilon}|\ (\%)$", "log"),
    "sigma_pct": (r"$\sigma_{\varepsilon}\ (\%)$", "log"),
    "confidence": (r"$C$", "unit"),
    "alpha_a": (r"$\alpha_{a}$", "unit"),
    "alpha_c": (r"$\alpha_{C}$", "unit"),
    "alpha_nw": (r"$\alpha_{y^+}$", "unit"),
}

# Error-curve styles shared by the PDF figures: (fill colour, line colour, label).
# baseline/corrected/eqwm follow src_err/pdf_utils.py; 'always' gets its own colour.
PDF_CURVE_STYLES: Dict[str, Tuple[str, str, str]] = {
    "baseline": ("steelblue", "darkblue", r"$\mathrm{Baseline}$"),
    "corrected": ("coral", "darkred", r"$\mathrm{Expert\ Model}$"),
    "eqwm": ("teal", "teal", r"$\mathrm{EQWM}$"),
    "always": ("mediumpurple", "indigo", r"$\mathrm{Always\ Corrected}$"),
}
_PDF_KEY_ALIASES = {"final": "corrected"}

# C_f line colours (TURB-NB c66/c67).
_CF_STYLES = {
    "baseline": ("b", "-", r"$\mathrm{Baseline}$"),
    "corrected": ("r", "-", r"$\mathrm{Expert\ Model}$"),
    "eqwm": ("g", "-", r"$\mathrm{EQWM}$"),
    "always": ("m", "--", r"$\mathrm{Always\ Corrected}$"),
}


def _no_usetex(func):
    """Run a figure function with mathtext (text.usetex off), whatever the global style says."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context({"text.usetex": False}):
            return func(*args, **kwargs)
    return wrapper


def mathrm(text: str) -> str:
    """Wrap plain text as upright mathtext (the notebooks' ``latex_class_names``).

    Text that already contains ``$`` is returned unchanged.
    """
    text = str(text)
    if "$" in text:
        return text
    for ch in ("\\", "{", "}", "_", "%", "&", "#", "^"):
        text = text.replace(ch, "\\" + ch if ch != "\\" else r"\backslash ")
    text = text.replace(" ", r"\ ")
    return r"$\mathrm{" + text + "}$"


def class_colors(class_names: Sequence[str]) -> List:
    """Colour per class: DEFAULT_CMAP in order for regular classes, OOD_COLOR for 'OOD'.

    More than 19 regular classes cycle through DEFAULT_CMAP (with a warning).
    """
    base = list(DEFAULT_CMAP.colors)
    out, k = [], 0
    for name in class_names:
        if str(name) == OOD_NAME:
            out.append(OOD_COLOR)
        else:
            out.append(base[k % len(base)])
            k += 1
    if k > len(base):
        warnings.warn(f"{k} classes but DEFAULT_CMAP has {len(base)} colours; colours repeat.")
    return out


def labels_with_ood(labels: np.ndarray, class_names: Sequence[str],
                    ood_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    """Relabel low-confidence points as 'OOD' without touching the inputs.

    Args:
        labels: integer class labels (N,).
        class_names: names in label order.
        ood_mask: boolean (N,) mask of OOD points, or None.

    Returns:
        (new_labels, new_class_names); 'OOD' is appended when absent and needed.
    """
    labels = np.asarray(labels, dtype=int).copy()
    names = [str(n) for n in class_names]
    if ood_mask is not None and np.any(ood_mask):
        if OOD_NAME not in names:
            names.append(OOD_NAME)
        labels[np.asarray(ood_mask, dtype=bool)] = names.index(OOD_NAME)
    return labels, names


def piout_to_cf(pi_out: np.ndarray, n1: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """|C_f| = 2 tau_w with tau_w = (pi_out nu / n1)^2, taking U_inf = 1 (as the notebooks)."""
    pi_out = np.abs(np.asarray(pi_out, dtype=float))
    return 2.0 * (pi_out * np.asarray(nu, dtype=float) / np.asarray(n1, dtype=float)) ** 2


def _limits(value) -> Optional[List[float]]:
    """Axis limits from an option: None or [] (TOML 'automatic') -> None, else [lo, hi]."""
    if value is None or len(value) == 0:
        return None
    return [float(v) for v in value]


def _pdf_method(method: str) -> str:
    """PDF estimator name; 'hist' (TOML) is an alias of 'histogram'."""
    method = str(method)
    return "histogram" if method == "hist" else method


def _name(prefix: str, *parts: str) -> str:
    return "_".join(p for p in (prefix, *parts) if p)


def save_figure(fig, out_dir: Optional[str], name: str, formats: Sequence[str] = ("png",),
                dpi: int = 300) -> List[str]:
    """Save ``fig`` as ``<out_dir>/<name>.<fmt>`` for every format.

    Args:
        fig: matplotlib figure.
        out_dir: output directory (created if needed); None skips saving.
        name: file stem (unique per figure family).
        formats: e.g. ("png", "pdf").
        dpi: raster resolution (300, as the notebooks).

    Returns:
        The written paths.
    """
    if out_dir is None:
        return []
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f"{name}.{str(fmt).lstrip('.')}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    print(f"Saved: {', '.join(paths)}")
    return paths


def _finish(fig, out_dir, name, formats, close):
    save_figure(fig, out_dir, name, formats)
    if out_dir is not None and close:
        plt.close(fig)
    return fig


def _log10_ticks(vmin: float, vmax: float) -> List[float]:
    pmin, pmax = int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax)))
    return [10.0 ** p for p in range(pmin, pmax + 1) if vmin <= 10.0 ** p <= vmax]


def _field_norm(key: str, values: np.ndarray, vmin: float, vmax: float):
    """Return (values to plot, norm, colourbar label, is_log) for a field key."""
    label, scale = FIELD_SPECS.get(key, (mathrm(key), "linear"))
    if key not in FIELD_SPECS:
        warnings.warn(f"plotting: unknown field '{key}', drawn on a linear scale.")
    v = np.asarray(values, dtype=float).ravel()
    if scale == "log":
        # |e%| of exactly 0 would be masked by LogNorm; clip so it shows the lowest colour.
        return np.clip(np.abs(v), 1e-12, None), LogNorm(vmin=vmin, vmax=vmax), label, True
    if scale == "unit":
        return v, Normalize(vmin=0.0, vmax=1.0), label, False
    return v, Normalize(vmin=np.nanmin(v), vmax=np.nanmax(v)), label, False


def _class_colorbar(fig, ax, mappable, names, labelsize=None):
    n = len(names)
    cbar = fig.colorbar(mappable, ax=ax, ticks=np.arange(n), boundaries=np.arange(-0.5, n + 0.5, 1))
    cbar.set_ticklabels([mathrm(nm) for nm in names])
    cbar.minorticks_off()
    if labelsize is None:
        labelsize = float(np.clip(110.0 / max(n, 1), 6, 14))
    cbar.ax.tick_params(labelrotation=0, labelsize=labelsize)
    return cbar


# ---------------------------------------------------------------------------
# Spatial scatter maps
# ---------------------------------------------------------------------------

@_no_usetex
def plot_spatial_maps(x: np.ndarray, n1: np.ndarray, delta: Optional[np.ndarray],
                      fields: Mapping[str, np.ndarray], labels: Optional[np.ndarray] = None,
                      class_names: Optional[Sequence[str]] = None, opts: Optional[Mapping] = None,
                      prefix: str = "", *, out_dir: Optional[str] = None,
                      formats: Sequence[str] = ("png",), selected_idx: Optional[np.ndarray] = None,
                      ood_mask: Optional[np.ndarray] = None, raw_n1: bool = False,
                      xlabel: str = r"$x$", close: bool = True) -> Dict[str, plt.Figure]:
    """Scatter maps of per-point fields over (x, n1/delta) or (x, n1) (TURB-NB c64, LT-NB c74/c75).

    Args:
        x: streamwise coordinate (N,).
        n1: wall-normal matching distance (N,).
        delta: boundary-layer thickness (N,); required for YAxis 'n1_over_delta'.
        fields: field key -> (N,) values. Known keys (FIELD_SPECS): 'baseline', 'corrected',
            'always', 'eqwm' (signed or |e%|; plotted as magnitudes, LogNorm), 'mu_pct',
            'sigma_pct' (inference-normalised predictions, LogNorm), 'confidence' and
            'alpha_*' (0-1).
        labels: predicted class labels (N,) for the 'labels' map, or None.
        class_names: display names in label order (may already contain 'OOD').
        opts: ``[plots.maps]``: YAxis ('n1_over_delta' | 'n1', default 'n1_over_delta'),
            YLim ([0, 0.2]), RawN1YLim ([0, 0.005]), VMin (1), VMax (100), MarkerSize (10),
            ShowSelected (true), XLim (optional, default auto).
        prefix: file-name prefix (e.g. a dataset tag).
        out_dir, formats: saving (None: no saving).
        selected_idx: indices of the selected n1 stations, overlaid in red on the
            'baseline' and 'labels' maps when ShowSelected.
        ood_mask: points relabelled 'OOD' in the label map (C below the confidence threshold).
        raw_n1: raw-n1 family ('map_rawn1_<field>', y = n1, ylim RawN1YLim, 15x3 figures).
        xlabel: x-axis label (e.g. r'$x/c$').
        close: close figures after saving.

    Returns:
        Dict field key -> figure (the label map under 'labels').
    """
    opts = dict(opts or {})
    vmin, vmax = float(opts.get("VMin", 1.0)), float(opts.get("VMax", 100.0))
    ms = float(opts.get("MarkerSize", 10))
    show_sel = bool(opts.get("ShowSelected", True)) and selected_idx is not None
    x = np.asarray(x, dtype=float).ravel()
    n1 = np.asarray(n1, dtype=float).ravel()
    if raw_n1:
        family, yvals, ylabel = "map_rawn1", n1, r"$n_1$"
        ylim, figsize = _limits(opts.get("RawN1YLim", [0.0, 0.005])), (15, 3)
    else:
        family, figsize = "map", (10, 3)
        ylim = _limits(opts.get("YLim", [0.0, 0.2]))
        if str(opts.get("YAxis", "n1_over_delta")) == "n1":
            yvals, ylabel = n1, r"$n_1$"
        else:
            if delta is None:
                raise ValueError("plot_spatial_maps: delta is required for YAxis = 'n1_over_delta'")
            yvals, ylabel = n1 / np.asarray(delta, dtype=float).ravel(), r"$n_1/\delta$"
    xlim = _limits(opts.get("XLim"))
    sel = np.asarray(selected_idx, dtype=int) if show_sel else None

    def _decorate(fig, ax):
        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_ylabel(ylabel, fontsize=25)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if xlim is not None:
            ax.set_xlim(*xlim)
        fig.tight_layout()

    figs = {}
    for key, values in fields.items():
        if values is None:
            continue
        v, norm, cl, is_log = _field_norm(key, values, vmin, vmax)
        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(x, yvals, c=v, cmap="viridis", norm=norm, s=ms, rasterized=True)
        if sel is not None and key == "baseline":
            ax.scatter(x[sel], yvals[sel], c="red", s=2, alpha=0.7, label="Selected points")
        cbar = fig.colorbar(sc, ax=ax, extend="both" if is_log else "neither")
        cbar.set_label(cl, fontsize=16)
        if is_log:
            cbar.set_ticks(_log10_ticks(vmin, vmax))
            cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        _decorate(fig, ax)
        figs[key] = _finish(fig, out_dir, _name(prefix, family, key), formats, close)

    if labels is not None:
        if class_names is None:
            raise ValueError("plot_spatial_maps: class_names is required with labels")
        lab, names = labels_with_ood(labels, class_names, ood_mask)
        cmap = colors.ListedColormap(class_colors(names))
        norm = colors.BoundaryNorm(np.arange(-0.5, len(names) + 0.5, 1), cmap.N)
        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(x, yvals, c=lab, cmap=cmap, norm=norm, s=ms, rasterized=True)
        if sel is not None:
            ax.scatter(x[sel], yvals[sel], c="red", s=2, alpha=0.7, label="Selected points")
        _class_colorbar(fig, ax, sc, names)
        _decorate(fig, ax)
        figs["labels"] = _finish(fig, out_dir, _name(prefix, family, "labels"), formats, close)
    return figs


# ---------------------------------------------------------------------------
# C_f line plots
# ---------------------------------------------------------------------------

def moving_average(data: np.ndarray, x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Index-based moving average with trimmed x (TURB-NB c67 semantics).

    ``window <= 1`` returns the inputs unchanged; a window longer than the data is
    shortened to the data length.
    """
    data, x = np.asarray(data, dtype=float), np.asarray(x, dtype=float)
    w = int(min(window, data.size))
    if w <= 1:
        return data, x
    smooth = np.convolve(data, np.ones(w) / w, mode="valid")
    return smooth, x[(w - 1) // 2: x.size - (w // 2)]


def subsample_stations_by_x(x: np.ndarray, n: int = 40, dx: float = 0.0,
                            xlim: Optional[Sequence[float]] = None) -> np.ndarray:
    """Indices of stations evenly spaced in x (not in array index).

    Args:
        x: station x positions (M,), any order.
        n: number of evenly spaced targets over the x range (used when dx <= 0).
        dx: target spacing; > 0 overrides n.
        xlim: optional range restricting the targets.

    Returns:
        Unique indices into ``x``, sorted by x, of the station nearest to each target.
    """
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros(0, dtype=int)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if xlim is not None:
        lo, hi = max(lo, xlim[0]), min(hi, xlim[1])
    if hi < lo:
        return np.zeros(0, dtype=int)
    targets = np.arange(lo, hi + 0.5 * dx, dx) if dx > 0 else np.linspace(lo, hi, max(int(n), 1))
    order = np.argsort(np.where(ok, x, np.inf))[: ok.sum()]
    xs = x[order]
    pos = np.clip(np.searchsorted(xs, targets), 1, max(xs.size - 1, 1))
    left = np.clip(pos - 1, 0, xs.size - 1)
    right = np.clip(pos, 0, xs.size - 1)
    pick = np.where(np.abs(xs[left] - targets) <= np.abs(xs[right] - targets), left, right)
    return order[np.unique(pick)]


def _truth_per_station(x_all, truth_all, round_decimals=12):
    """One truth value per unique x (median over the wall-normal points at that x)."""
    xr = np.round(np.asarray(x_all, dtype=float), round_decimals)
    t = np.asarray(truth_all, dtype=float)
    order = np.argsort(xr, kind="stable")
    xs, ts = xr[order], t[order]
    ux, start = np.unique(xs, return_index=True)
    med = np.array([np.nanmedian(seg) for seg in np.split(ts, start[1:])])
    return ux, med


@_no_usetex
def plot_cf_lines(x_sel: np.ndarray, fields_sel: Mapping[str, np.ndarray],
                  truth_sel: Optional[np.ndarray] = None, x_all: Optional[np.ndarray] = None,
                  truth_all: Optional[np.ndarray] = None, sigma_pct_sel: Optional[np.ndarray] = None,
                  conf_sel: Optional[np.ndarray] = None, opts: Optional[Mapping] = None,
                  prefix: str = "", *, out_dir: Optional[str] = None,
                  formats: Sequence[str] = ("png",), xlabel: str = r"$x$",
                  close: bool = True) -> Tuple[plt.Figure, plt.Figure]:
    """|C_f| along x at the selected n1 stations, raw and moving-averaged (TURB-NB c66/c67).

    All C_f inputs are magnitudes ``|C_f| = 2 tau_w``, tau_w = (pi_out nu / n1)^2 with
    U_inf = 1 (use :func:`piout_to_cf`).

    Args:
        x_sel: x of the selected stations (one per x; any order, sorted here).
        fields_sel: 'baseline', 'corrected' (final blended), 'eqwm', optional 'always' -> |C_f| (M,).
        truth_sel: truth |C_f| at the stations (used when x_all/truth_all are None).
        x_all, truth_all: x and truth |C_f| at all points; reduced to one value per x
            (median), so markers do not depend on array order.
        sigma_pct_sel: predicted sigma % at the stations (twin axis), or None.
        conf_sel: confidence C at the stations (twin axis, shown as 100 C), or None.
        opts: ``[plots.cf_lines]``: MovingAverage (50), XLim (auto), YLim (optional,
            default bottom 0), TruthMarkers (40 evenly spaced in x), TruthMarkerDx
            (optional, > 0 overrides TruthMarkers). Selection keys are not read here.
        prefix: file-name prefix. out_dir, formats: saving.
        xlabel: x-axis label.
        close: close figures after saving.

    Returns:
        (raw figure 'cf_lines', moving-average figure 'cf_lines_ma').
    """
    opts = dict(opts or {})
    window = int(opts.get("MovingAverage", 50))
    xlim = _limits(opts.get("XLim"))
    ylim = _limits(opts.get("YLim"))
    n_markers = int(opts.get("TruthMarkers", 40))
    dx_markers = float(opts.get("TruthMarkerDx", 0.0) or 0.0)

    xs = np.asarray(x_sel, dtype=float)
    order = np.argsort(xs, kind="stable")
    xs = xs[order]
    cf = {k: np.asarray(v, dtype=float)[order] for k, v in fields_sel.items() if v is not None}
    sig = None if sigma_pct_sel is None else np.abs(np.asarray(sigma_pct_sel, dtype=float))[order]
    conf = None if conf_sel is None else np.asarray(conf_sel, dtype=float).ravel()[order]

    if x_all is not None and truth_all is not None:
        xt, tt = _truth_per_station(x_all, truth_all)
    elif truth_sel is not None:
        xt, tt = xs, np.asarray(truth_sel, dtype=float)[order]
    else:
        xt = tt = None
    if xt is not None:
        mk = subsample_stations_by_x(xt, n=n_markers, dx=dx_markers, xlim=xlim)
        xt_m, tt_m = xt[mk], tt[mk]

    ylabel = r"$|C_f| = 2\,(\pi_{out}\,\nu/n_1)^2\ \ (U_\infty = 1)$"

    def _twin(ax1, xv, sv, cv):
        if sv is None and cv is None:
            return []
        ax2 = ax1.twinx()
        if sv is not None:
            ax2.plot(xv, sv, c="orange", alpha=0.7, label=r"$\sigma_{\varepsilon}\ (\%)$")
        if cv is not None:
            ax2.plot(xv, 100.0 * cv, c="purple", alpha=0.7, label=r"$100\,C$")
        ax2.set_ylabel(r"$\sigma_{\varepsilon}\ (\%),\ 100\,C$", fontsize=18)
        ax2.set_ylim(0, 101)
        return ax2.get_legend_handles_labels()

    def _axes_common(ax1, fontsize):
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax1.set_xlabel(xlabel, fontsize=fontsize)
        ax1.set_ylabel(ylabel, fontsize=fontsize * 0.6)
        if xlim is not None:
            ax1.set_xlim(*xlim)
        if ylim is not None:
            ax1.set_ylim(*ylim)
        else:
            ax1.set_ylim(bottom=0.0)

    # Raw values at the stations (c66): every station's truth as open circles.
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for k, v in cf.items():
        c, ls, lab = _CF_STYLES.get(k, ("k", ":", mathrm(k)))
        ax1.plot(xs, v, c=c, ls=ls, alpha=0.8, label=lab)
    if xt is not None:
        ax1.scatter(xt, tt, s=20, marker="o", facecolors="none", edgecolors="black",
                    linewidths=0.5, label=r"$\mathrm{Truth}$")
    _axes_common(ax1, 22)
    ax1.grid(True)
    h2 = _twin(ax1, xs, sig, conf)
    h1 = ax1.get_legend_handles_labels()
    ax1.legend(h1[0] + (h2[0] if h2 else []), h1[1] + (h2[1] if h2 else []), loc="best", fontsize=11,
               ncol=2, framealpha=0.85)
    fig1.tight_layout()
    fig1 = _finish(fig1, out_dir, _name(prefix, "cf_lines"), formats, close)

    # Moving average over the stations (c67), truth markers subsampled by x.
    fig2, ax1 = plt.subplots(figsize=(12, 5))
    xm = xs
    for k, v in cf.items():
        sm, xm = moving_average(v, xs, window)
        c, ls, lab = _CF_STYLES.get(k, ("k", ":", mathrm(k)))
        ax1.plot(xm, sm, c=c, ls=ls, label=lab)
    if xt is not None:
        ax1.scatter(xt_m, tt_m, s=10, marker="o", facecolors="none", edgecolors="black",
                    linewidths=0.5, label=r"$\mathrm{Truth}$")
    _axes_common(ax1, 30)
    sig_m = None if sig is None else moving_average(sig, xs, window)[0]
    conf_m = None if conf is None else moving_average(conf, xs, window)[0]
    if sig_m is not None or conf_m is not None:
        xm = moving_average(np.zeros_like(xs), xs, window)[1]
    h2 = _twin(ax1, xm, sig_m, conf_m)
    h1 = ax1.get_legend_handles_labels()
    ax1.legend(h1[0] + (h2[0] if h2 else []), h1[1] + (h2[1] if h2 else []), loc="best", fontsize=13,
               ncol=2, framealpha=0.85)
    ax1.set_title(f"moving average over {max(min(window, xs.size), 1)} stations", fontsize=12)
    fig2.tight_layout()
    fig2 = _finish(fig2, out_dir, _name(prefix, "cf_lines_ma"), formats, close)
    return fig1, fig2


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------

def _plain(label: str) -> str:
    """Mathtext label -> plain text for console logs."""
    return re.sub(r"\\mathrm\{(.*)\}", r"\1", label.strip("$")).replace("\\ ", " ")


def _kde_input(values: np.ndarray, lo: float, hi: float, max_points: int) -> np.ndarray:
    v = np.abs(np.asarray(values, dtype=float).ravel())
    v = v[np.isfinite(v)]
    v = v[(v >= lo) & (v <= hi)]
    if max_points > 0 and v.size > max_points:
        # Deterministic thinning keeps the KDE cost bounded for pooled benchmark data.
        v = v[np.linspace(0, v.size - 1, max_points).astype(int)]
    return v


def _draw_pdf_curves(ax, curves, x_range, method, bins, fill_alpha, line_width,
                     max_points, legend_fontsize):
    """Draw error PDFs; each KDE is fitted only to the values inside x_range."""
    from scipy.stats import gaussian_kde

    method = _pdf_method(method)
    grid = np.linspace(x_range[0], x_range[1], 500)
    handles = []
    for values, fill, line, label in curves:
        v = _kde_input(values, x_range[0], x_range[1], max_points)
        if v.size == 0:
            print(f"  {_plain(label)}: no points in range {list(x_range)}")
            continue
        print(f"  {_plain(label)}: n_in_range={v.size}, mean={np.mean(v):.2f}%, std={np.std(v):.2f}%")
        if method in ("histogram", "both"):
            ax.hist(v, bins=bins, range=tuple(x_range), alpha=fill_alpha, color=fill,
                    density=True, edgecolor="black", label=label)
        if method in ("kde", "both"):
            if v.size > 1 and np.std(v) > 0:
                dens = gaussian_kde(v)(grid)
                ax.plot(grid, dens, color=line, linewidth=line_width)
                ax.fill_between(grid, dens, alpha=fill_alpha, color=fill)
            else:
                ax.axvline(float(v.mean()), color=line, linewidth=line_width, linestyle="--")
            handles.append(Patch(facecolor=fill, edgecolor=fill, alpha=fill_alpha, label=label))
    if method == "kde":
        if handles:
            ax.legend(handles=handles, fontsize=legend_fontsize, loc="upper right")
    elif ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=legend_fontsize, loc="upper right")


def _pdf_axes_style(ax, xlabel, x_range, tick_size):
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(r"$\mathrm{PDF}$", fontsize=24)
    ax.set_xlim(x_range)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.set_aspect("auto")
    ax.set_box_aspect(0.65)


@_no_usetex
def plot_confidence_pdf(confidence: np.ndarray, conf_range: Sequence[float] = (0.7, 1.0),
                        method: str = "kde", bins: int = 50, *, name: str = "pdf_confidence",
                        out_dir: Optional[str] = None, formats: Sequence[str] = ("png",),
                        max_points: int = 1_000_000, close: bool = True) -> plt.Figure:
    """Confidence PDF (pdf_utils semantics: KDE fitted on all finite C, drawn over conf_range)."""
    from scipy.stats import gaussian_kde

    method = _pdf_method(method)
    c = np.asarray(confidence, dtype=float).ravel()
    c = c[np.isfinite(c)]
    if max_points > 0 and c.size > max_points:
        c = c[np.linspace(0, c.size - 1, max_points).astype(int)]
    fig, ax = plt.subplots(figsize=(8, 8))
    if c.size:
        print(f"  Confidence: n={c.size}, mean={np.mean(c):.4f}, std={np.std(c):.4f}")
    if method in ("histogram", "both"):
        ax.hist(c, bins=bins, range=tuple(conf_range), alpha=0.6, color="mediumseagreen",
                density=True, edgecolor="black", label=r"$\mathrm{Confidence\ Score}$")
    if method in ("kde", "both") and c.size > 1 and np.std(c) > 0:
        grid = np.linspace(conf_range[0], conf_range[1], 500)
        dens = gaussian_kde(c)(grid)
        ax.plot(grid, dens, color="darkgreen", linewidth=1 if method == "kde" else 2)
        ax.fill_between(grid, dens, alpha=0.3, color="mediumseagreen")
    _pdf_axes_style(ax, r"$\mathrm{Confidence\ Score}$", list(conf_range), 24)
    fig.tight_layout()
    return _finish(fig, out_dir, name, formats, close)


@_no_usetex
def plot_pdfs(errors: Mapping[str, np.ndarray], confidence: Optional[np.ndarray] = None,
              opts: Optional[Mapping] = None, prefix: str = "", *, out_dir: Optional[str] = None,
              formats: Sequence[str] = ("png",), labels: Optional[Mapping[str, str]] = None,
              max_points: int = 1_000_000,
              close: bool = True) -> Tuple[plt.Figure, Optional[plt.Figure]]:
    """Error and confidence PDFs (port of ``src_err/pdf_utils.plot_training_pdfs``).

    The error KDEs are fitted only to |e%| values inside XRange, so each curve is the
    density conditional on |e%| lying in XRange (curves integrate to 1 over XRange,
    and the fraction outside the range is not shown). The confidence KDE is fitted on
    all values and drawn over ConfRange.

    Args:
        errors: curve key -> e% (N,) (abs taken). Keys, drawn in this order: 'baseline',
            'corrected' (alias 'final'), 'eqwm', 'always'.
        confidence: C (N,), or None to skip the confidence figure.
        opts: ``[plots.pdfs]``: Method ('kde' | 'histogram' (alias 'hist') | 'both', default 'kde'),
            XRange ([0, 40]), ConfRange ([0.7, 1]), Bins (50).
        prefix: file-name prefix; files '<prefix>_pdf_errors', '<prefix>_pdf_confidence'.
        out_dir, formats: saving.
        labels: optional legend-label overrides per key.
        max_points: deterministic thinning cap per curve before the KDE (0 = none).
        close: close figures after saving.

    Returns:
        (error-PDF figure, confidence-PDF figure or None).
    """
    opts = dict(opts or {})
    method = str(opts.get("Method", "kde"))
    x_range = list(opts.get("XRange", [0.0, 40.0]))
    conf_range = list(opts.get("ConfRange", [0.7, 1.0]))
    bins = int(opts.get("Bins", 50))
    errs = {_PDF_KEY_ALIASES.get(k, k): v for k, v in errors.items() if v is not None}
    unknown = set(errs) - set(PDF_CURVE_STYLES)
    if unknown:
        raise ValueError(f"plot_pdfs: unknown curve keys {sorted(unknown)}; valid: {list(PDF_CURVE_STYLES)}")
    curves = []
    for key, (fill, line, lab) in PDF_CURVE_STYLES.items():
        if key in errs:
            lab = mathrm(labels[key]) if labels and key in labels else lab
            curves.append((errs[key], fill, line, lab))
    print(f"PDFs ({method}); KDEs fitted within XRange={x_range} (conditional densities):")
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_pdf_curves(ax, curves, x_range, method, bins, 0.3, 1.0, max_points, 24)
    _pdf_axes_style(ax, r"$\mathrm{Model\ Error\ (\%)}$", x_range, 24)
    fig.tight_layout()
    fig = _finish(fig, out_dir, _name(prefix, "pdf_errors"), formats, close)
    fig_c = None
    if confidence is not None:
        fig_c = plot_confidence_pdf(confidence, conf_range, method, bins,
                                    name=_name(prefix, "pdf_confidence"), out_dir=out_dir,
                                    formats=formats, max_points=max_points, close=close)
    return fig, fig_c


# ---------------------------------------------------------------------------
# Smoothed tripcolor maps (LT-NB cell 78)
# ---------------------------------------------------------------------------

class _GridSmoother:
    """Scattered -> regular grid -> Gaussian filter -> back to the points (LT-NB c78).

    Equivalent to the notebook's ``griddata`` calls, with the triangulation and the
    nearest-neighbour lookups computed once and reused for every field.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, nx: int, ny: int):
        from scipy.spatial import Delaunay, cKDTree

        self.pts = np.column_stack([x, y])
        self.xi = np.linspace(x.min(), x.max(), nx)
        self.yi = np.linspace(y.min(), y.max(), ny)
        Xi, Yi = np.meshgrid(self.xi, self.yi)
        self.grid = np.column_stack([Xi.ravel(), Yi.ravel()])
        self.shape = Xi.shape
        self.tri = Delaunay(self.pts)
        self.nearest = cKDTree(self.pts).query(self.grid)[1]
        # Nearest grid node of each point (independent rounding per axis on a regular grid).
        dx = (self.xi[-1] - self.xi[0]) / max(nx - 1, 1) or 1.0
        dy = (self.yi[-1] - self.yi[0]) / max(ny - 1, 1) or 1.0
        self.ix = np.clip(np.rint((x - self.xi[0]) / dx).astype(int), 0, nx - 1)
        self.iy = np.clip(np.rint((y - self.yi[0]) / dy).astype(int), 0, ny - 1)

    def field(self, z: np.ndarray, sigma: float, log_space: bool) -> np.ndarray:
        from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
        from scipy.ndimage import gaussian_filter

        z = np.asarray(z, dtype=float)
        work = np.log10(np.clip(z, 1e-12, None)) if log_space else z.copy()
        zi = LinearNDInterpolator(self.tri, work)(self.grid)
        zi = np.where(np.isnan(zi), work[self.nearest], zi).reshape(self.shape)
        zs = gaussian_filter(zi, sigma=sigma)
        back = RegularGridInterpolator((self.yi, self.xi), zs, method="linear",
                                       bounds_error=False, fill_value=np.nan)(self.pts[:, ::-1])
        back = np.where(np.isnan(back), work, back)
        return 10.0 ** back if log_space else back

    def labels(self, z: np.ndarray, sigma: float) -> np.ndarray:
        from scipy.ndimage import uniform_filter

        z = np.asarray(z, dtype=int)
        zi = z[self.nearest].reshape(self.shape)
        size = max(5, int(sigma * 5))
        votes = np.stack([uniform_filter((zi == k).astype(float), size=size)
                          for k in range(int(z.max()) + 1)], axis=0)
        zs = np.argmax(votes, axis=0)
        return zs[self.iy, self.ix].astype(int)


def _triangle_majority(triangles: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    """Per-triangle majority of the three vertex labels (first vertex when all differ)."""
    a, b, c = (vertex_labels[triangles[:, i]] for i in range(3))
    return np.where((a == b) | (a == c), a, np.where(b == c, b, a))


@_no_usetex
def plot_tripcolor_maps(x: np.ndarray, n1_over_delta: np.ndarray, fields: Mapping[str, np.ndarray],
                        labels: Optional[np.ndarray] = None,
                        class_names: Optional[Sequence[str]] = None,
                        masks: Optional[Mapping[str, np.ndarray]] = None,
                        opts: Optional[Mapping] = None, prefix: str = "", *,
                        out_dir: Optional[str] = None, formats: Sequence[str] = ("png",),
                        selected_idx: Optional[np.ndarray] = None, show_selected: bool = True,
                        xlabel: str = r"$x$", ylabel: str = r"$n_1/\delta$",
                        close: bool = True) -> Dict[str, plt.Figure]:
    """Smoothed tripcolor maps over (x, n1/delta) (faithful port of LT-NB cell 78).

    Continuous fields are interpolated to a GridNx x GridNy grid (linear, NaNs filled by
    nearest), Gaussian-filtered (Sigma grid cells; in log10 space for the 'log' fields)
    and sampled back to the points, then drawn with Gouraud tripcolor. Class labels are
    smoothed by a per-class uniform-filter majority vote and drawn with 'flat' shading
    whose triangle colour is the majority of its vertex labels (class ids are never
    averaged). The OOD and near-wall fallback masks are hatched over the label map.

    Args:
        x, n1_over_delta: point coordinates (N,).
        fields: field key -> (N,) values (same keys as :func:`plot_spatial_maps`).
        labels: predicted class labels (N,) or None.
        class_names: display names in label order (may already contain 'OOD').
        masks: optional {'ood': C < threshold, 'nearwall': y+ < threshold} (N,) booleans.
            'ood' points are also relabelled 'OOD' in the label map.
        opts: ``[plots.tripcolor]``: YLim ([0.01, 0.2]), GridNx (600), GridNy (150),
            Sigma (3.0), XLim (optional, default data range), VMin (1), VMax (100).
        prefix: file-name prefix; files '<prefix>_tri_<field>', '<prefix>_tri_labels'.
        out_dir, formats: saving.
        selected_idx: selected-station indices, overlaid on the 'baseline' map.
        show_selected: draw the overlay when selected_idx is given.
        xlabel, ylabel: axis labels.
        close: close figures after saving.

    Returns:
        Dict field key -> figure (label map under 'labels').
    """
    opts = dict(opts or {})
    ylim = _limits(opts.get("YLim", [0.01, 0.2]))
    xlim = _limits(opts.get("XLim"))
    sigma = float(opts.get("Sigma", 3.0))
    vmin, vmax = float(opts.get("VMin", 1.0)), float(opts.get("VMax", 100.0))
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(n1_over_delta, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.all():
        warnings.warn(f"plot_tripcolor_maps: dropping {np.sum(~ok)} points with non-finite coordinates")
    x, y = x[ok], y[ok]
    smoother = _GridSmoother(x, y, int(opts.get("GridNx", 600)), int(opts.get("GridNy", 150)))
    triang = mtri.Triangulation(x, y, triangles=smoother.tri.simplices)
    sel_mask = np.zeros(ok.size, dtype=bool)
    if selected_idx is not None and show_selected:
        sel_mask[np.asarray(selected_idx, dtype=int)] = True
    sel_mask = sel_mask[ok]

    def _decorate(fig, ax):
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlim(*(xlim if xlim is not None else (x.min(), x.max())))
        if ylim is not None:
            ax.set_ylim(*ylim)
        fig.tight_layout()

    figs = {}
    for key, values in fields.items():
        if values is None:
            continue
        v, norm, cl, is_log = _field_norm(key, np.asarray(values, dtype=float).ravel()[ok], vmin, vmax)
        z = smoother.field(v, sigma, log_space=is_log)
        if not is_log and isinstance(norm, Normalize):
            z = np.clip(z, norm.vmin, norm.vmax)
        fig, ax = plt.subplots(figsize=(15, 3))
        tc = ax.tripcolor(triang, z, cmap="viridis", norm=norm, shading="gouraud", rasterized=True)
        if key == "baseline" and sel_mask.any():
            ax.scatter(x[sel_mask], y[sel_mask], c="red", s=5, alpha=0.7, label="Selected points")
        cbar = fig.colorbar(tc, ax=ax, label=cl, extend="neither")
        if is_log:
            cbar.set_ticks(_log10_ticks(vmin, vmax))
            cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        elif key in FIELD_SPECS and FIELD_SPECS[key][1] == "unit":
            cbar.set_ticks([0.0, 0.5, 1.0])
        _decorate(fig, ax)
        figs[key] = _finish(fig, out_dir, _name(prefix, "tri", key), formats, close)

    if labels is not None:
        if class_names is None:
            raise ValueError("plot_tripcolor_maps: class_names is required with labels")
        masks = dict(masks or {})
        ood = masks.get("ood")
        lab, names = labels_with_ood(np.asarray(labels)[ok] if np.size(labels) == ok.size else labels,
                                     class_names, None if ood is None else np.asarray(ood)[ok])
        cmap = colors.ListedColormap(class_colors(names))
        bounds = np.arange(-0.5, len(names) + 0.5, 1)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        z_lab = smoother.labels(lab, sigma)
        face = _triangle_majority(triang.triangles, z_lab).astype(float)
        fig, ax = plt.subplots(figsize=(15, 3))
        tc = ax.tripcolor(triang, facecolors=face, cmap=cmap, norm=norm, shading="flat", rasterized=True)
        hatch_handles = []
        for mkey, hatch, text in (("ood", ".....", r"$\mathrm{Low\ confidence\ (OOD)}$"),
                                  ("nearwall", "\\\\\\\\", r"$\mathrm{Near{-}wall\ fallback}$")):
            m = masks.get(mkey)
            if m is None:
                continue
            mz = np.asarray(m, dtype=float).ravel()[ok]
            if mz.max(initial=0.0) < 0.5:
                continue
            ax.tricontourf(triang, mz, levels=[0.5, 1.5], colors="none", hatches=[hatch])
            hatch_handles.append(Patch(facecolor="white", edgecolor="black", hatch=hatch, label=text))
        if hatch_handles:
            ax.legend(handles=hatch_handles, loc="upper right", fontsize=10, framealpha=0.8)
        _class_colorbar(fig, ax, tc, names, labelsize=12 if len(names) <= 10 else None)
        _decorate(fig, ax)
        figs["labels"] = _finish(fig, out_dir, _name(prefix, "tri", "labels"), formats, close)
    return figs


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------

@_no_usetex
def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], *, normalize: bool = True,
                          title: str = "Normalized Confusion Matrix (Validation)",
                          ignored_fraction: Optional[float] = None, name: str = "confusion_matrix",
                          out_dir: Optional[str] = None, formats: Sequence[str] = ("png",),
                          close: bool = True) -> plt.Figure:
    """Row-normalised confusion matrix with display names (TURB-NB c24).

    Args:
        cm: (K, K) counts or an already row-normalised matrix (re-normalising is idempotent).
        class_names: display names in label order (length K).
        normalize: divide each row by its sum (rows with no samples stay 0).
        title: axes title.
        ignored_fraction: fraction of points dropped by the known-class filter, noted in the title.
        name, out_dir, formats: saving.
        close: close the figure after saving.
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = np.asarray(cm, dtype=float)
    if normalize:
        rows = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    k = cm.shape[0]
    size = max(8.0, 0.55 * k + 3.0)
    fig, ax = plt.subplots(figsize=(size * 1.2, size))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(n) for n in class_names]).plot(
        cmap="Blues", ax=ax, values_format=".2f", colorbar=True)
    if ignored_fraction is not None:
        title = f"{title}\n(points outside the trained classes ignored: {100 * ignored_fraction:.2f}%)"
    ax.set_title(title)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontsize(12)
    plt.setp(ax.get_xticklabels(), rotation=90)
    if k > 12:
        for t in ax.texts:
            t.set_fontsize(max(5, 12 - (k - 12) // 2))
    fig.tight_layout()
    return _finish(fig, out_dir, name, formats, close)


def _first_key(history: Mapping, candidates: Sequence[str]) -> Optional[str]:
    return next((k for k in candidates if k in history and len(history[k])), None)


@_no_usetex
def plot_sngp_history(history: Mapping[str, Sequence[float]], *, name: str = "sngp_history",
                      out_dir: Optional[str] = None, formats: Sequence[str] = ("png",),
                      close: bool = True) -> plt.Figure:
    """SNGP training history: loss/val_loss and accuracy/val_accuracy vs epoch.

    Accepts Keras history keys ('sparse_categorical_accuracy', 'accuracy' or 'acc' and
    their 'val_' versions).
    """
    acc_names = ("sparse_categorical_accuracy", "accuracy", "acc")
    acc = _first_key(history, acc_names)
    vacc = _first_key(history, ["val_" + a for a in acc_names])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for key, lab, c in (("loss", "train", "tab:blue"), ("val_loss", "validation", "tab:orange")):
        if key in history and len(history[key]):
            a1.plot(np.arange(1, len(history[key]) + 1), history[key], c=c, label=lab)
    a1.set_xlabel("epoch")
    a1.set_ylabel("loss (SCCE)")
    a1.set_yscale("log")
    a1.grid(True, alpha=0.3)
    a1.legend()
    for key, lab, c in ((acc, "train", "tab:blue"), (vacc, "validation", "tab:orange")):
        if key is not None:
            a2.plot(np.arange(1, len(history[key]) + 1), history[key], c=c, label=lab)
    a2.set_xlabel("epoch")
    a2.set_ylabel("accuracy")
    a2.grid(True, alpha=0.3)
    if acc or vacc:
        a2.legend()
    fig.tight_layout()
    return _finish(fig, out_dir, name, formats, close)


@_no_usetex
def plot_error_history(history: Mapping[str, Sequence[float]], best_epoch: Optional[int] = None, *,
                       name: str = "error_history", out_dir: Optional[str] = None,
                       formats: Sequence[str] = ("png",), close: bool = True) -> plt.Figure:
    """Error-model history: NLL loss/val_loss (linear: NLL can be negative) and lr on a twin axis.

    Args:
        history: keys 'loss', 'val_loss', optional 'lr' (per epoch).
        best_epoch: 1-based epoch restored by early stopping (vertical marker), or None.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, lab, c in (("loss", "train loss", "tab:blue"), ("val_loss", "validation loss", "tab:orange")):
        if key in history and len(history[key]):
            ax.plot(np.arange(1, len(history[key]) + 1), history[key], c=c, lw=1, label=lab)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="k", ls=":", lw=1, label=f"best epoch {best_epoch}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (Gaussian NLL, normalised)")
    ax.grid(True, alpha=0.3)
    handles, labs = ax.get_legend_handles_labels()
    if "lr" in history and len(history["lr"]):
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(history["lr"]) + 1), history["lr"], c="tab:green", lw=1, label="learning rate")
        ax2.set_ylabel("learning rate", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labs = handles + h2, labs + l2
    ax.legend(handles, labs, loc="upper right")
    fig.tight_layout()
    return _finish(fig, out_dir, name, formats, close)


# ---------------------------------------------------------------------------
# A-priori benchmark (tools/apriori_benchmark.py; EVB / LOG / AIAA notebooks)
# ---------------------------------------------------------------------------

# EVB CASE_MARKER_MAP (lower-case case keys), verbatim.
CASE_MARKER_MAP: Dict[str, str] = {
    "ch": "$c$", "syn": "$s$", "ch_rot": "$r$", "pipe": "$p$", "duct": "$d$",
    "convdiv": "d", "diffuser": "$3$", "swept_wing": "8", "smoothramp": "p",
    "round_step": "s", "hump": "h", "ph_b": "1", "backstep": "+", "backstep_inc": "+",
    "apg_gasser": "H", "transition": "P", "bub_a": "4", "bub_b": "4", "bub_c": "4",
    "bub_k": "4", "bub_c35": "|",
}

# EVB CASE_MARKER_RULES, verbatim and in order (first regex match wins; some later
# entries, e.g. 'apg_gasser' after 'apg_', are unreachable, as in the notebook).
CASE_MARKER_RULES: List[Tuple[str, str]] = [
    (r"spinning_driver", "x"),
    (r"apg_", "o"),
    (r"ph_", "1"),
    (r"round_step", "s"),
    (r"smoothramp", "p"),
    (r"oscillating pipe", "$p$"),
    (r"pipe", "$p$"),
    (r"hump", "h"),
    (r"swept_wing", "8"),
    (r"convdiv", "d"),
    (r"curved_tbl", "^"),
    (r"axissym", "3"),
    (r"duct", "$d$"),
    (r"bend", "2"),
    (r"^naca", "$n$"),
    (r"naca", "$n$"),
    (r"^aairfoil", "$a$"),
    (r"aairfoil", "$a$"),
    (r"tbl_gungor", "$g$"),
    (r"fs_", "$l$"),
    (r"falkner", "$l$"),
    (r"diffuser", "$3$"),
    (r"backstep", "+"),
    (r"backstep_inc", "+"),
    (r"tbl_volino", ">"),
    (r"strained_tbl", "<"),
    (r"gaussian", "*"),
    (r"apg_gasser", "H"),
    (r"tbl_", "v"),
    (r"transition_coupland", "X"),
]


def get_case_marker(case_name: Optional[str], rename_mapping: Optional[Mapping[str, str]] = None,
                    default_marker: str = "o") -> str:
    """Marker for a case (EVB ``get_case_marker``, exact port).

    Order: lower-case direct CASE_MARKER_MAP lookup; else reverse ``rename_mapping``
    (raw key -> display name) to the raw key and look that up; else the first
    CASE_MARKER_RULES regex matching the (raw, if found) lower-case name; else default.
    """
    if case_name is None:
        return default_marker
    name = str(case_name).strip()
    if not name:
        return default_marker
    name_lower = name.lower()
    if name_lower in CASE_MARKER_MAP:
        return CASE_MARKER_MAP[name_lower]
    reverse_map = {v: k for k, v in (rename_mapping or {}).items()}
    raw_name = reverse_map.get(name) if reverse_map else None
    if raw_name:
        raw_lower = raw_name.lower()
        if raw_lower in CASE_MARKER_MAP:
            return CASE_MARKER_MAP[raw_lower]
        name_lower = raw_lower
    for pattern, marker in CASE_MARKER_RULES:
        if re.search(pattern, name_lower):
            return marker
    return default_marker


ROLE_COLORS = {True: "blue", False: "maroon"}


def _col(cases, key, n=None, required=True):
    if key in cases and cases[key] is not None:
        return np.asarray(cases[key])
    if required:
        raise KeyError(f"scatter_cases: cases has no '{key}' column")
    return None


@_no_usetex
def scatter_cases(ax, cases: Mapping[str, Sequence], color_mode: str = "role", scale: str = "linear",
                  limits: Optional[Sequence[float]] = (-0.5, 50.0),
                  y_limits: Optional[Sequence[float]] = None, *, box: Optional[Sequence[float]] = None,
                  legend: bool = False, color_range: Optional[Sequence[float]] = None,
                  rename_mapping: Optional[Mapping[str, str]] = None, colorbar: bool = True,
                  ylabel: str = "Expert Model Error (%)",
                  xlabel: str = "Baseline Error (%)") -> List[str]:
    """One marker per case at (mean |e_base%|, mean |e_final%|) (EVB F1-F4 and LOG variants).

    Args:
        ax: target axes.
        cases: dict-like (dict of arrays or DataFrame) with columns 'name' (display name),
            'baseline', 'final' (mean |e%| per case), 'training' (bool; role mode),
            'sigma' (mean sigma %; sigma mode), 'confidence' (mean C; confidence mode).
        color_mode: 'role' (training blue / untrained maroon), 'sigma' (jet over
            color_range, default [0, 30], horizontal colourbar) or 'confidence' (jet over
            color_range, default [0.8, 1]).
        scale: 'linear' or 'log'. Log (LOG notebook rules): non-finite or non-positive
            points are skipped with a warning listing them; a limit <= 0 is replaced by the
            min/max of the positive plotted values; the y = x line spans the full range;
            the box is drawn only if both box values are > 0.
        limits: x (and default y) limits; None = auto (linear: [0, max + 10]).
        y_limits: separate y limits (default = limits).
        box: [a, b] dash-dot square (zoom region); linear always, log only if a, b > 0.
        legend: role legend (Training / Untrained / Same Performance), upper left.
        color_range: colour-scale range for 'sigma'/'confidence'.
        rename_mapping: raw key -> display name, for the marker reverse lookup.
        colorbar: add the horizontal colourbar in 'sigma'/'confidence' modes.
        ylabel, xlabel: axis labels (plain text, rendered as mathrm).

    Returns:
        The names of skipped (not drawn) cases.
    """
    if color_mode not in ("role", "sigma", "confidence"):
        raise ValueError(f"scatter_cases: color_mode must be role|sigma|confidence, got {color_mode!r}")
    if scale not in ("linear", "log"):
        raise ValueError(f"scatter_cases: scale must be linear|log, got {scale!r}")
    names = [str(n) for n in _col(cases, "name")]
    xb = np.asarray(_col(cases, "baseline"), dtype=float)
    yf = np.asarray(_col(cases, "final"), dtype=float)
    if color_mode == "role":
        roles = _col(cases, "training", required=False)
        roles = np.zeros(len(names), dtype=bool) if roles is None else np.asarray(roles, dtype=bool)
        cvals, cmap, norm = None, None, None
    else:
        key = "sigma" if color_mode == "sigma" else "confidence"
        cvals = np.asarray(_col(cases, key), dtype=float)
        rng = color_range or ((0.0, 30.0) if key == "sigma" else (0.8, 1.0))
        cmap, norm = plt.cm.jet, Normalize(vmin=rng[0], vmax=rng[1])

    finite = np.isfinite(xb) & np.isfinite(yf)
    drawable = finite & (xb > 0) & (yf > 0) if scale == "log" else finite
    skipped = [n for n, d in zip(names, drawable) if not d]
    if skipped:
        msg = (f"scatter_cases ({scale}): skipping {len(skipped)} case(s) with "
               f"{'non-positive or ' if scale == 'log' else ''}non-finite errors: {', '.join(skipped)}")
        warnings.warn(msg)
    if not drawable.any():
        raise ValueError(f"scatter_cases: no drawable cases for scale={scale}")

    for i, nm in enumerate(names):
        if not drawable[i]:
            continue
        marker = get_case_marker(nm, rename_mapping)
        color = ROLE_COLORS[bool(roles[i])] if cvals is None else cmap(norm(cvals[i]))
        ax.scatter(xb[i], yf[i], s=70, alpha=0.7, marker=marker, color=color,
                   edgecolor=None, linewidth=1.0, label=nm)

    y_limits = limits if y_limits is None or len(y_limits) == 0 else y_limits
    if scale == "log":
        pos = np.concatenate([xb[drawable], yf[drawable]])
        lo, hi = float(pos.min()), float(pos.max())
        lim = list(limits) if limits is not None else [0.0, 0.0]
        ylm = list(y_limits) if y_limits is not None else [0.0, 0.0]
        x_min = lim[0] if lim[0] > 0 else lo
        x_max = lim[1] if lim[1] > 0 else hi
        y_min = ylm[0] if ylm[0] > 0 else lo
        y_max = ylm[1] if ylm[1] > 0 else hi
        d0, d1 = min(x_min, y_min), max(x_max, y_max)
        ax.plot([d0, d1], [d0, d1], "k--", alpha=1.0, linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        draw_box = box is not None and box[0] > 0 and box[1] > 0
    else:
        max_val = max(np.nanmax(xb[drawable]) + 10, np.nanmax(yf[drawable]) + 10)
        ax.plot([0, max_val], [0, max_val], "k--", alpha=1.0, linewidth=1)
        if limits is None:
            x_min, x_max = 0.0, max_val
        else:
            x_min, x_max = limits
        y_min, y_max = (x_min, x_max) if y_limits is None else y_limits
        draw_box = box is not None
    if draw_box:
        a, b = box
        ax.plot([a, b, b, a, a], [a, a, b, b, a], "k-.", alpha=1.0, linewidth=1)

    ax.set_xlabel(mathrm(xlabel), fontsize=24)
    ax.set_ylabel(mathrm(ylabel), fontsize=24)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_aspect(0.85)

    if color_mode == "role" and legend:
        handles = [
            Line2D([0], [0], marker="o", color="w", label=r"$\mathrm{Training\ Cases}$",
                   markerfacecolor=ROLE_COLORS[True], markersize=14, alpha=0.7),
            Line2D([0], [0], marker="o", color="w", label=r"$\mathrm{Untrained\ Cases}$",
                   markerfacecolor=ROLE_COLORS[False], markersize=14, alpha=0.7),
            Line2D([0], [0], linestyle="--", color="k", label=r"$\mathrm{Same\ Performance}$"),
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=22)
    if cvals is not None and colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        label = r"$\mathrm{Aleatoric\ Error\ (\%)}$" if color_mode == "sigma" else r"$C$"
        cbar = ax.figure.colorbar(sm, ax=ax, label=label, orientation="horizontal", shrink=0.5)
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(label, fontsize=20)
    return skipped


_BENCH_SCATTER = ("train_test", "train_test_zoom", "sigma", "confidence")


@_no_usetex
def benchmark_scatter_figures(cases: Mapping[str, Sequence], opts: Mapping, run_name: str, *,
                              out_dir: Optional[str] = None, formats: Sequence[str] = ("pdf", "png"),
                              scale: Optional[str] = None,
                              rename_mapping: Optional[Mapping[str, str]] = None,
                              close: bool = True) -> Dict[str, plt.Figure]:
    """Benchmark scatter figures F1 (train_test), F2 (train_test_zoom), F3 (sigma), F4 (confidence).

    Args:
        cases: per-case table, columns as in :func:`scatter_cases`.
        opts: benchmark ``[plots]``: Scale ('both'), LinearLimits ([-0.5, 50]),
            ZoomLimits ([0, 10]; F2 limits and the F1 linear box), LogLimits ([0.1, 100]),
            LogYLimits ([] = LogLimits), YLabel ('Expert Model Error (%)'), Legend (true),
            SigmaColorRange ([0, 30]), ConfidenceColorRange ([0.8, 1]), Figures (all four
            scatter names; PDF names are ignored here).
        run_name: file-name stem; files '<RunName>_<fig>[_log]'.
        out_dir, formats: saving.
        scale: override of opts Scale ('linear' | 'log' | 'both').
        rename_mapping: raw key -> display name (marker lookup).
        close: close figures after saving.

    Returns:
        Dict '<fig>[_log]' -> figure. F2 is linear only.
    """
    scale = str(scale or opts.get("Scale", "both"))
    scales = ["linear", "log"] if scale == "both" else [scale]
    wanted = [f for f in opts.get("Figures", list(_BENCH_SCATTER)) if f in _BENCH_SCATTER]
    lin = list(opts.get("LinearLimits", [-0.5, 50.0]))
    zoom = list(opts.get("ZoomLimits", [0.0, 10.0]))
    loglim = list(opts.get("LogLimits", [0.1, 100.0]))
    logy = list(opts.get("LogYLimits", [])) or None
    ylabel = str(opts.get("YLabel", "Expert Model Error (%)"))
    legend = bool(opts.get("Legend", True))
    ranges = {"sigma": opts.get("SigmaColorRange", [0.0, 30.0]),
              "confidence": opts.get("ConfidenceColorRange", [0.8, 1.0])}
    figs = {}
    for sc in scales:
        suffix = "_log" if sc == "log" else ""
        for fname in wanted:
            if fname == "train_test_zoom" and sc == "log":
                continue
            fig, ax = plt.subplots(figsize=(8, 8))
            if fname == "train_test":
                kw = dict(color_mode="role", legend=legend, box=zoom if sc == "linear" else None)
            elif fname == "train_test_zoom":
                kw = dict(color_mode="role", legend=False, box=None)
            else:
                kw = dict(color_mode=fname, color_range=ranges[fname])
            if sc == "log":
                limits, ylims = loglim, logy
            else:
                limits, ylims = (zoom if fname == "train_test_zoom" else lin), None
            scatter_cases(ax, cases, scale=sc, limits=limits, y_limits=ylims, ylabel=ylabel,
                          rename_mapping=rename_mapping, **kw)
            fig.tight_layout()
            key = fname + suffix
            figs[key] = _finish(fig, out_dir, f"{run_name}_{key}", formats, close)
    return figs


def _subset_points(points: Mapping[str, np.ndarray], subset: str) -> Dict[str, np.ndarray]:
    if subset == "all":
        return dict(points)
    if subset not in ("train", "test"):
        raise ValueError(f"PdfSubset must be all|train|test, got {subset!r}")
    if points.get("training") is None:
        warnings.warn("PdfSubset requested but points has no 'training' mask; using all points.")
        return dict(points)
    m = np.asarray(points["training"], dtype=bool)
    m = m if subset == "train" else ~m
    return {k: (np.asarray(v)[m] if v is not None and np.size(v) == m.size else v) for k, v in points.items()}


@_no_usetex
def benchmark_pdf_figures(points: Mapping[str, np.ndarray], opts: Mapping, run_name: str, *,
                          out_dir: Optional[str] = None, formats: Sequence[str] = ("pdf", "png"),
                          max_points: int = 1_000_000, close: bool = True) -> Dict[str, plt.Figure]:
    """Benchmark PDFs F6/F7 ('pdf') and F8 ('pdf_always_vs_blended'), pooled over all points.

    Args:
        points: pooled per-point arrays: 'baseline', 'final', 'eqwm', 'always' (e%; abs taken),
            'confidence', optional 'training' (bool, for PdfSubset).
        opts: benchmark ``[plots]``: Figures, PdfMethod ('kde'), PdfErrorRange ([0, 40]),
            PdfConfRange ([0.7, 1]), PdfBins (50), PdfCurves (['baseline', 'final']; optional
            'eqwm', 'always'), PdfSubset ('all' | 'train' | 'test').
        run_name: file stem; files '<RunName>_pdf', '<RunName>_pdf_confidence',
            '<RunName>_pdf_always_vs_blended'.
        out_dir, formats: saving.
        max_points: deterministic thinning cap per curve before the KDE (0 = none).
        close: close figures after saving.

    Returns:
        Dict 'pdf' / 'pdf_confidence' / 'pdf_always_vs_blended' -> figure.
    """
    wanted = list(opts.get("Figures", ["pdf", "pdf_always_vs_blended"]))
    method = str(opts.get("PdfMethod", "kde"))
    x_range = list(opts.get("PdfErrorRange", [0.0, 40.0]))
    conf_range = list(opts.get("PdfConfRange", [0.7, 1.0]))
    bins = int(opts.get("PdfBins", 50))
    pts = _subset_points(points, str(opts.get("PdfSubset", "all")))
    figs = {}
    if "pdf" in wanted:
        curves = {c: pts.get(c) for c in opts.get("PdfCurves", ["baseline", "final"])}
        missing = [c for c, v in curves.items() if v is None]
        if missing:
            warnings.warn(f"benchmark PDFs: no data for curves {missing}; skipped")
        f_err, f_conf = plot_pdfs({c: v for c, v in curves.items() if v is not None},
                                  pts.get("confidence"),
                                  {"Method": method, "XRange": x_range, "ConfRange": conf_range, "Bins": bins},
                                  prefix="", out_dir=None, max_points=max_points, close=False)
        figs["pdf"] = _finish(f_err, out_dir, f"{run_name}_pdf", formats, close)
        if f_conf is not None:
            figs["pdf_confidence"] = _finish(f_conf, out_dir, f"{run_name}_pdf_confidence", formats, close)
    if "pdf_always_vs_blended" in wanted:
        if pts.get("always") is None:
            warnings.warn("benchmark: 'pdf_always_vs_blended' needs the 'always' errors (CompareAlways); skipped")
        else:
            figs["pdf_always_vs_blended"] = plot_pdf_always_vs_blended(
                pts["baseline"], pts["always"], pts["final"], x_range=x_range, method=method,
                bins=bins, name=f"{run_name}_pdf_always_vs_blended", out_dir=out_dir,
                formats=formats, max_points=max_points, close=close)
    return figs


@_no_usetex
def plot_pdf_always_vs_blended(e_base: np.ndarray, e_always: np.ndarray, e_final: np.ndarray, *,
                               x_range: Sequence[float] = (0.0, 12.0), method: str = "kde",
                               bins: int = 50, name: str = "pdf_always_vs_blended",
                               out_dir: Optional[str] = None, formats: Sequence[str] = ("png",),
                               max_points: int = 1_000_000, close: bool = True) -> plt.Figure:
    """Baseline vs always-corrected vs blended error PDFs (AIAA ``plot_error_pdf_no_filter_vs_filter``).

    Both corrected curves come from the same prediction pass (the AIAA notebook needed
    two kernel runs). Values are |e%|, finite and inside x_range (conditional densities).
    """
    curves = [
        (e_base, "steelblue", "darkblue", r"$\mathrm{Baseline}$"),
        (e_always, "coral", "darkred", r"$\mathrm{Error\ Model\ (always)}$"),
        (e_final, "yellow", "goldenrod", r"$\mathrm{Error\ Model\ (blended)}$"),
    ]
    print(f"PDF always vs blended ({method}), x_range={list(x_range)}:")
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_pdf_curves(ax, curves, list(x_range), method, bins, 0.35, 1.5, max_points, 18)
    _pdf_axes_style(ax, r"$\mathrm{Model\ Error\ (\%)}$", list(x_range), 20)
    fig.tight_layout()
    return _finish(fig, out_dir, name, formats, close)
