"""Core utilities for quantum error correction code simulations.

Provides common data structures, code/decoder builders, CSV helpers, and
plotting functionality. Supports BB (Bivariate Bicycle), BT (Bivariate Tricycle),
TT (Trivariate Tricycle) and other quantum LDPC codes.
"""

from __future__ import annotations

import os
import json
import csv
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any, Protocol, cast

import numpy as np
import stim
from matplotlib import pyplot as plt

from bposd.css import css_code
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
import time
import logging


@dataclass
class ResultPoint:
    decoder: str
    l: int
    m: int
    rounds: int
    p: float
    shots: int
    errors: int
    seconds: float
    # Optional descriptors for plotting/naming
    code_type: str = "BB"
    n: int = -1

    @property
    def ler(self) -> float:
        return self.errors / max(1, self.shots)


def build_bb_code(a_poly: list, b_poly: list, l: int, m: int) -> css_code:
    from bivariate_bicycle_codes import get_BB_Hx_Hz  # local import to avoid cycles

    Hx, Hz = get_BB_Hx_Hz(a_poly, b_poly, l, m)
    code = css_code(hx=Hx, hz=Hz, name=f"BB_{l}x{m}")
    code.test()
    return code


def build_bt_code(a_poly: list, b_poly: list, c_poly: list, l: int, m: int) -> css_code:
    from bivariate_tricycle_codes import get_BT_Hx_Hz  # local import to avoid cycles

    Hx, Hz = get_BT_Hx_Hz(a_poly, b_poly, c_poly, l, m)
    code = css_code(hx=Hx, hz=Hz, name=f"BT_{l}x{m}")
    code.test()
    return code


def build_tt_code(a_poly: list, b_poly: list, c_poly: list, l: int, m: int, n: int) -> css_code:
    from trivariate_tricycle_codes import get_TT_Hx_Hz  # local import to avoid cycles

    Hx, Hz = get_TT_Hx_Hz(a_poly, b_poly, c_poly, l, m, n)
    code = css_code(hx=Hx, hz=Hz, name=f"TT_{l}x{m}x{n}")
    code.test()
    return code


def build_code_generic(code_type: str, **params) -> css_code:
    """Generic code builder dispatcher.
    
    Args:
        code_type: "BB", "BT", or "TT"
        **params: Code-specific parameters
        
    Returns:
        Constructed CSS code
        
    Raises:
        ValueError: If code_type is unknown or required parameters are missing
    """
    code_type = code_type.upper()
    
    if code_type == "BB":
        required = {"a_poly", "b_poly", "l", "m"}
        if not required.issubset(params.keys()):
            raise ValueError(f"BB code requires parameters: {required}")
        return build_bb_code(params["a_poly"], params["b_poly"], params["l"], params["m"])
    
    elif code_type == "BT":
        required = {"a_poly", "b_poly", "c_poly", "l", "m"}
        if not required.issubset(params.keys()):
            raise ValueError(f"BT code requires parameters: {required}")
        return build_bt_code(params["a_poly"], params["b_poly"], params["c_poly"], params["l"], params["m"])
    
    elif code_type == "TT":
        required = {"a_poly", "b_poly", "c_poly", "l", "m", "n"}
        if not required.issubset(params.keys()):
            raise ValueError(f"TT code requires parameters: {required}")
        return build_tt_code(params["a_poly"], params["b_poly"], params["c_poly"], params["l"], params["m"], params["n"])
    
    else:
        raise ValueError(f"Unknown code type: {code_type}. Supported types: BB, BT, TT")


def generate_default_resume_csv(code_type: str, output_dir: str, runner_type: str, **params) -> str:
    """Generate default resume CSV filename based on code type and parameters.
    
    Args:
        code_type: "BB", "BT", or "TT"
        output_dir: Output directory
        runner_type: "serial" or "mp" 
        **params: Code parameters (l, m, n, etc.)
        
    Returns:
        Default resume CSV path
    """
    code_type = code_type.lower()
    
    if code_type == "bb":
        return f"{output_dir}/bb_{params['l']}_{params['m']}_{runner_type}_resume.csv"
    elif code_type == "bt":
        return f"{output_dir}/bt_{params['l']}_{params['m']}_{runner_type}_resume.csv"
    elif code_type == "tt":
        return f"{output_dir}/tt_{params['l']}_{params['m']}_{params['n']}_{runner_type}_resume.csv"
    else:
        raise ValueError(f"Unknown code type: {code_type}")


def extract_code_params_from_config(config: dict) -> Tuple[str, dict]:
    """Extract code type and parameters from configuration.
    
    Returns:
        Tuple of (code_type, code_params_dict)
    """
    code_type = config.get('code_type', 'BB').upper()
    
    code_params = {
        'a_poly': config['a_poly'],
        'b_poly': config['b_poly'],
        'l': config['l'],
        'm': config['m'],
    }
    
    if code_type in ['BT', 'TT']:
        code_params['c_poly'] = config['c_poly']
    
    if code_type == 'TT':
        code_params['n'] = config['n']
        
    return code_type, code_params


def build_decoder_from_circuit(
    circuit: stim.Circuit, *, bp_iters: int, osd_order: int, decompose_dem: Optional[bool] = None
) -> Tuple[BpOsdDecoder, np.ndarray]:
    t0 = time.time()
    # Optional DEM decomposition toggle for speed debugging
    if decompose_dem is None:
        decompose = os.getenv("QEC_DEM_DECOMPOSE")
        decompose_flag = bool(decompose and decompose not in ("0", "false", "False"))
    else:
        decompose_flag = bool(decompose_dem)
    if decompose_flag:
        dem = circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
    else:
        dem = circuit.detector_error_model(decompose_errors=False)
    t1 = time.time()
    mats = detector_error_model_to_check_matrices(
        dem, allow_undecomposed_hyperedges=True
    )
    t2 = time.time()
    try:
        H = mats.check_matrix
        O = mats.observables_matrix
        h_shape = tuple(getattr(H, "shape", ()))
        o_shape = tuple(getattr(O, "shape", ()))
        # Sparsity stats when available
        nnz = getattr(H, "nnz", None)
        row_deg_max = col_deg_max = avg_col_deg = None
        try:
            H_csr = H.tocsr()
            row_deg = np.diff(H_csr.indptr)
            row_deg_max = int(row_deg.max()) if row_deg.size else 0
            # Column degrees via CSC (avoid explicit transpose data copy when large)
            H_csc = H.tocsc()
            col_deg = np.diff(H_csc.indptr)
            col_deg_max = int(col_deg.max()) if col_deg.size else 0
            avg_col_deg = float(col_deg.mean()) if col_deg.size else 0.0
        except Exception:
            pass
        if nnz is not None and row_deg_max is not None and col_deg_max is not None:
            logging.info(
                "[DEC] matrices: H%s nnz=%s row_max=%s col_max=%s col_avg=%.2f, O%s",
                h_shape,
                nnz,
                row_deg_max,
                col_deg_max,
                avg_col_deg if avg_col_deg is not None else 0.0,
                o_shape,
            )
        else:
            logging.info("[DEC] matrices: H%s, O%s", h_shape, o_shape)
    except Exception:
        pass
    osd_method = "osd_e" if osd_order and osd_order > 0 else "osd0"
    bposd = BpOsdDecoder(
        mats.check_matrix,
        error_channel=list(mats.priors),
        max_iter=bp_iters,
        bp_method="ms",
        ms_scaling_factor=0.625,
        schedule="parallel",
        osd_method=osd_method,
        osd_order=osd_order,
    )
    t3 = time.time()
    logging.info(
        "[DEC] DEM build %.2fs | dem->mats %.2fs | BpOsd init %.2fs",
        t1 - t0,
        t2 - t1,
        t3 - t2,
    )
    return bposd, mats.observables_matrix


class _LockLike(Protocol):
    def acquire(self) -> None: ...
    def release(self) -> None: ...


def append_resume_csv(
    path: str,
    *,
    shots: int,
    errors: int,
    seconds: float,
    decoder: str,
    json_metadata: dict,
    custom_counts: Optional[dict] = None,
    lock: Optional[_LockLike] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = [
        "shots",
        "errors",
        "discards",
        "seconds",
        "decoder",
        "json_metadata",
        "custom_counts",
    ]
    if lock is not None:
        lock.acquire()
    try:
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            meta_json = json.dumps(json_metadata, separators=(",", ":"))
            counts_json = json.dumps(custom_counts or {}, separators=(",", ":"))
            writer.writerow(
                [
                    int(shots),
                    int(errors),
                    0,
                    float(seconds),
                    str(decoder),
                    meta_json,
                    counts_json,
                ]
            )
    finally:
        if lock is not None:
            lock.release()


def save_summary_csv(
    points: List[ResultPoint],
    path: str,
    *,
    meta_common: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = [
        "p",
        "rounds",
        "shots",
        "errors",
        "ler",
        "seconds",
        "decoder",
        "json_metadata",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in points:
            meta = {
                "l": int(r.l),
                "m": int(r.m),
                "p": float(r.p),
                "rounds": int(r.rounds),
            }
            if meta_common:
                meta.update(meta_common)
            writer.writerow(
                [
                    r.p,
                    r.rounds,
                    r.shots,
                    r.errors,
                    r.ler,
                    r.seconds,
                    r.decoder,
                    json.dumps(meta, separators=(",", ":")),
                ]
            )


def _wilson_confidence_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson confidence interval for binomial proportion.
    
    Args:
        k: Number of successes (errors)
        n: Number of trials (shots)
        z: Z-score for confidence level (1.96 for 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return lo, hi


def _transform_error_rate(p_any: float, r_rounds: int, y_mode: str, K: int | None = None) -> float:
    """Transform logical error rate based on selected y_mode.
    
    Args:
        p_any: Input logical error rate
        r_rounds: Number of rounds
        y_mode: Transform mode ('ler', 'per_round', 'per_logical')
        K: Number of logical operators (required for 'per_logical' mode)
        
    Returns:
        Transformed error rate
        
    Raises:
        ValueError: If y_mode is invalid or K is missing for 'per_logical'
    """
    if y_mode == "ler":
        return p_any
    if y_mode == "per_round":
        rr = max(1, int(r_rounds) if r_rounds and r_rounds > 0 else 1)
        rr_float = float(rr)
        result = 1.0 - (1.0 - p_any) ** (1.0 / rr_float)
        return cast(float, result)
    if y_mode == "per_logical":
        if not K or K <= 0:
            raise ValueError(
                "K must be a positive integer for y_mode='per_logical'."
            )
        K_float = float(K)
        result = 1.0 - (1.0 - p_any) ** (1.0 / K_float)
        return cast(float, result)
    raise ValueError(f"Unknown y_mode: {y_mode}")


def _setup_plot_style() -> Tuple[List[str], Any]:
    """Setup matplotlib style and return markers and colormap.
    
    Returns:
        Tuple of (markers_list, colormap)
    """
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    
    markers = ["o", "s", "^", "D", "P", "v", "+", "x", "*", "h"]
    cmap = plt.get_cmap("tab10")
    return markers, cmap


def _group_points_by_configuration(points: List[ResultPoint]) -> Dict[Tuple[str, int, int, int, str], List[ResultPoint]]:
    """Group result points by (l, m, decoder) configuration.
    
    Args:
        points: List of result points to group
        
    Returns:
        Dictionary mapping (l, m, decoder) tuples to lists of points
    """
    by_group: Dict[Tuple[str, int, int, int, str], List[ResultPoint]] = {}
    for r in points:
        code_t = r.code_type if getattr(r, "code_type", None) else "BB"
        n_dim = int(getattr(r, "n", -1))
        by_group.setdefault((code_t, int(r.l), int(r.m), n_dim, r.decoder), []).append(r)
    return by_group


def _calculate_plot_bounds(all_x_vals: List[float], all_y_vals: List[float]) -> Optional[Tuple[float, float, float, float]]:
    """Calculate plot bounds ensuring positive values for log scale.
    
    Args:
        all_x_vals: All x-axis values
        all_y_vals: All y-axis values
        
    Returns:
        Tuple of (x_min, x_max, y_min, y_max) or None if no valid bounds
    """
    if not all_x_vals or not all_y_vals:
        return None
    
    x_pos = [v for v in all_x_vals if v > 0]
    y_pos = [v for v in all_y_vals if v > 0]
    if not x_pos or not y_pos:
        return None
    
    x_min = min(x_pos)
    x_max = max(x_pos)
    y_min = min(y_pos)
    y_max = max(y_pos)
    return x_min, x_max, y_min, y_max


def _apply_plot_styling(
    ax: plt.Axes,
    code_type: str,
    l: int,
    m: int,
    n: int | None,
    decoder: str,
    y_mode: str,
    K: int | None = None,
) -> None:
    """Apply styling to a plot axis.
    
    Args:
        ax: Matplotlib axis to style
        l: First BB code parameter
        m: Second BB code parameter  
        decoder: Decoder name
        y_mode: Y-axis mode
        K: Number of logical operators for definition text
    """
    ax.set_xlabel("Physical error rate p")
    y_label = {
        "ler": "Logical error rate",
        "per_logical": "Per-logical-operator error rate",
        "per_round": "Per-round error rate",
    }.get(y_mode, "Logical error rate")
    ax.set_ylabel(y_label)
    ct = (code_type or "BB").upper()
    if ct == "TT" and n is not None and int(n) > 0:
        title = f"{ct} {l}×{m}×{int(n)} ({decoder})"
    else:
        title = f"{ct} {l}×{m} ({decoder})"
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(frameon=False, fontsize=10)

    # Add definition text inside plot
    def_text = None
    if y_mode == "per_logical":
        def_text = f"p_single = 1 - (1 - p_any)^(1/K), K={K}"
    elif y_mode == "per_round":
        def_text = "p_round = 1 - (1 - p_shot)^(1/r), r = rounds"
    if def_text:
        ax.text(
            0.02,
            0.02,
            def_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
            ),
        )


def _save_plot(
    fig: plt.Figure,
    code_type: str,
    l: int,
    m: int,
    n: int | None,
    y_mode: str,
    out_png: str | None,
    show: bool,
) -> None:
    """Save plot to file and optionally display it.
    
    Args:
        fig: Matplotlib figure to save
        l: First BB code parameter for default filename
        m: Second BB code parameter for default filename
        y_mode: Y-axis mode for filename suffix
        out_png: Output path or None for default
        show: Whether to display the plot
    """
    if out_png:
        path = out_png
    else:
        suffix = {
            "ler": "parsed_results",
            "per_logical": "per_logical", 
            "per_round": "per_round",
        }.get(y_mode, "parsed_results")
        ct = (code_type or "bb").lower()
        if (code_type or "BB").upper() == "TT" and n is not None and int(n) > 0:
            path = f"Data/{ct}_{l}_{m}_{int(n)}_{suffix}.png"
        else:
            path = f"Data/{ct}_{l}_{m}_{suffix}.png"
    
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def plot_points(
    points: List[ResultPoint],
    *,
    out_png: str | None = None,
    show: bool = False,
    y_mode: str = "ler",
    K: int | None = None,
) -> None:
    """Generate plots for BB code simulation results with confidence intervals.
    
    Creates separate plots for each (l, m, decoder) configuration, with
    different curves for each number of rounds. Supports multiple y-axis
    transformations and includes Wilson confidence intervals.
    
    Args:
        points: List of result points to plot
        out_png: Output PNG path, or None for default naming
        show: Whether to display plots interactively
        y_mode: Y-axis transform mode ('ler', 'per_round', 'per_logical')
        K: Number of logical operators (required for 'per_logical' mode)
    """
    markers, cmap = _setup_plot_style()
    by_group = _group_points_by_configuration(points)

    for (code_t, l, m, n_dim, dec), pts in by_group.items():
        pts = list(pts)
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        
        # Group by number of rounds
        by_rounds: Dict[int, List[ResultPoint]] = {}
        for p in pts:
            by_rounds.setdefault(p.rounds, []).append(p)

        plotted_any = False
        all_x_vals: List[float] = []
        all_y_vals: List[float] = []
        
        for idx, (rounds, rows) in enumerate(sorted(by_rounds.items())):
            rows = sorted(rows, key=lambda x: x.p)
            # Filter out zero-error points (no observed errors)
            rows_nonzero = [r for r in rows if r.errors > 0]
            if not rows_nonzero:
                continue
            
            xs = [r.p for r in rows_nonzero]
            ys = [_transform_error_rate(r.ler, rounds, y_mode, K) for r in rows_nonzero]

            # Calculate Wilson confidence intervals
            lo_list = []
            hi_list = []
            for r in rows_nonzero:
                lo, hi = _wilson_confidence_interval(r.errors, r.shots)
                lo_t = max(1e-15, _transform_error_rate(lo, rounds, y_mode, K))
                hi_t = max(1e-15, _transform_error_rate(hi, rounds, y_mode, K))
                lo_list.append(lo_t)
                hi_list.append(hi_t)
            
            # Plot data with confidence intervals
            color = cmap(idx % 10)
            marker = markers[idx % len(markers)]
            ax.plot(
                xs,
                ys,
                marker=marker,
                linestyle="-",
                color=color,
                linewidth=1.5,
                markersize=5,
                label=f"rounds={rounds}",
            )
            ax.fill_between(xs, lo_list, hi_list, color=color, alpha=0.15, linewidth=0)
            
            plotted_any = True
            all_x_vals.extend(xs)
            all_y_vals.extend(ys)

        # Skip if nothing to plot
        if not plotted_any:
            plt.close(fig)
            continue

        # Set up log scale with proper bounds
        bounds = _calculate_plot_bounds(all_x_vals, all_y_vals)
        if not bounds:
            plt.close(fig)
            continue
        
        x_min, x_max, y_min, y_max = bounds
        ax.set_xlim(x_min * 0.9, x_max * 1.1)
        ax.set_ylim(max(1e-15, y_min * 0.8), y_max * 1.25)
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        _apply_plot_styling(ax, code_t, l, m, n_dim, dec, y_mode, K)
        _save_plot(fig, code_t, l, m, n_dim, y_mode, out_png, show)
