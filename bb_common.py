"""Shared utilities for BB code simulations (serial and multiprocess).

Provides common data structures, code/decoder builders, CSV helpers, and
plotting so scripts stay small and consistent.
"""

from __future__ import annotations

import os
import json
import csv
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any, Protocol

import numpy as np
import stim
from matplotlib import pyplot as plt

from bposd.css import css_code
from ldpc.bposd_decoder import BpOsdDecoder
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices


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

    @property
    def ler(self) -> float:
        return self.errors / max(1, self.shots)


def build_bb_code(a_poly: list, b_poly: list, l: int, m: int) -> css_code:
    from BB_tools import get_BB_Hx_Hz  # local import to avoid cycles

    Hx, Hz = get_BB_Hx_Hz(a_poly, b_poly, l, m)
    code = css_code(hx=Hx, hz=Hz, name=f"BB_{l}x{m}")
    code.test()
    return code


def build_decoder_from_circuit(
    circuit: stim.Circuit, *, bp_iters: int, osd_order: int
) -> Tuple[BpOsdDecoder, np.ndarray]:
    dem = circuit.detector_error_model(decompose_errors=False)
    mats = detector_error_model_to_check_matrices(
        dem, allow_undecomposed_hyperedges=True
    )
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
        try:
            lock.acquire()
        except Exception:
            pass
    try:
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            meta_json = json.dumps(json_metadata, separators=(",", ":"))
            counts_json = json.dumps(custom_counts or {}, separators=(",", ":"))
            writer.writerow([
                int(shots),
                int(errors),
                0,
                float(seconds),
                str(decoder),
                meta_json,
                counts_json,
            ])
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


def save_summary_csv(points: List[ResultPoint], path: str, *, meta_common: Optional[Dict[str, Any]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = ["p", "rounds", "shots", "errors", "ler", "seconds", "decoder", "json_metadata"]
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
            writer.writerow([
                r.p,
                r.rounds,
                r.shots,
                r.errors,
                r.ler,
                r.seconds,
                r.decoder,
                json.dumps(meta, separators=(",", ":")),
            ])


def plot_points(
    points: List[ResultPoint], *, out_png: str | None = None, show: bool = False,
    y_mode: str = "ler", K: int | None = None
) -> None:
    # Pretty plotting similar to sinter's plot_error_rate with CIs.
    # y_mode: 'ler' (default), 'per_logical', or 'per_round'.

    def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
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

    # Group by (l,m,decoder)
    by_group: Dict[Tuple[int, int, str], List[ResultPoint]] = {}
    for r in points:
        by_group.setdefault((r.l, r.m, r.decoder), []).append(r)

    # Style tweaks
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    markers = ["o", "s", "^", "D", "P", "v", "+", "x", "*", "h"]
    cmap = plt.get_cmap("tab10")

    for (l, m, dec), pts in by_group.items():
        pts = list(pts)
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
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
                # Nothing to plot for this rounds group
                continue
            xs = [r.p for r in rows_nonzero]

            # Transform function for selected y_mode
            def _transform(p_any: float, r_rounds: int) -> float:
                if y_mode == "ler":
                    return p_any
                if y_mode == "per_round":
                    rr = max(1, int(r_rounds) if r_rounds and r_rounds > 0 else 1)
                    return 1.0 - (1.0 - p_any) ** (1.0 / rr)
                if y_mode == "per_logical":
                    if not K or K <= 0:
                        raise ValueError("K must be a positive integer for y_mode='per_logical'.")
                    return 1.0 - (1.0 - p_any) ** (1.0 / K)
                raise ValueError(f"Unknown y_mode: {y_mode}")

            ys = [_transform(r.ler, rounds) for r in rows_nonzero]

            # Wilson CI shaded band (transform bounds through monotonic mapping)
            lo_list = []
            hi_list = []
            for r in rows_nonzero:
                lo, hi = _wilson_interval(r.errors, r.shots)
                lo_t = max(1e-15, _transform(lo, rounds))
                hi_t = max(1e-15, _transform(hi, rounds))
                lo_list.append(lo_t)
                hi_list.append(hi_t)
            color = cmap(idx % 10)
            marker = markers[idx % len(markers)]
            ax.plot(xs, ys, marker=marker, linestyle="-", color=color, linewidth=1.5, markersize=5, label=f"rounds={rounds}")
            ax.fill_between(xs, lo_list, hi_list, color=color, alpha=0.15, linewidth=0)
            plotted_any = True
            all_x_vals.extend(xs)
            all_y_vals.extend(ys)

        # If nothing plotted (e.g., all points had zero errors), skip this figure to avoid log-scale errors
        if not plotted_any:
            plt.close(fig)
            continue

        # Ensure positive, finite limits before enabling log scale
        if not all_x_vals or not all_y_vals:
            plt.close(fig)
            continue
        x_pos = [v for v in all_x_vals if v > 0]
        y_pos = [v for v in all_y_vals if v > 0]
        if not x_pos or not y_pos:
            plt.close(fig)
            continue
        x_min = min(x_pos)
        x_max = max(x_pos)
        y_min = min(y_pos)
        y_max = max(y_pos)
        # Expand slightly to avoid degenerate ranges
        ax.set_xlim(x_min * 0.9, x_max * 1.1)
        ax.set_ylim(max(1e-15, y_min * 0.8), y_max * 1.25)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical error rate p")
        y_label = {
            "ler": "Logical error rate",
            "per_logical": "Per-logical-operator error rate",
            "per_round": "Per-round error rate",
        }.get(y_mode, "Logical error rate")
        ax.set_ylabel(y_label)
        ax.set_title(f"BB {l}×{m} ({dec})")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        # Tight legend
        ax.legend(frameon=False, fontsize=10)

        # Definition text inside plot
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
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
            )

        # Save
        # Include y_mode suffix in default filename to distinguish variants
        if out_png:
            path = out_png
        else:
            suffix = {"ler": "parsed_results", "per_logical": "per_logical", "per_round": "per_round"}.get(y_mode, "parsed_results")
            path = f"Data/bb_{l}_{m}_{suffix}.png"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=200)
        if show:
            plt.show()
        plt.close(fig)
