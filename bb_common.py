"""Shared utilities for BB code simulations (serial and multiprocess).

Provides common data structures, code/decoder builders, CSV helpers, and
plotting so scripts stay small and consistent.
"""

from __future__ import annotations

import os
import json
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
    header = (
        "shots,errors,discards,seconds,decoder,json_metadata,custom_counts\n"
    )
    if lock is not None:
        try:
            lock.acquire()
        except Exception:
            pass
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w") as f:
                f.write(header)
        meta_json = json.dumps(json_metadata, separators=(",", ":"))
        counts_json = json.dumps(custom_counts or {}, separators=(",", ":"))
        line = f"{shots},{errors},0,{seconds:.6g},{decoder},\"{meta_json}\",\"{counts_json}\"\n"
        with open(path, "a") as f:
            f.write(line)
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


def save_summary_csv(points: List[ResultPoint], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("p,rounds,shots,errors,ler,seconds\n")
        for r in points:
            f.write(f"{r.p},{r.rounds},{r.shots},{r.errors},{r.ler},{r.seconds}\n")


def plot_points(points: List[ResultPoint], *, out_png: str | None = None, show: bool = False) -> None:
    # Group by (l,m,decoder) and plot per rounds
    by_group: Dict[Tuple[int, int, str], List[ResultPoint]] = {}
    for r in points:
        by_group.setdefault((r.l, r.m, r.decoder), []).append(r)

    for (l, m, dec), pts in by_group.items():
        import numpy as _np

        pts = list(pts)
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        by_rounds: Dict[int, List[ResultPoint]] = {}
        for p in pts:
            by_rounds.setdefault(p.rounds, []).append(p)
        for rounds, rows in sorted(by_rounds.items()):
            rows = sorted(rows, key=lambda x: x.p)
            xs = [r.p for r in rows]
            ys = [r.ler for r in rows]
            ax.plot(xs, ys, marker="o", label=f"rounds={rounds}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Physical Error Rate p")
        ax.set_ylabel("Logical Error Rate (any obs)")
        ax.set_title(f"BB {l}x{m} ({dec})")
        ax.grid(True, which="both", ls=":")
        ax.legend()
        path = out_png
        if path is None:
            path = f"Data/bb_{l}_{m}_parsed_results.png"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        plt.close(fig)
