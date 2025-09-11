"""BB code LER vs rounds (decoder-based).

Builds a BB code from polynomials, generates repeated-syndrome Stim circuits,
and estimates post-decoding logical error rate (LER) vs rounds using BpOsd.

Usage examples
--------------

    # Quick default sweep
    python bb_ler_vs_rounds.py

    # Custom code, noise, rounds, and output
    python bb_ler_vs_rounds.py \
        --a-poly "[[3,0],[0,1],[0,2]]" \
        --b-poly "[[0,3],[1,0],[2,0]]" \
        --l 6 --m 6 \
        --p-list 0.003 0.006 \
        --rounds 1 2 4 8 \
        --shots 20000 --max-errors 200 \
        --bp-iters 30 --osd-order 3 \
        --csv Data/bb_ler_vs_rounds.csv

    python bb_ler_vs_rounds.py --rounds 1 2 3 4 --p-list 0.006 --shots 5000 --max-errors 500 --plot-per-round bb_per_round.png

Notes
-----
- This script mirrors the serial path in `simulation_generic.py` but for BB codes.
- `p2` (two-qubit) and `p_spam` are set to `p`; `p1` is set to `p/10`.
"""

from __future__ import annotations

import argparse
import os
import json
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import stim

from ..quantum_circuit_builder import generate_full_circuit
from ..qec_simulation_core import (
    build_bb_code,
    build_decoder_from_circuit,
    append_resume_csv,
)


@dataclass
class Point:
    rounds: int
    p: float
    shots: int
    errors: int
    seconds: float
    per_round: Optional[float] = None

    @property
    def ler(self) -> float:
        return float(self.errors) / max(1, int(self.shots))


def _parse_poly(s: str) -> list:
    try:
        val = json.loads(s)
        if not isinstance(val, list):
            raise ValueError("poly must be a list of [i,j] terms")
        return val
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"invalid poly JSON: {e}")


def run_bb_rounds_sweep(
    *,
    a_poly: list,
    b_poly: list,
    l: int,
    m: int,
    p_list: Iterable[float],
    rounds_list: Iterable[int],
    shots: int,
    max_errors: Optional[int],
    seed: Optional[int],
    bp_iters: int,
    osd_order: int,
    decompose_dem: Optional[bool] = None,
    use_css_splitting: bool = False,
    resume_csv: Optional[str] = None,
) -> List[Point]:
    code = build_bb_code(a_poly, b_poly, l, m, estimate_distance=False)
    rng = np.random.default_rng(seed)
    out: List[Point] = []

    for rounds in rounds_list:
        for p in p_list:
            # Build circuit using the canonical generator to preserve detector ordering
            # circuit = generate_full_circuit(
            #     code=code,
            #     rounds=int(rounds),
            #     p1=float(p) / 10.0,
            #     p2=float(p),
            #     p_spam=float(p),
            #     seed=int(rng.integers(0, 2**32 - 1)) if seed is not None else 0,
            # )

            circuit = stim.Circuit.from_file("../r=6,d=6,p=0.001,noise=si1000,c=bivariate_bicycle_Z,nkd=[[72,12,6]],q=144,iscolored=True,A_poly=x^3+y+y^2,B_poly=y^3+x+x^2.stim")
            # Modify the REPEAT count based on rounds parameter
            circuit_str = str(circuit)
            # Replace any REPEAT N with REPEAT (rounds-1)
            import re
            repeat_pattern = r'REPEAT\s+(\d+)\s*{'
            circuit_str = re.sub(repeat_pattern, f'REPEAT {rounds-1} {{', circuit_str)
            circuit = stim.Circuit(circuit_str)

            # Build decoder (BB codes don't have single-shot functionality)
            decoder, obs_mat = build_decoder_from_circuit(
                circuit,
                bp_iters=int(bp_iters),
                osd_order=int(osd_order),
                decompose_dem=decompose_dem,
                code=code,
                p=float(p),
                use_bt_singleshot=False,  # BB codes don't support single-shot
                use_css_splitting=use_css_splitting,
            )

            # Sample and decode with early stopping
            sampler = circuit.compile_detector_sampler()
            errors = 0
            used = 0
            t_all0 = time.time()
            batch = min(200, max(1, shots // 20))  # small batches for responsiveness
            while used < shots and (max_errors is None or errors < int(max_errors)):
                t = min(batch, shots - used)
                dets, obs = sampler.sample(shots=t, separate_observables=True, bit_packed=False)
                batch_err = 0
                for i in range(t):
                    corr = decoder.decode(dets[i, :])
                    pred = (obs_mat @ corr) % 2
                    if np.any(pred != obs[i, :]):
                        batch_err += 1
                    if max_errors is not None and (errors + batch_err) >= int(max_errors):
                        used += (i + 1)
                        break
                else:
                    used += t
                errors += batch_err
            seconds = time.time() - t_all0

            ler = errors / max(1, used)
            # Per-round transform: 1 - (1 - ler)**(1/rounds)
            pr = 1.0 - (1.0 - float(ler)) ** (1.0 / float(max(1, int(rounds))))
            out.append(
                Point(
                    rounds=int(rounds),
                    p=float(p),
                    shots=int(used),
                    errors=int(errors),
                    seconds=float(seconds),
                    per_round=float(pr),
                )
            )
            print(
                f"BB r={rounds:<3d} p={p:.4g} shots={used:<7d} errors={errors:<5d} LER={ler:.6f} per_round={pr:.6f} time={seconds:.2f}s"
            )

            if resume_csv is not None and used > 0:
                meta = {
                    "code_type": "BB",
                    "code_l": int(l),
                    "code_m": int(m),
                    "code_k": int(getattr(code, "K", -1)),
                    "code_n": int(getattr(code, "N", -1)),
                    "p": float(p),
                    "rounds": int(rounds),
                    "code_a_poly": a_poly,
                    "code_b_poly": b_poly,
                }
                append_resume_csv(
                    resume_csv,
                    shots=used,
                    errors=errors,
                    seconds=seconds,
                    decoder="bposd",
                    json_metadata=meta,
                )

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="BB code LER vs rounds (BpOsd)")
    ap.add_argument("--a-poly", type=_parse_poly, default="[[3,0],[0,1],[0,2]]", help="JSON list of [i,j] for a(x,y)")
    ap.add_argument("--b-poly", type=_parse_poly, default="[[0,3],[1,0],[2,0]]", help="JSON list of [i,j] for b(x,y)")
    ap.add_argument("--l", type=int, default=6, help="L dimension")
    ap.add_argument("--m", type=int, default=6, help="M dimension")
    ap.add_argument("--p-list", type=float, nargs="+", default=[0.003], help="Physical error rates to sweep")
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 4, 8], help="Rounds to sweep")
    ap.add_argument("--shots", type=int, default=20000, help="Max shots per point")
    ap.add_argument("--max-errors", type=int, default=200, help="Early stop after this many errors")
    ap.add_argument("--bp-iters", type=int, default=20, help="BP iterations for BpOsd")
    ap.add_argument("--osd-order", type=int, default=0, help="OSD order for BpOsd")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (None for nondeterministic)")
    ap.add_argument("--csv", type=str, default=None, help="Optional resume-style CSV output path")
    ap.add_argument("--plot-per-round", type=str, default=None, help="Path to save per-round plot PNG")
    ap.add_argument("--dem-decompose", action="store_true", help="Decompose DEM hyperedges for decoder")
    ap.add_argument("--force-css", action="store_true", help="Force-enable CSS splitting (unsupported; may give incorrect results)")
    args = ap.parse_args()

    if args.force_css:
        os.environ["QEC_FORCE_CSS_SPLITTING"] = "1"

    seed = None if args.seed is None else int(args.seed)
    points = run_bb_rounds_sweep(
        a_poly=args.a_poly,
        b_poly=args.b_poly,
        l=args.l,
        m=args.m,
        p_list=args.p_list,
        rounds_list=args.rounds,
        shots=args.shots,
        max_errors=args.max_errors,
        seed=seed,
        bp_iters=args.bp_iters,
        osd_order=args.osd_order,
        decompose_dem=True if args.dem_decompose else None,
        use_css_splitting=True,
        resume_csv=args.csv,
    )

    # Optional plotting
    def _maybe_plot(points: List[Point], out_path: Optional[str], *, mode: str) -> None:
        if not out_path:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipping plot")
            return
        # Group by physical error rate p
        by_p = {}
        for pt in points:
            by_p.setdefault(pt.p, []).append(pt)
        plt.figure(figsize=(7.5, 5.0), dpi=120)
        for p, pts in sorted(by_p.items()):
            pts = sorted(pts, key=lambda q: q.rounds)
            xs = [q.rounds for q in pts]
            if mode == "per_round":
                ys = [float(q.per_round) if q.per_round is not None else 0.0 for q in pts]
                ylabel = "Per-round logical error"
                title = "BB code: per-round LER vs rounds (grouped by p)"
            else:
                ys = [q.errors / max(1, q.shots) for q in pts]
                ylabel = "Logical error rate"
                title = "BB code: LER vs rounds (grouped by p)"
            plt.plot(xs, ys, marker="o", label=f"p={p}")
        plt.xscale("log", base=2)
        plt.xlabel("Rounds")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which="both", ls=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")

    _maybe_plot(points, args.plot_per_round if hasattr(args, 'plot_per_round') else None, mode="per_round")

    # Summary
    # Group by p
    by_p = {}
    for pt in points:
        by_p.setdefault(pt.p, []).append(pt)
    for p, pts in sorted(by_p.items()):
        pts = sorted(pts, key=lambda q: q.rounds)
        line = ", ".join(f"r={q.rounds}: {q.ler:.4g} (pr={q.per_round:.4g})" for q in pts)
        print(f"Summary p={p:.4g}: {line}")


if __name__ == "__main__":
    main()
