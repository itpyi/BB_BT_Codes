"""Surface code LER vs distance/rounds using Stim generators and PyMatching MWPM decoder.

Generates rotated surface code Z-memory circuits via `stim.Circuit.generated`
and estimates the logical error rate (LER) using PyMatching's MWPM decoder.
Also reports a per-round transformed rate:

    p_round = 1 - (1 - P_any) ** (1 / rounds)

which is a common way to visualize how risk scales with the number of rounds.

Usage examples
--------------

    # Quick sweep with defaults
    python surface_code_ler_vs_rounds.py

    # Custom distances/rounds and noise (units are probabilities)
    python surface_code_ler_vs_rounds.py \
        --distances 3 \
        --rounds 1 2 4 8 \
        --shots 50000 \
        --p-clifford 0.001 \
        --p-data 0.0005 \
        --p-meas 0.001 \
        --p-reset 0.001 \
        --plot out_surface_code.png

Notes
-----
- This script uses PyMatching's MWPM decoder to decode detector syndromes
  and predict logical observables, then compares with actual observables.
- The per-round transform assumes i.i.d. round contributions and is for trend
  visualization; don't over-interpret it at very small rounds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import stim
import pymatching


@dataclass
class Point:
    distance: int
    rounds: int
    shots: int
    ler_any: float
    per_round: float


def _per_round_transform(p_any: float, rounds: int) -> float:
    r = max(1, int(rounds))
    return float(1.0 - (1.0 - float(p_any)) ** (1.0 / float(r)))


def estimate_ler_for(
    *,
    distance: int,
    rounds: int,
    shots: int,
    p_clifford: float,
    p_data: float,
    p_meas: float,
    p_reset: float,
    seed: Optional[int] = None,
) -> Tuple[int, int, int, float, float]:
    """Build a rotated memory-Z surface code circuit and estimate LER using MWPM decoder.

    Returns (distance, rounds, shots, p_any, p_round).
    """
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=int(distance),
        rounds=int(rounds),
        after_clifford_depolarization=float(p_clifford),
        before_round_data_depolarization=float(p_data),
        before_measure_flip_probability=float(p_meas),
        after_reset_flip_probability=float(p_reset),
    )
    
    # Create MWPM decoder from detector error model
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    
    # Sample syndromes and observables
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True, bit_packed=False)
    
    # Decode syndromes and compare predictions with actual observables
    predictions = matcher.decode_batch(dets)
    
    # Count errors (when prediction != actual observable)
    if obs.ndim == 1:
        errors = np.sum(predictions != obs)
    else:
        # Handle multiple observables by checking if any prediction differs from actual
        errors = np.sum(np.any(predictions != obs, axis=1))
    
    p_any = float(errors) / float(shots)
    p_round = _per_round_transform(p_any, rounds)
    return int(distance), int(rounds), int(shots), p_any, p_round


def run_sweep(
    distances: Iterable[int],
    rounds_list: Iterable[int],
    *,
    shots: int,
    p_clifford: float,
    p_data: float,
    p_meas: float,
    p_reset: float,
    seed: Optional[int] = 0,
) -> List[Point]:
    rng = np.random.default_rng(seed)
    out: List[Point] = []
    for d in distances:
        for r in rounds_list:
            _seed = int(rng.integers(0, 2**32 - 1)) if seed is not None else None
            _, _, _, p_any, p_round = estimate_ler_for(
                distance=d,
                rounds=r,
                shots=shots,
                p_clifford=p_clifford,
                p_data=p_data,
                p_meas=p_meas,
                p_reset=p_reset,
                seed=_seed,
            )
            out.append(Point(distance=int(d), rounds=int(r), shots=int(shots), ler_any=p_any, per_round=p_round))
            print(
                f"d={d:>2} r={r:>3} shots={shots:<7} LER={p_any:.6f} per_round={p_round:.6f}"
            )
    return out


def _maybe_plot(points: List[Point], out_path: Optional[str]) -> None:
    if not out_path:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot")
        return
    # Group by distance
    by_d = {}
    for p in points:
        by_d.setdefault(p.distance, []).append(p)
    plt.figure(figsize=(7.5, 5.0), dpi=120)
    for idx, (d, pts) in enumerate(sorted(by_d.items())):
        pts = sorted(pts, key=lambda q: q.rounds)
        rs = [p.rounds for p in pts]
        ys = [p.per_round for p in pts]
        plt.plot(rs, ys, marker="o", label=f"d={d}")
    plt.xscale("log", base=2)
    plt.xlabel("Rounds")
    plt.ylabel("Per-round logical error")
    plt.title("Rotated surface code Z-memory: per-round LER vs rounds")
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rotated surface code LER vs rounds using Stim generators")
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7], help="Odd distances to sweep")
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="Rounds to sweep")
    ap.add_argument("--shots", type=int, default=20000, help="Shots per point")
    ap.add_argument("--p-clifford", type=float, default=0.001, help="After-Clifford depolarization prob")
    ap.add_argument("--p-data", type=float, default=0.0005, help="Before-round data depolarization prob")
    ap.add_argument("--p-meas", type=float, default=0.001, help="Before-measure flip probability")
    ap.add_argument("--p-reset", type=float, default=0.001, help="After-reset flip probability")
    ap.add_argument("--plot", type=str, default=None, help="Optional path to save per-round plot PNG")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (set None for nondeterministic)")
    args = ap.parse_args()

    pts = run_sweep(
        distances=args.distances,
        rounds_list=args.rounds,
        shots=args.shots,
        p_clifford=args.p_clifford,
        p_data=args.p_data,
        p_meas=args.p_meas,
        p_reset=args.p_reset,
        seed=args.seed,
    )
    _maybe_plot(pts, args.plot)


if __name__ == "__main__":
    main()

