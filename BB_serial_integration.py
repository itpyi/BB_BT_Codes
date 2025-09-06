"""Simple serial BB code runner without sinter.

Generates Stim circuits with detectors, converts to a detector error model,
decodes shots with BPOSD in small chunks to control memory, and plots results.

Tuning parameters: rounds and physical error rate p.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, List, Tuple, Optional
import json

import numpy as np
import stim
from matplotlib import pyplot as plt

from bb_common import (
    ResultPoint,
    build_bb_code,
    build_decoder_from_circuit,
    append_resume_csv,
    save_summary_csv,
    plot_points,
)

from circuit_utils import generate_full_circuit


# Helper removed (not used); decoding happens in early-stop loop below


pass


def run_BB_serial_simulation(
    *,
    a_poly: list,
    b_poly: list,
    l: int,
    m: int,
    p_list: Iterable[float],
    rounds_list: Iterable[int],
    max_shots: Optional[int] = 10_000,
    max_errors: Optional[int] = None,
    seed: int = 0,
    bp_iters: int = 20,
    osd_order: int = 3,
    resume_csv: Optional[str] = None,
    resume_every: int = 200,
) -> List[ResultPoint]:
    code = build_bb_code(a_poly, b_poly, l, m)
    rng = np.random.default_rng(seed)
    results: List[ResultPoint] = []
    if resume_csv is None:
        resume_csv = f"Data/bb_{l}_{m}_serial_resume.csv"

    for rounds in rounds_list:
        for p in p_list:
            circuit = generate_full_circuit(
                code=code,
                rounds=rounds,
                p1=p / 10.0,
                p2=p,
                p_spam=p,
                seed=int(rng.integers(0, 2**32 - 1)),
            )

            decoder, obs_mat = build_decoder_from_circuit(
                circuit, bp_iters=bp_iters, osd_order=osd_order
            )

            # Prepare metadata for resume CSV
            meta = {
                "a_poly": str(a_poly),
                "b_poly": str(b_poly),
                "code_k": code.K,
                "code_n": code.N,
                "l": l,
                "m": m,
                "p": float(p),
                "rounds": int(rounds),
            }

            # Decode in batches with early stop criteria
            sampler = circuit.compile_detector_sampler()
            errors = 0
            shots_done = 0
            t_all0 = time.time()
            max_shots_eff = int(max_shots) if max_shots is not None else 1_000_000_000
            max_errors_eff = int(max_errors) if max_errors is not None else 1_000_000_000
            while shots_done < max_shots_eff and errors < max_errors_eff:
                t = min(resume_every, max_shots_eff - shots_done)
                dets, obs = sampler.sample(shots=t, separate_observables=True, bit_packed=False)
                batch_errors = 0
                t0 = time.time()
                for i in range(t):
                    corr = decoder.decode(dets[i, :])
                    pred = (obs_mat @ corr) % 2
                    if np.any(pred != obs[i, :]):
                        batch_errors += 1
                dt = time.time() - t0
                shots_done += t
                errors += batch_errors
                if resume_csv:
                    append_resume_csv(
                        resume_csv,
                        shots=t,
                        errors=batch_errors,
                        seconds=dt,
                        decoder="bposd",
                        json_metadata=meta,
                        custom_counts=None,
                    )

            dt = time.time() - t_all0
            results.append(ResultPoint(decoder="bposd", l=l, m=m, rounds=rounds, p=p, shots=shots_done, errors=errors, seconds=dt))
            print(f"rounds={rounds} p={p:.4g} shots={shots_done} errors={errors} LER={errors/max(1,shots_done):.3g} time={dt:.2f}s")

    return results


def plot_results(results: List[ResultPoint], *, out_png: str | None = None, show: bool = False) -> None:
    plot_points(results, out_png=out_png, show=show)


def save_csv(results: List[ResultPoint], path: str, *, meta_common: Optional[dict] = None) -> None:
    save_summary_csv(results, path, meta_common=meta_common)


def main() -> None:
    # Example: BB code from 2308.07915 (Bivariate Bicycle codes)
    a_poly = [(3, 0), (0, 1), (0, 2)]  # x^3 + y + y^2
    b_poly = [(0, 3), (1, 0), (2, 0)]  # y^3 + x + x^2
    l, m = 12, 6

    # Sweep a couple p values; user can edit here
    p_list = [1e-3, 2e-3]
    rounds_list = [6]

    results = run_BB_serial_simulation(
        a_poly=a_poly,
        b_poly=b_poly,
        l=l,
        m=m,
        p_list=p_list,
        rounds_list=rounds_list,
        max_shots=100,
        max_errors=10,
        seed=42,
        bp_iters=10,
        osd_order=0,
        resume_csv=f"Data/bb_{l}_{m}_serial_resume.csv",
        resume_every=10,
    )

    save_csv(
        results,
        f"Data/bb_{l}_{m}_serial_results.csv",
        meta_common={
            "a_poly": str(a_poly),
            "b_poly": str(b_poly),
            "code_k": int(code.K) if 'code' in locals() else None,
            "code_n": int(code.N) if 'code' in locals() else None,
        },
    )
    plot_results(results, out_png=f"Data/bb_{l}_{m}_serial_results.png", show=False)


if __name__ == "__main__":
    main()
