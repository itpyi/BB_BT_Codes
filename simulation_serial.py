"""Simple serial BB code runner without sinter.

Generates Stim circuits with detectors, converts to a detector error model,
decodes shots with BPOSD in small chunks to control memory, and plots results.

Tuning parameters: rounds and physical error rate p.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, List, Tuple, Optional, Dict, Any, cast
import json
import csv
import math
import argparse
import sys

import numpy as np
import stim
from matplotlib import pyplot as plt

from simulation_common import (
    ResultPoint,
    build_bb_code,
    build_decoder_from_circuit,
    append_resume_csv,
    save_summary_csv,
    plot_points,
)
from shared_utilities import (
    safe_json_loads,
    existing_counts_for_point,
    DEFAULT_MAX_SHOTS,
    DEFAULT_MAX_ERRORS,
    DEFAULT_RESUME_EVERY,
    DEFAULT_BP_ITERS,
    DEFAULT_OSD_ORDER,
)
from quantum_circuit_builder import generate_full_circuit


# Helper removed (not used); decoding happens in early-stop loop below


# Using safe_json_loads from shared_utilities


# Using existing_counts_for_point from shared_utilities


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

            # Decode in batches with early stop criteria and resume pre-check
            sampler = circuit.compile_detector_sampler()
            errors = 0
            shots_done = 0
            t_all0 = time.time()
            max_shots_eff = int(max_shots) if max_shots is not None else DEFAULT_MAX_SHOTS
            max_errors_eff = (
                int(max_errors) if max_errors is not None else DEFAULT_MAX_ERRORS
            )
            # Check existing counts from resume; skip if already complete
            existing_shots, existing_errors = existing_counts_for_point(
                resume_csv, meta_filter=meta
            )
            if existing_shots >= max_shots_eff or existing_errors >= max_errors_eff:
                print(
                    f"[SER] rounds={rounds} p={p:.4g} already complete in resume (shots={existing_shots}, errors={existing_errors}); skipping."
                )
                results.append(
                    ResultPoint(
                        decoder="bposd",
                        l=l,
                        m=m,
                        rounds=rounds,
                        p=p,
                        shots=0,
                        errors=0,
                        seconds=0.0,
                    )
                )
                continue

            while (existing_shots + shots_done) < max_shots_eff and (
                existing_errors + errors
            ) < max_errors_eff:
                rem = max_shots_eff - (existing_shots + shots_done)
                t = min(resume_every, rem)
                dets, obs = sampler.sample(
                    shots=t, separate_observables=True, bit_packed=False
                )
                batch_errors = 0
                t_used = 0
                t0 = time.time()
                for i in range(t):
                    corr = decoder.decode(dets[i, :])
                    pred = (obs_mat @ corr) % 2
                    t_used += 1
                    if np.any(pred != obs[i, :]):
                        batch_errors += 1
                    if (existing_errors + errors + batch_errors) >= max_errors_eff:
                        break
                dt = time.time() - t0
                shots_done += t_used
                errors += batch_errors
                if resume_csv and t_used > 0:
                    append_resume_csv(
                        resume_csv,
                        shots=t_used,
                        errors=batch_errors,
                        seconds=dt,
                        decoder="bposd",
                        json_metadata=meta,
                        custom_counts=None,
                    )

            dt = time.time() - t_all0
            results.append(
                ResultPoint(
                    decoder="bposd",
                    l=l,
                    m=m,
                    rounds=rounds,
                    p=p,
                    shots=shots_done,
                    errors=errors,
                    seconds=dt,
                )
            )
            print(
                f"rounds={rounds} p={p:.4g} shots={shots_done} errors={errors} LER={errors/max(1,shots_done):.3g} time={dt:.2f}s"
            )

    return results


def plot_results(
    results: List[ResultPoint], *, out_png: str | None = None, show: bool = False
) -> None:
    plot_points(results, out_png=out_png, show=show)


def save_csv(
    results: List[ResultPoint], path: str, *, meta_common: Optional[dict] = None
) -> None:
    save_summary_csv(results, path, meta_common=meta_common)


def load_config_from_json(json_path: str) -> dict:
    """Load simulation configuration from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def run_simulation_from_config(config: dict, *, output_dir: str = "Data") -> List[ResultPoint]:
    """Run simulation using configuration dictionary."""
    # Extract parameters with defaults
    a_poly = config['a_poly']
    b_poly = config['b_poly']
    l = config['l']
    m = config['m']
    
    # Convert p_range to p_list if needed
    if 'p_range' in config:
        p_min, p_max = config['p_range']['min'], config['p_range']['max']
        num_points = config['p_range']['num_points']
        p_list = np.logspace(np.log10(p_min), np.log10(p_max), num_points)
    else:
        p_list = config['p_list']
    
    rounds_list = config['rounds_list']
    
    # Optional parameters with defaults
    max_shots = config.get('max_shots', 10_000)
    max_errors = config.get('max_errors', None)
    seed = config.get('seed', 0)
    bp_iters = config.get('bp_iters', DEFAULT_BP_ITERS)
    osd_order = config.get('osd_order', DEFAULT_OSD_ORDER)
    resume_csv = config.get('resume_csv', None)
    if resume_csv is None:
        resume_csv = f"{output_dir}/bb_{l}_{m}_serial_resume.csv"
    resume_every = config.get('resume_every', DEFAULT_RESUME_EVERY)
    
    return run_BB_serial_simulation(
        a_poly=a_poly,
        b_poly=b_poly,
        l=l,
        m=m,
        p_list=p_list,
        rounds_list=rounds_list,
        max_shots=max_shots,
        max_errors=max_errors,
        seed=seed,
        bp_iters=bp_iters,
        osd_order=osd_order,
        resume_csv=resume_csv,
        resume_every=resume_every,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BB Code Serial Simulation")
    parser.add_argument('--config', type=str, help='JSON configuration file path')
    parser.add_argument('--output-dir', type=str, default='Data', help='Output directory for results')
    args = parser.parse_args()

    if args.config:
        # Load configuration from JSON file
        print(f"Loading configuration from {args.config}")
        config = load_config_from_json(args.config)
        results = run_simulation_from_config(config, output_dir=args.output_dir)
        
        # Extract parameters for output file naming
        l, m = config['l'], config['m']
        a_poly, b_poly = config['a_poly'], config['b_poly']
    else:
        # Default configuration for backward compatibility
        print("No config file specified, using default parameters")
        a_poly = [(3, 0), (0, 1), (0, 2)]  # x^3 + y + y^2
        b_poly = [(0, 3), (1, 0), (2, 0)]  # y^3 + x + x^2
        l, m = 12, 6

        # Sweep a couple p values; user can edit here
        p_list = [5e-3, 7e-3]
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
            resume_csv=f"{args.output_dir}/bb_{l}_{m}_serial_resume.csv",
            resume_every=10,
        )

    # Build code for metadata
    code_meta = build_bb_code(a_poly, b_poly, l, m)
    
    # Aggregate from resume CSV for consolidated totals
    from results_parser_plotter import load_resume_csv

    resume_path = f"{args.output_dir}/bb_{l}_{m}_serial_resume.csv"
    aggregated = load_resume_csv([resume_path]) if os.path.exists(resume_path) else []

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    save_csv(
        aggregated if aggregated else results,
        f"{args.output_dir}/bb_{l}_{m}_serial_results.csv",
        meta_common={
            "a_poly": str(a_poly),
            "b_poly": str(b_poly),
            "code_k": int(code_meta.K),
            "code_n": int(code_meta.N),
            "l": int(l),
            "m": int(m),
        },
    )
    plot_src = aggregated if aggregated else results
    if plot_src and any(pt.errors > 0 for pt in plot_src):
        plot_points(plot_src, out_png=f"{args.output_dir}/bb_{l}_{m}_serial_results.png", show=False)
        print(f"Results saved to {args.output_dir}/bb_{l}_{m}_serial_results.csv")
        print(f"Plot saved to {args.output_dir}/bb_{l}_{m}_serial_results.png")
    else:
        print("[SER] No nonzero-error points to plot; skipping plot generation.")


if __name__ == "__main__":
    main()
