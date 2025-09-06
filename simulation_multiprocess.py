"""Multiprocess BB code runner without sinter.

Spawns worker processes per (p, rounds) point to sample and decode shots in
small batches, periodically appending to a resume CSV. Produces summary results
compatible with the serial runner.
"""

from __future__ import annotations

import os
import time
import json
from typing import Iterable, List, Tuple, Optional, Any

import numpy as np
import stim
from matplotlib import pyplot as plt

import multiprocessing as mp

from quantum_circuit_builder import generate_full_circuit
from simulation_common import (
    ResultPoint,
    build_bb_code,
    build_decoder_from_circuit,
    append_resume_csv,
    plot_points,
    save_summary_csv,
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
import csv
import math
import argparse
import sys


# Using safe_json_loads from shared_utilities


# Using existing_counts_for_point from shared_utilities


def _append_resume_csv(
    path: str,
    *,
    shots: int,
    errors: int,
    seconds: float,
    decoder: str,
    json_metadata: dict,
    custom_counts: Optional[dict] = None,
    lock: Optional[Any] = None,
) -> None:
    append_resume_csv(
        path,
        shots=shots,
        errors=errors,
        seconds=seconds,
        decoder=decoder,
        json_metadata=json_metadata,
        custom_counts=custom_counts,
        lock=lock,
    )


def _build_decoder_from_circuit(
    circuit: stim.Circuit, *, bp_iters: int, osd_order: int
) -> Tuple[Any, np.ndarray]:
    return build_decoder_from_circuit(circuit, bp_iters=bp_iters, osd_order=osd_order)


def _worker_run(
    *,
    circuit_text: str,
    max_shots: Optional[int],
    max_errors: Optional[int],
    resume_csv: Optional[str],
    resume_every: int,
    meta: dict,
    bp_iters: int,
    osd_order: int,
    lock: Optional[Any],
    shared_shots: Optional[Any],
    shared_errors: Optional[Any],
) -> Tuple[int, int, float]:
    circuit = stim.Circuit(circuit_text)
    decoder, obs_mat = _build_decoder_from_circuit(
        circuit, bp_iters=bp_iters, osd_order=osd_order
    )
    sampler = circuit.compile_detector_sampler()

    errors = 0
    seconds_total = 0.0
    local_shots = 0
    max_shots_eff = int(max_shots) if max_shots is not None else DEFAULT_MAX_SHOTS
    max_errors_eff = int(max_errors) if max_errors is not None else DEFAULT_MAX_ERRORS
    while True:
        # Reserve work under lock to avoid exceeding global caps.
        if lock is not None:
            lock.acquire()
        try:
            cur_shots = shared_shots.value if shared_shots is not None else 0
            cur_errors = shared_errors.value if shared_errors is not None else 0
            if cur_shots >= max_shots_eff or cur_errors >= max_errors_eff:
                break
            t = min(resume_every, max_shots_eff - cur_shots)
            if shared_shots is not None:
                shared_shots.value = cur_shots + t
        finally:
            if lock is not None:
                lock.release()

        dets, obs = sampler.sample(shots=t, separate_observables=True, bit_packed=False)
        batch_errors = 0
        t_used = 0
        t0 = time.time()
        for i in range(t):
            corr = decoder.decode(dets[i, :])
            pred = (obs_mat @ corr) % 2
            t_used += 1
            if np.any(pred != obs[i, :]):
                batch_errors += 1
            # Early stop if hitting error cap within the batch
            if max_errors is not None and shared_errors is not None:
                if lock is not None:
                    lock.acquire()
                try:
                    cur_errors = shared_errors.value
                finally:
                    if lock is not None:
                        lock.release()
                if cur_errors + batch_errors >= max_errors_eff:
                    break
        dt = time.time() - t0

        local_shots += t_used
        errors += batch_errors
        seconds_total += dt

        if lock is not None:
            lock.acquire()
        try:
            if shared_errors is not None:
                shared_errors.value += batch_errors
            # Refund unused reserved shots if we stopped early
            if shared_shots is not None and t_used < t:
                shared_shots.value -= t - t_used
        finally:
            if lock is not None:
                lock.release()

        if resume_csv:
            _append_resume_csv(
                resume_csv,
                shots=t_used,
                errors=batch_errors,
                seconds=dt,
                decoder="bposd",
                json_metadata=meta,
                custom_counts=None,
                lock=lock,
            )

    return local_shots, errors, seconds_total


def _worker_entry(kw: dict) -> Tuple[int, int, float]:
    return _worker_run(**kw)


def run_BB_multiprocess_simulation(
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
    bp_iters: int = 50,
    osd_order: int = 0,
    num_workers: int = 2,
    resume_csv: Optional[str] = None,
    resume_every: int = 50,
) -> List[ResultPoint]:
    # Set a stable start method
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
    except RuntimeError:
        pass

    code = build_bb_code(a_poly, b_poly, l, m)
    rng = np.random.default_rng(seed)
    results = []

    # Shared file lock for resume CSV writes (manager-proxied for spawn)
    # Default resume path with bb_l_m in the name
    if resume_csv is None:
        resume_csv = f"Data/bb_{l}_{m}_mp_resume.csv"
    manager = mp.Manager()
    lock = manager.Lock()

    for rounds in rounds_list:
        for p in p_list:
            # Build circuit once and pass its text to workers
            circuit = generate_full_circuit(
                code=code,
                rounds=rounds,
                p1=p / 10.0,
                p2=p,
                p_spam=p,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
            circuit_text = str(circuit)

            # Metadata for resume CSV rows
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

            # Check resume CSV for existing totals for this point
            existing_shots, existing_errors = existing_counts_for_point(
                resume_csv, meta_filter=meta
            )

            # Determine effective caps
            max_shots_eff = int(max_shots) if max_shots is not None else DEFAULT_MAX_SHOTS
            max_errors_eff = (
                int(max_errors) if max_errors is not None else DEFAULT_MAX_ERRORS
            )

            # If already at/over caps from resume data, skip running workers
            if existing_shots >= max_shots_eff or existing_errors >= max_errors_eff:
                print(
                    f"[MP] rounds={rounds} p={p:.4g} already complete in resume (shots={existing_shots}, errors={existing_errors}); skipping."
                )
                # Append a zero-work result for traceability of walltime 0
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

            # Shared counters across workers for early stopping, seeded from resume
            shared_shots = manager.Value("i", int(existing_shots))
            shared_errors = manager.Value("i", int(existing_errors))

            w = max(1, int(num_workers))
            args = [
                {
                    "circuit_text": circuit_text,
                    "max_shots": max_shots,
                    "max_errors": max_errors,
                    "resume_csv": resume_csv,
                    "resume_every": resume_every,
                    "meta": meta,
                    "bp_iters": bp_iters,
                    "osd_order": osd_order,
                    "lock": lock,
                    "shared_shots": shared_shots,
                    "shared_errors": shared_errors,
                }
                for _ in range(w)
            ]

            t0 = time.time()
            with mp.Pool(processes=w) as pool:
                outs = pool.map(_worker_entry, args)
            dt = time.time() - t0

            shots_done = sum(o[0] for o in outs)
            errors = sum(o[1] for o in outs)
            # seconds_total here is wall time dt; workers' sum of seconds is CPU work
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
            total_shots = existing_shots + shots_done
            total_errors = existing_errors + errors
            print(
                f"[MP] rounds={rounds} p={p:.4g} shots+={shots_done} (total={total_shots}) errors+={errors} (total={total_errors}) LER_batch={errors/max(1,shots_done):.3g} wall={dt:.2f}s"
            )

    # Clean up manager
    manager.shutdown()
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
    num_workers = config.get('num_workers', 2)
    resume_csv = config.get('resume_csv', None)
    if resume_csv is None:
        resume_csv = f"{output_dir}/bb_{l}_{m}_mp_resume.csv"
    resume_every = config.get('resume_every', DEFAULT_RESUME_EVERY)
    
    return run_BB_multiprocess_simulation(
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
        num_workers=num_workers,
        resume_csv=resume_csv,
        resume_every=resume_every,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BB Code Multiprocess Simulation")
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
        a_poly = [(3, 0), (0, 1), (0, 2)]
        b_poly = [(0, 3), (1, 0), (2, 0)]
        l, m = 6, 6

        p_min = 1e-3
        p_max = 7e-3
        num_points = 2
        p_list = np.logspace(np.log10(p_min), np.log10(p_max), num_points)
        rounds_list = [8, 12]

        results = run_BB_multiprocess_simulation(
            a_poly=a_poly,
            b_poly=b_poly,
            l=l,
            m=m,
            p_list=p_list,
            rounds_list=rounds_list,
            max_shots=1000,
            max_errors=4,
            seed=42,
            bp_iters=10,
            osd_order=0,
            num_workers=1,
            resume_csv=f"{args.output_dir}/bb_{l}_{m}_mp_resume.csv",
            resume_every=50,
        )

    # Build code for metadata
    code_meta = build_bb_code(a_poly, b_poly, l, m)
    # Aggregate from resume CSV for full totals (including previous runs)
    from results_parser_plotter import load_resume_csv

    resume_path = f"{args.output_dir}/bb_{l}_{m}_mp_resume.csv"
    aggregated = load_resume_csv([resume_path]) if os.path.exists(resume_path) else []

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    save_csv(
        aggregated if aggregated else results,
        f"{args.output_dir}/bb_{l}_{m}_mp_results.csv",
        meta_common={
            "a_poly": str(a_poly),
            "b_poly": str(b_poly),
            "code_k": int(code_meta.K),
            "code_n": int(code_meta.N),
            "l": int(l),
            "m": int(m),
        },
    )
    # Generate plots from aggregated points when available; otherwise from this run's results
    plot_src = aggregated if aggregated else results
    if plot_src:
        plot_points(
            plot_src, out_png=f"{args.output_dir}/bb_{l}_{m}_mp_results.png", show=False, y_mode="ler"
        )
        plot_points(
            plot_src,
            out_png=f"{args.output_dir}/bb_{l}_{m}_mp_results_per_logical.png",
            show=False,
            y_mode="per_logical",
            K=int(code_meta.K),
        )
        plot_points(
            plot_src,
            out_png=f"{args.output_dir}/bb_{l}_{m}_mp_results_per_round.png",
            show=False,
            y_mode="per_round",
        )
        print(f"Results saved to {args.output_dir}/bb_{l}_{m}_mp_results.csv")
        print(f"Plots saved to {args.output_dir}/bb_{l}_{m}_mp_results*.png")
    else:
        print("No results to plot")


if __name__ == "__main__":
    main()
