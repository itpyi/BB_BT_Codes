"""Generic QEC code simulation runner.

Supports BB (Bivariate Bicycle), BT (Bivariate Tricycle), and TT (Trivariate Tricycle) codes
through a unified API. Works in both serial and multiprocess modes.
"""

from __future__ import annotations

import os
import time
import json
import argparse
import logging
import faulthandler
import signal
from typing import Iterable, List, Optional, Dict, Any, Tuple, Union

import numpy as np
import stim
import multiprocessing as mp

from quantum_circuit_builder import generate_full_circuit
from qec_simulation_core import (
    ResultPoint,
    build_code_generic,
    build_decoder_from_circuit,
    append_resume_csv,
    plot_points,
    save_summary_csv,
    generate_default_resume_csv,
    extract_code_params_from_config,
)
from file_io_utils import (
    safe_json_loads,
    existing_counts_for_point,
    DEFAULT_MAX_SHOTS,
    DEFAULT_MAX_ERRORS,
    DEFAULT_RESUME_EVERY,
    DEFAULT_BP_ITERS,
    DEFAULT_OSD_ORDER,
)


def _enable_watchdog_from_env() -> None:
    """Enable lightweight watchdogs if requested via env vars.

    - Always register SIGUSR1 to dump stack traces on demand.
    - If QEC_WATCHDOG_SECS is set to an int > 0, auto-dump all
      thread tracebacks every that many seconds (non-fatal).
    """
    # Register on-demand stack dump (kill -USR1 <pid>)
    try:
        faulthandler.register(signal.SIGUSR1)
    except Exception:
        pass

    # Optional periodic auto-dump for long hangs
    wd = os.getenv("QEC_WATCHDOG_SECS")
    if wd:
        try:
            sec = int(wd)
            if sec > 0:
                faulthandler.dump_traceback_later(sec, repeat=True)
                logging.getLogger(__name__).info(
                    f"Watchdog armed: auto-dump tracebacks every {sec}s"
                )
        except ValueError:
            pass


def run_QEC_serial_simulation(
    *,
    code_type: str,
    code_params: dict,
    p_list: Iterable[float],
    rounds_list: Iterable[int],
    max_shots: Optional[int] = 10_000,
    max_errors: Optional[int] = None,
    seed: int = 0,
    bp_iters: int = 20,
    osd_order: int = 3,
    resume_csv: Optional[str] = None,
    resume_every: int = 200,
    decompose_dem: Optional[bool] = None,
) -> List[ResultPoint]:
    """Run generic QEC simulation in serial mode.
    
    Args:
        code_type: "BB", "BT", or "TT"
        code_params: Dictionary with code-specific parameters
        p_list: Physical error rates to test
        rounds_list: Syndrome extraction rounds to test
        max_shots: Maximum shots per parameter point
        max_errors: Maximum errors before stopping
        seed: Random seed
        bp_iters: BP decoder iterations
        osd_order: OSD decoder order
        resume_csv: Resume CSV file path (optional)
        resume_every: Save resume data every N shots
        
    Returns:
        List of result points
    """
    # Build code using generic dispatcher
    code = build_code_generic(code_type, **code_params)
    rng = np.random.default_rng(seed)
    results: List[ResultPoint] = []
    
    # Generate default resume CSV name if not provided
    if resume_csv is None:
        resume_csv = generate_default_resume_csv(code_type, "Data", "serial", **code_params)

    for rounds in rounds_list:
        for p in p_list:
            # Generate circuit (same API for all code types)
            t0 = time.time()
            circuit = generate_full_circuit(
                code=code,
                rounds=rounds,
                p1=p / 10.0,
                p2=p,
                p_spam=p,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
            logging.info(
                f"[SER] built circuit in {time.time() - t0:.2f}s (rounds={rounds}, p={p:.4g})"
            )

            t1 = time.time()
            decoder, obs_mat = build_decoder_from_circuit(
                circuit, bp_iters=bp_iters, osd_order=osd_order, decompose_dem=decompose_dem
            )
            logging.info(
                f"[SER] built decoder in {time.time() - t1:.2f}s (rounds={rounds}, p={p:.4g})"
            )

            # Prepare metadata for resume CSV
            meta = {
                "code_type": code_type,
                "code_k": code.K,
                "code_n": code.N,
                "p": float(p),
                "rounds": int(rounds),
                **{f"code_{k}": v for k, v in code_params.items()},  # Add all code params
            }

            # Decode in batches with early stop criteria and resume pre-check
            sampler = circuit.compile_detector_sampler()
            errors = 0
            shots_done = 0
            t_all0 = time.time()
            max_shots_eff = int(max_shots) if max_shots is not None else DEFAULT_MAX_SHOTS
            max_errors_eff = int(max_errors) if max_errors is not None else DEFAULT_MAX_ERRORS
            
            # Check existing counts from resume; skip if already complete
            existing_shots, existing_errors = existing_counts_for_point(
                resume_csv, meta_filter=meta
            )
            if existing_shots >= max_shots_eff or existing_errors >= max_errors_eff:
                print(
                    f"[SER] {code_type} rounds={rounds} p={p:.4g} already complete in resume (shots={existing_shots}, errors={existing_errors}); skipping."
                )
                results.append(
                    ResultPoint(
                        decoder="bposd",
                        l=code_params["l"],
                        m=code_params["m"], 
                        rounds=rounds,
                        p=p,
                        shots=0,
                        errors=0,
                        seconds=0.0,
                        code_type=code_type,
                        n=int(code_params.get("n", -1)),
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
                    l=code_params["l"],
                    m=code_params["m"],
                    rounds=rounds,
                    p=p,
                    shots=shots_done,
                    errors=errors,
                    seconds=dt,
                    code_type=code_type,
                    n=int(code_params.get("n", -1)),
                )
            )
            print(
                f"{code_type} rounds={rounds} p={p:.4g} shots={shots_done} errors={errors} LER={errors/max(1,shots_done):.3g} time={dt:.2f}s"
            )

    return results


def _worker_run_generic(
    *,
    circuit_text: str,
    max_shots: Optional[int],
    max_errors: Optional[int],
    resume_csv: Optional[str],
    resume_every: int,
    meta: dict,
    bp_iters: int,
    osd_order: int,
    decompose_dem: Optional[bool],
    lock: Optional[Any],
    shared_shots: Optional[Any],
    shared_errors: Optional[Any],
) -> Tuple[int, int, float]:
    """Generic worker function for multiprocess simulation."""
    circuit = stim.Circuit(circuit_text)
    t0 = time.time()
    decoder, obs_mat = build_decoder_from_circuit(
        circuit, bp_iters=bp_iters, osd_order=osd_order, decompose_dem=decompose_dem
    )
    logging.info(f"[WRK] decoder built in {time.time() - t0:.2f}s")
    sampler = circuit.compile_detector_sampler()

    errors = 0
    seconds_total = 0.0
    local_shots = 0
    max_shots_eff = int(max_shots) if max_shots is not None else DEFAULT_MAX_SHOTS
    max_errors_eff = int(max_errors) if max_errors is not None else DEFAULT_MAX_ERRORS
    
    last_hb = time.time()
    hb_env = os.getenv("QEC_HEARTBEAT_SECS")
    hb_sec = int(hb_env) if (hb_env and hb_env.isdigit()) else 0
    while True:
        # Reserve work under lock to avoid exceeding global caps
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

        # Optional worker heartbeat
        now = time.time()
        if hb_sec > 0 and (now - last_hb) >= hb_sec:
            last_hb = now
            cur_shots = shared_shots.value if shared_shots is not None else local_shots
            cur_errors = shared_errors.value if shared_errors is not None else errors
            logging.info(f"[WRK] heartbeat: shots={cur_shots}, errors={cur_errors}")

        if resume_csv:
            append_resume_csv(
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


def _worker_entry_generic(kw: dict) -> Tuple[int, int, float]:
    return _worker_run_generic(**kw)


def run_QEC_multiprocess_simulation(
    *,
    code_type: str,
    code_params: dict,
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
    decompose_dem: Optional[bool] = None,
) -> List[ResultPoint]:
    """Run generic QEC simulation in multiprocess mode.
    
    Args:
        code_type: "BB", "BT", or "TT"
        code_params: Dictionary with code-specific parameters
        (other args same as serial version)
        
    Returns:
        List of result points
    """
    # Set a stable start method
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
    except RuntimeError:
        pass

    # Build code using generic dispatcher
    code = build_code_generic(code_type, **code_params)
    rng = np.random.default_rng(seed)
    results = []

    # Generate default resume CSV name if not provided
    if resume_csv is None:
        resume_csv = generate_default_resume_csv(code_type, "Data", "mp", **code_params)
        
    # Shared file lock for resume CSV writes
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
                "code_type": code_type,
                "code_k": code.K,
                "code_n": code.N,
                "p": float(p),
                "rounds": int(rounds),
                **{f"code_{k}": v for k, v in code_params.items()},  # Add all code params
            }

            # Check resume CSV for existing totals for this point
            existing_shots, existing_errors = existing_counts_for_point(
                resume_csv, meta_filter=meta
            )

            # Determine effective caps
            max_shots_eff = int(max_shots) if max_shots is not None else DEFAULT_MAX_SHOTS
            max_errors_eff = int(max_errors) if max_errors is not None else DEFAULT_MAX_ERRORS

            # If already at/over caps from resume data, skip running workers
            if existing_shots >= max_shots_eff or existing_errors >= max_errors_eff:
                print(
                    f"[MP] {code_type} rounds={rounds} p={p:.4g} already complete in resume (shots={existing_shots}, errors={existing_errors}); skipping."
                )
                # Append a zero-work result for traceability of walltime 0
                results.append(
                    ResultPoint(
                        decoder="bposd",
                        l=code_params["l"],
                        m=code_params["m"],
                        rounds=rounds,
                        p=p,
                        shots=0,
                        errors=0,
                        seconds=0.0,
                        code_type=code_type,
                        n=int(code_params.get("n", -1)),
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
                    "decompose_dem": decompose_dem,
                    "lock": lock,
                    "shared_shots": shared_shots,
                    "shared_errors": shared_errors,
                }
                for _ in range(w)
            ]

            print("Start bp osd decoding")
            t0 = time.time()
            with mp.Pool(processes=w) as pool:
                outs = pool.map(_worker_entry_generic, args)
            dt = time.time() - t0

            shots_done = sum(o[0] for o in outs)
            errors = sum(o[1] for o in outs)
            
            results.append(
                ResultPoint(
                    decoder="bposd",
                    l=code_params["l"],
                    m=code_params["m"],
                    rounds=rounds,
                    p=p,
                    shots=shots_done,
                    errors=errors,
                    seconds=dt,
                    code_type=code_type,
                    n=int(code_params.get("n", -1)),
                )
            )
            total_shots = existing_shots + shots_done
            total_errors = existing_errors + errors
            print(
                f"[MP] {code_type} rounds={rounds} p={p:.4g} shots+={shots_done} (total={total_shots}) errors+={errors} (total={total_errors}) LER_batch={errors/max(1,shots_done):.3g} wall={dt:.2f}s"
            )

    # Clean up manager
    manager.shutdown()
    return results


def load_config_from_json(json_path: str) -> Any:
    """Load simulation configuration from JSON file.

    Supports:
    - Single experiment (dict) matching the existing schema.
    - Multi-experiment: either a list of dicts, or a dict with key
      "experiments": list[dict]. No implicit parameter merging is performed.
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def _expand_experiments(config: Any) -> List[dict]:
    """Normalize config into a list of experiment dicts.

    Accepted forms:
    - Single dict: returns [dict]
    - List[dict]: returns as-is (filters to dict items)
    - Dict with {"experiments": [dict]}: returns that list (no merging)

    No cross-experiment defaults or parameter merging is applied.
    Each experiment must be self-contained.
    """
    if isinstance(config, list):
        return [c for c in config if isinstance(c, dict)]
    if isinstance(config, dict) and isinstance(config.get("experiments"), list):
        return [c for c in config["experiments"] if isinstance(c, dict)]
    if isinstance(config, dict):
        # Treat as single experiment
        return [config]
    raise ValueError("Unsupported config format. Expected dict, list[dict], or {experiments:[...]}.")


def run_simulation_from_config(config: Union[dict, List[dict]], *, output_dir: str = "Data", multiprocess: bool = False, decompose_dem: Optional[bool] = None) -> List[ResultPoint]:
    """Run simulation using configuration dictionary or multi-experiment config.

    - If given a single config dict, runs one experiment (original behavior).
    - If given a list of dicts, or a dict containing key "experiments", runs each.
    """
    # Expand to experiments list if needed
    exps = _expand_experiments(config)
    all_results: List[ResultPoint] = []
    for exp in exps:
        # Extract code type and parameters
        code_type, code_params = extract_code_params_from_config(exp)

        # Convert p_range to p_list if needed
        if 'p_range' in exp:
            p_min, p_max = exp['p_range']['min'], exp['p_range']['max']
            num_points = exp['p_range']['num_points']
            p_list = np.logspace(np.log10(p_min), np.log10(p_max), num_points)
        else:
            p_list = exp['p_list']

        rounds_list = exp['rounds_list']

        # Optional parameters with defaults
        max_shots = exp.get('max_shots', 10_000)
        max_errors = exp.get('max_errors', None)
        seed = exp.get('seed', 0)
        bp_iters = exp.get('bp_iters', DEFAULT_BP_ITERS)
        osd_order = exp.get('osd_order', DEFAULT_OSD_ORDER)
        resume_csv = exp.get('resume_csv', None)
        if resume_csv is None:
            runner_type = "mp" if multiprocess else "serial"
            resume_csv = generate_default_resume_csv(code_type, output_dir, runner_type, **code_params)
        resume_every = exp.get('resume_every', DEFAULT_RESUME_EVERY)

        # Choose simulation runner
        # Allow config to specify dem_decompose; CLI arg overrides if provided
        config_decompose = exp.get('dem_decompose', None)
        decompose_eff = decompose_dem if decompose_dem is not None else config_decompose
        if multiprocess:
            num_workers = exp.get('num_workers', 2)
            res = run_QEC_multiprocess_simulation(
                code_type=code_type,
                code_params=code_params,
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
                decompose_dem=decompose_eff,
            )
        else:
            res = run_QEC_serial_simulation(
                code_type=code_type,
                code_params=code_params,
                p_list=p_list,
                rounds_list=rounds_list,
                max_shots=max_shots,
                max_errors=max_errors,
                seed=seed,
                bp_iters=bp_iters,
                osd_order=osd_order,
                resume_csv=resume_csv,
                resume_every=resume_every,
                decompose_dem=decompose_eff,
            )
        all_results.extend(res)
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic QEC Code Simulation")
    parser.add_argument('--config', type=str, required=True, help='JSON configuration file path')
    parser.add_argument('--output-dir', type=str, default='Data', help='Output directory for results')
    parser.add_argument('--multiprocess', action='store_true', help='Use multiprocess simulation')
    parser.add_argument('--dem-decompose', dest='dem_decompose', action='store_true', help='Decompose DEM for faster decoding')
    parser.add_argument('--no-dem-decompose', dest='dem_decompose', action='store_false', help='Disable DEM decomposition')
    parser.set_defaults(dem_decompose=None)
    args = parser.parse_args()

    # Basic logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _enable_watchdog_from_env()

    # Load configuration from JSON file
    print(f"Loading configuration from {args.config}")
    cfg_raw = load_config_from_json(args.config)

    # Expand into experiments
    experiments = _expand_experiments(cfg_raw)

    # Run each experiment and emit outputs independently
    from results_parser_plotter import load_resume_csv
    runner_type = "mp" if args.multiprocess else "serial"
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, exp in enumerate(experiments, start=1):
        # Run
        results = run_simulation_from_config(
            exp,
            output_dir=args.output_dir,
            multiprocess=args.multiprocess,
            decompose_dem=args.dem_decompose,
        )

        # Build code for metadata & filenames
        code_type, code_params = extract_code_params_from_config(exp)
        code_meta = build_code_generic(code_type, **code_params)

        # Aggregate from resume CSV for full totals (including previous runs)
        resume_path = generate_default_resume_csv(code_type, args.output_dir, runner_type, **code_params)
        aggregated = load_resume_csv([resume_path]) if os.path.exists(resume_path) else []

        # Output base
        code_prefix = code_type.lower()
        if code_type == "TT":
            size_suffix = f"{code_params['l']}_{code_params['m']}_{code_params['n']}"
        else:
            size_suffix = f"{code_params['l']}_{code_params['m']}"
        output_base = f"{args.output_dir}/{code_prefix}_{size_suffix}_{runner_type}"

        # Save CSV results
        save_summary_csv(
            aggregated if aggregated else results,
            f"{output_base}_results.csv",
            meta_common={
                "code_type": code_type,
                "code_k": int(code_meta.K),
                "code_n": int(code_meta.N),
                **{f"code_{k}": v for k, v in code_params.items()},
            },
        )

        # Plots
        plot_src = aggregated if aggregated else results
        if plot_src:
            plot_points(
                plot_src, out_png=f"{output_base}_results.png", show=False, y_mode="ler"
            )
            plot_points(
                plot_src,
                out_png=f"{output_base}_results_per_logical.png",
                show=False,
                y_mode="per_logical",
                K=int(code_meta.K),
            )
            plot_points(
                plot_src,
                out_png=f"{output_base}_results_per_round.png",
                show=False,
                y_mode="per_round",
            )
            print(f"[{idx}/{len(experiments)}] Results saved to {output_base}_results.csv")
            print(f"[{idx}/{len(experiments)}] Plots saved to {output_base}_results*.png")
        else:
            print(f"[{idx}/{len(experiments)}] No results to plot")


if __name__ == "__main__":
    main()
