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

from circuit_utils import generate_full_circuit
from bb_common import (
    ResultPoint,
    build_bb_code,
    build_decoder_from_circuit,
    append_resume_csv,
    plot_points,
    save_summary_csv,
)


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


def _build_decoder_from_circuit(circuit: stim.Circuit, *, bp_iters: int, osd_order: int) -> Tuple[Any, np.ndarray]:
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
    decoder, obs_mat = _build_decoder_from_circuit(circuit, bp_iters=bp_iters, osd_order=osd_order)
    sampler = circuit.compile_detector_sampler()

    errors = 0
    seconds_total = 0.0
    local_shots = 0
    max_shots_eff = int(max_shots) if max_shots is not None else 1_000_000_000
    max_errors_eff = int(max_errors) if max_errors is not None else 1_000_000_000
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
        t0 = time.time()
        for i in range(t):
            corr = decoder.decode(dets[i, :])
            pred = (obs_mat @ corr) % 2
            if np.any(pred != obs[i, :]):
                batch_errors += 1
        dt = time.time() - t0

        local_shots += t
        errors += batch_errors
        seconds_total += dt

        if lock is not None:
            lock.acquire()
        try:
            if shared_errors is not None:
                shared_errors.value += batch_errors
        finally:
            if lock is not None:
                lock.release()

        if resume_csv:
            _append_resume_csv(
                resume_csv,
                shots=t,
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

            # Shared counters across workers for early stopping
            shared_shots = manager.Value('i', 0)
            shared_errors = manager.Value('i', 0)

            w = max(1, int(num_workers))
            args = [{
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
            } for _ in range(w)]

            t0 = time.time()
            with mp.Pool(processes=w) as pool:
                outs = pool.map(_worker_entry, args)
            dt = time.time() - t0

            shots_done = sum(o[0] for o in outs)
            errors = sum(o[1] for o in outs)
            # seconds_total here is wall time dt; workers' sum of seconds is CPU work
            results.append(ResultPoint(decoder="bposd", l=l, m=m, rounds=rounds, p=p, shots=shots_done, errors=errors, seconds=dt))
            print(f"[MP] rounds={rounds} p={p:.4g} shots={shots_done} errors={errors} LER={errors/max(1,shots_done):.3g} wall={dt:.2f}s")

    # Clean up manager
    manager.shutdown()
    return results


def plot_results(results: List[ResultPoint], *, out_png: str | None = None, show: bool = False) -> None:
    plot_points(results, out_png=out_png, show=show)


def save_csv(results: List[ResultPoint], path: str, *, meta_common: Optional[dict] = None) -> None:
    save_summary_csv(results, path, meta_common=meta_common)


def main() -> None:
    a_poly = [(3, 0), (0, 1), (0, 2)]
    b_poly = [(0, 3), (1, 0), (2, 0)]
    # l, m = 12, 6
    l, m = 6, 6

    # Build code once for metadata and K used in per-logical plots
    code_meta = build_bb_code(a_poly, b_poly, l, m)

    p_min = 1e-3
    p_max = 7e-3
    num_points = 1
    p_list = np.logspace(np.log10(p_min), np.log10(p_max), num_points)
    rounds_list = [8]
    # rounds_list = [8,12]

    # results = run_BB_multiprocess_simulation(
    #     a_poly=a_poly,
    #     b_poly=b_poly,
    #     l=l,
    #     m=m,
    #     p_list=p_list,
    #     rounds_list=rounds_list,
    #     max_shots=10000,
    #     max_errors=30,
    #     seed=42,
    #     bp_iters=10,
    #     osd_order=0,
    #     num_workers=8,
    #     resume_csv=f"Data/bb_{l}_{m}_mp_resume.csv",
    #     resume_every=50,
    # )
    results = run_BB_multiprocess_simulation(
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
        num_workers=1,
        resume_csv=f"Data/bb_{l}_{m}_mp_resume.csv",
        resume_every=50,
    )

    save_csv(
        results,
        f"Data/bb_{l}_{m}_mp_results.csv",
        meta_common={
            "a_poly": str(a_poly),
            "b_poly": str(b_poly),
            "code_k": int(code_meta.K),
            "code_n": int(code_meta.N),
            "l": int(l),
            "m": int(m),
        },
    )
    # Generate three plots: LER, per-logical (using K), and per-round
    plot_points(results, out_png=f"Data/bb_{l}_{m}_mp_results.png", show=False, y_mode="ler")
    plot_points(results, out_png=f"Data/bb_{l}_{m}_mp_results_per_logical.png", show=False, y_mode="per_logical", K=int(code_meta.K))
    plot_points(results, out_png=f"Data/bb_{l}_{m}_mp_results_per_round.png", show=False, y_mode="per_round")


if __name__ == "__main__":
    main()
