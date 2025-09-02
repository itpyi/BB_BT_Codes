"""BB code LER sweep and plotting script.

Computes logical error rates (LER) for a Bivariate Bicycle (BB) CSS code across
an error-rate sweep using the BP/OSD decoder, and plots the results.

Outputs
- Saves checkpoints to: Data/BB{N}_LERs_w={cycles}.npy (shape: [res, 2])
  with columns [failures, trials].
- Saves p-list to: Data/p_list_BB{N}_w={cycles}.npy
- Saves plot to: Data/BB{N}_w={cycles}_ler.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from bposd.css import css_code

try:
    # Optional; improves speed when allowed by environment
    from multiprocess import Pool
except Exception:  # pragma: no cover - fallback when multiprocess unavailable
    Pool = None

import BPOSD_threshold


def get_BB_matrix(a: List[List[int]], l: int, m: int) -> np.ndarray:
    # Aggregate Kronecker sums into an (l*m) x (l*m) matrix
    A = np.zeros((l * m, l * m), dtype=int)
    for i in range(len(a)):
        for j in range(len(a[i])):
            if i == 0:
                A += np.kron(
                    np.roll(np.identity(l), a[i][j], axis=1), np.identity(m)
                ).astype(int)
            else:
                A += np.kron(
                    np.identity(l), np.roll(np.identity(m), a[i][j], axis=1)
                ).astype(int)
    return A % 2


def get_BB_Hx_Hz(a: List[List[int]], b: List[List[int]], l: int, m: int) -> Tuple[np.ndarray, np.ndarray]:
    A = get_BB_matrix(a, l, m)
    B = get_BB_matrix(b, l, m)
    Hx = np.concatenate((A, B), axis=1)
    Hz = np.concatenate((B.T, A.T), axis=1)
    return Hx, Hz


def ler_sweep(
    code: css_code,
    p_list: np.ndarray,
    cycles: int,
    iters: np.ndarray,
    decoder_pars: Tuple[int, int] = (1000, 5),
    seed: int = 0,
    threads: int = 8,
) -> np.ndarray:
    """Return array [[failures, trials], ...] for each p in p_list.

    Uses multiprocessing when available, falls back to sequential otherwise.
    """
    res = len(p_list)
    out = np.zeros((res, 2), dtype=int)

    # Try to enable multiprocessing pool
    pool = None
    if threads > 1 and Pool is not None:
        try:
            pool = Pool(processes=threads)
        except Exception as e:
            print(f"Multiprocessing disabled: {e}")
            pool = None
            threads = 1
    else:
        threads = 1

    for r in range(res):
        print(f"Progress: {r+1} / {res}    ", end="\r")
        p = float(p_list[r])
        trials_per_worker = int(iters[r] // max(1, threads))
        args = (
            code,
            [decoder_pars[0], decoder_pars[1]],
            p / 10.0,
            p,
            p,
            trials_per_worker,
            cycles,
            seed,
        )
        if pool is not None:
            failures_list = pool.starmap(BPOSD_threshold.get_BPOSD_failures, [args] * threads)
            failures = int(np.sum(failures_list))
            trials = trials_per_worker * threads
        else:
            failures = int(
                BPOSD_threshold.get_BPOSD_failures(
                    code, [decoder_pars[0], decoder_pars[1]], p / 10.0, p, p, int(iters[r]), cycles, seed
                )
            )
            trials = int(iters[r])

        out[r] = [failures, trials]

    if pool is not None:
        pool.close()
    print()
    return out


def plot_ler(p_list: np.ndarray, counts: np.ndarray, title: str, out_path: Path) -> None:
    failures = counts[:, 0].astype(float)
    trials = counts[:, 1].astype(float)
    q = np.divide(failures, trials, out=np.zeros_like(failures), where=trials > 0)
    err = np.sqrt(np.maximum(q * (1 - q) / trials, 0.0))

    plt.figure(figsize=(6, 4))
    plt.loglog(p_list, q, marker="o", linestyle="-", label="LER")
    # Error bands (approximate; symmetric in linear space)
    plt.fill_between(p_list, np.maximum(q - err, 1e-12), q + err, alpha=0.2)
    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    # plt.show()  # enable when running interactively


def main() -> None:
    # Define a BB code (example from 2308.07915, Table 3)
    Hx, Hz = get_BB_Hx_Hz([[3], [1, 2]], [[1, 2], [3]], 12, 6)
    code = css_code(hx=Hx, hz=Hz, name="BB code")
    code.D = 12  # expected distance, optional
    code.test()

    # Sweep parameters
    res = 10
    p_min = 1e-3
    p_max = 1e-2
    p_list = np.logspace(np.log10(p_min), np.log10(p_max), res)
    cycles = 2  # set to O(d) if desired
    iters = np.logspace(5, 2, res, dtype=int)
    seed = 0
    threads = 8

    # Compute LERs
    counts = ler_sweep(code, p_list, cycles, iters, decoder_pars=(1000, 5), seed=seed, threads=threads)


    # Plot
    # Save plot under Data/
    data_dir = Path("Data")
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_ler(p_list, counts, f"BB{code.N} LER (w={cycles})", data_dir / f"BB{code.N}_w={cycles}_ler.png")


if __name__ == "__main__":
    main()
