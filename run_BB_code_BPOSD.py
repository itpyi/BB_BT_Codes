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
from typing import List, Tuple, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from bposd.css import css_code

try:
    # Optional; improves speed when allowed by environment
    from multiprocess import Pool
except Exception:  # pragma: no cover - fallback when multiprocess unavailable
    Pool = None

import BPOSD_threshold


def _shift_mat(n: int, k: int) -> np.ndarray:
    """Return n x n cyclic shift-by-k matrix over GF(2) as uint8."""
    return np.roll(np.identity(n, dtype=np.uint8), int(k) % n, axis=1)


def _parse_bivariate_terms(
    spec: Union[Sequence[Tuple[int, int]], np.ndarray]
) -> List[Tuple[int, int]]:
    """Normalize polynomial spec into list of (i, j) exponents for x^i y^j.

    Accepts:
    - list of pairs: [(i, j), ...] e.g., [(2,0), (1,1), (0,2)] for x^2 + xy + y^2
    - ndarray shape (k, 2) with integer exponents
    """
    # ndarray of pairs
    if isinstance(spec, np.ndarray):
        arr = np.asarray(spec, dtype=np.uint8)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [(int(i), int(j)) for i, j in arr]
        raise ValueError("ndarray spec must have shape (k, 2)")

    # list of pairs (i, j)
    if len(spec) > 0 and isinstance(spec[0], (list, tuple)) and len(spec[0]) == 2:
        return [(int(p[0]), int(p[1])) for p in spec]  

    raise ValueError("Unsupported polynomial spec format for bivariate terms")


def get_BB_matrix(a: Union[Sequence[Tuple[int, int]], np.ndarray], l: int, m: int) -> np.ndarray:
    """Return the (l*m) x (l*m) binary matrix for polynomial a(x, y).

    The polynomial is specified by exponent pairs for monomials x^i y^j. For
    each (i, j), contribute kron(shift_x(i), shift_y(j)).
    """
    terms = _parse_bivariate_terms(a)
    A = np.zeros((l * m, l * m), dtype=np.uint8)
    for (ix, iy) in terms:
        term = np.kron(_shift_mat(l, ix), _shift_mat(m, iy)).astype(np.uint8)
        # XOR accumulates modulo-2 for binary matrices in {0,1}
        A += term
    return A % 2



def get_BB_Hx_Hz(
    a: Union[Sequence[Tuple[int, int]], np.ndarray],
    b: Union[Sequence[Tuple[int, int]], np.ndarray],
    l: int,
    m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build BB code Hx, Hz from bivariate polynomials a(x,y), b(x,y).

    Accepts list of exponent pairs [(i, j), ...] or an ndarray of shape (k, 2).
    """
    A = get_BB_matrix(a, l, m)
    B = get_BB_matrix(b, l, m)
    Hx = np.concatenate((A, B), axis=1).astype(np.uint8)
    Hz = np.concatenate((B.T, A.T), axis=1).astype(np.uint8)
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
    # Example: general bivariate with an xy term
    a = [(3, 0), (0, 1), (0, 2)]  # x^3 + y + y^2
    # b = [(2, 0), (1, 1), (0, 2)]  # x^2 + xy + y^2
    b = [(0, 3), (1, 0), (2, 0)]  # y^3 + x + x^2
    Hx, Hz = get_BB_Hx_Hz(a, b, 12, 6)
    code = css_code(hx=Hx, hz=Hz, name="BB code")
    code.D = 12  # expected distance, optional
    code.test()

    # Sweep parameters
    res = 10
    p_min = 1e-3
    p_max = 1e-2
    p_list = np.logspace(np.log10(p_min), np.log10(p_max), res)
    cycles = 10  # set to O(d) if desired
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
    # main()

    
    from ldpc.codes import rep_code, hamming_code
    from bposd.hgp import hgp
    from circuit_utils import generate_full_circuit
    d = 3
    res = 10
    surface_code = hgp(h1=rep_code(d),h2=rep_code(d))
    cycles = d
    circuit_seed = 0
    rounds = 5
    p1 = 0.005
    p_spam = 0.005
    p2 = 0.01
    seed = 42

    circuits = generate_full_circuit(
    surface_code,
    rounds,
    p1,
    p2,
    p_spam,
    seed)


    print(circuits)

    