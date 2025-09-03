"""MWPM-based threshold estimation utilities.

Builds a spacetime parity-check graph across rounds and decodes with MWPM.
"""

import numpy as np
from scipy.sparse import csr_matrix
from pymatching import Matching
import circuit_utils
from typing import Protocol, Any


class CSSCode(Protocol):
    hx: Any
    hz: Any
    lz: Any


def get_MWPM_failures(
    code: CSSCode,
    p1: float,
    p2: float,
    p_spam: float,
    iters: int,
    rounds: int,
    seed: int,
) -> int:
    """Return number of logical failures using MWPM over ``iters`` shots.

    Parameters
    - code: CSS code with ``hz`` and ``lz``.
    - p1, p2, p_spam: noise parameters for 1q/2q depolarizing and SPAM.
    - iters: number of Monte Carlo shots to sample.
    - rounds: number of repeated syndrome measurement rounds.
    - seed: RNG seed for circuit generation.
    """
    H = code.hz.toarray()
    m, n = H.shape
    mx = code.hx.shape[0]

    # Construct spacetime decoding graph as in BPOSD_threshold.
    H_dec = np.kron(np.eye(rounds + 1, dtype=int), H)
    H_dec = np.concatenate(
        (H_dec, np.zeros([m * (rounds + 1), m * rounds], dtype=int)), axis=1
    )
    for j in range(m * rounds):
        H_dec[j, n * (rounds + 1) + j] = 1
        H_dec[m + j, n * (rounds + 1) + j] = 1
    H_dec = csr_matrix(H_dec)

    # Log-odds weights derived from p2.
    matching = Matching(H_dec, weights=np.log((1 - p2) / p2))

    c = circuit_utils.generate_full_circuit(
        code, rounds=rounds, p1=p1, p2=p2, p_spam=p_spam, seed=seed
    )
    sampler = c.compile_sampler()

    failures = 0
    outer_reps = iters // 256  # Stim samples a minimum of 256 shots at a time
    remainder = iters % 256
    for j in range(outer_reps + 1):
        num_shots = 256 if j < outer_reps else remainder
        if num_shots == 0:
            continue
        output = sampler.sample(shots=num_shots)
        k = m + mx
        for i in range(num_shots):
            rec = output[i]
            z_rounds = np.zeros((rounds, m), dtype=int)
            for r in range(rounds):
                base = r * k
                z_rounds[r, :] = rec[base + mx: base + mx + m]
            data_meas = rec[-n:]

            syndromes = np.zeros([rounds + 1, m], dtype=int)
            syndromes[:rounds] = z_rounds
            syndromes[-1] = H @ data_meas % 2
            # Difference syndrome connects time-slices in the graph.
            syndromes[1:] = syndromes[1:] ^ syndromes[:-1]

            mwpm_output = np.reshape(
                matching.decode(np.ravel(syndromes))[: n * (rounds + 1)],
                [rounds + 1, n],
            )
            correction = mwpm_output.sum(axis=0) % 2
            final_state = data_meas ^ correction
            if (code.lz @ final_state % 2).any():
                failures += 1

    return failures
