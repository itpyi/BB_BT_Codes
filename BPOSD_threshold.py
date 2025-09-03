"""BP/OSD-based threshold estimation utilities.

This module builds a spacetime parity-check matrix for repeated Z-syndrome
extraction rounds, decodes with BP+OSD, and counts logical failures.
"""

import numpy as np
from scipy.sparse import csr_matrix
from ldpc import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
import circuit_utils
from typing import Sequence, Protocol, Any


class CSSCode(Protocol):
    hx: Any
    hz: Any
    lz: Any
    lx: Any
    N: int


def get_BPOSD_failures(
    code: CSSCode,
    par: Sequence[int],
    p1: float,
    p2: float,
    p_spam: float,
    iters: int,
    rounds: int,
    seed: int,
) -> int:
    """Return number of logical failures over ``iters`` shots.

    Parameters
    - code: CSS code object with attributes ``hx``, ``hz``, ``lz`` and ``N``.
    - par: list-like ``[bp_iters, osd_order]`` for the BP/OSD decoder.
    - p1: single-qubit depolarizing rate used in circuit noise.
    - p2: two-qubit depolarizing rate used in circuit noise and decoder weight.
    - p_spam: SPAM error rate (prep/measure X-errors).
    - iters: number of Monte Carlo shots to sample.
    - rounds: number of repeated syndrome measurement rounds.
    - seed: RNG seed forwarded to circuit generation.

    Notes
    - Stim samples in blocks of 256; sampling is batched accordingly.
    - The spacetime parity-check ties successive rounds via difference syndromes.
    """
    # Parity-check for Z errors; shape (m, n)
    H = code.hz.toarray()
    m, n = H.shape

    # Construct spacetime decoding graph H_dec for (rounds + 1) time slices.
    # Rightmost block encodes "time-like" edges to enforce difference syndromes.
    H_dec = np.kron(np.eye(rounds + 1, dtype=int), H)
    H_dec = np.concatenate(
        (H_dec, np.zeros([m * (rounds + 1), m * rounds], dtype=int)), axis=1
    )
    for j in range(m * rounds):
        H_dec[j, n * (rounds + 1) + j] = 1
        H_dec[m + j, n * (rounds + 1) + j] = 1
    H_dec = csr_matrix(H_dec)

    # Decoder: BP with OSD post-processing; scale error_rate ~ p2.
    bpd = BpOsdDecoder(
        H_dec,
        error_rate=float(5 * p2),
        max_iter=par[0],
        bp_method="ms",
        osd_method="osd_cs",
        osd_order=par[1],
    )
    # Alternative (not used here): LSD variant
    # lsd = BpLsdDecoder(...)

    # Build noisy circuit and sampler for repeated syndrome extraction.
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
        mx = code.hx.shape[0]
        k = m + mx
        for i in range(num_shots):
            # Parse measurement records: each round has (mx + m) check MRs in order [X then Z].
            rec = output[i]
            z_rounds = np.zeros((rounds, m), dtype=int)
            for r in range(rounds):
                base = r * k
                z_rounds[r, :] = rec[base + mx: base + mx + m]
            # Final data record is at the end
            data_meas = rec[-n:]

            syndromes = np.zeros([rounds + 1, m], dtype=int)
            syndromes[:rounds] = z_rounds
            syndromes[-1] = H @ data_meas % 2
            # Difference syndrome makes time-like edges local in H_dec.
            syndromes[1:] = syndromes[1:] ^ syndromes[:-1]

            # Decode flattened spacetime syndrome; fold time slices to a data correction.
            bpd_output = np.reshape(
                bpd.decode(np.ravel(syndromes))[: n * (rounds + 1)], [rounds + 1, n]
            )
            correction = bpd_output.sum(axis=0) % 2

            # Apply correction to final data measurement and detect a logical flip.
            final_state = data_meas ^ correction
            if (code.lz @ final_state % 2).any():
                failures += 1

    return failures
