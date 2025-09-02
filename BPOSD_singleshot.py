"""Single-shot BP/OSD helpers.

Implements a two-stage decode: a rolling (bulk) pass over rounds to absorb
syndrome history, followed by a final pass on the terminal syndrome.
"""

import numpy as np
from ldpc import BpOsdDecoder
from ldpc.bplsd_decoder import BpLsdDecoder
from scipy.sparse import csr_matrix, hstack, vstack, identity
import circuit_utils
from typing import Any, Sequence, Tuple, Union, Protocol


def bulk_BPOSD_decode(
    syndromes: np.ndarray,
    H: Union[np.ndarray, csr_matrix],
    M: Union[np.ndarray, csr_matrix],
    p: float,
    pars: Sequence[int],
    cycles: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return aggregate correction and residual syndrome after ``cycles``.

    Parameters
    - syndromes: array of shape ``(cycles+1, m)``; per-round Z-syndromes and final.
    - H: parity-check matrix (``m x n``) for Z errors.
    - M: parity-check matrix for ancilla/time-like constraints to stabilize history.
    - p: error rate used to set decoder weights.
    - pars: ``[bp_iters, osd_order]`` for the LSD decoder.
    - cycles: number of repeated measurement rounds.

    Returns ``(bulk_correction, corr_syndrome)`` where ``bulk_correction`` is a
    length-``n`` binary vector and ``corr_syndrome`` is the residual length-``m``
    syndrome to be handled by a final pass.
    """
    m, n = H.shape

    # Build an augmented system coupling data and syndrome-memory bits.
    H_dec1 = hstack([H, identity(m, dtype=int, format="csr")], format="csr")
    H_dec2 = hstack([csr_matrix((M.shape[0], n), dtype=int), M], format="csr")
    H_dec = vstack([H_dec1, H_dec2], format="csr")

    bpd = BpLsdDecoder(
        H_dec,
        error_rate=float(5 * p),
        max_iter=pars[0],
        bp_method="ms",
        osd_method="lsd_cs",
        osd_order=pars[1],
        schedule="serial",
    )

    bulk_correction = np.zeros(n, dtype=int)
    corr_syndrome = np.zeros(m, dtype=int)
    for c in range(cycles):
        # Roll in the correction-induced syndrome from previous steps.
        syndrome = syndromes[c] ^ corr_syndrome
        # Decode joint (data | memory) variables; keep data part.
        correction = bpd.decode(
            np.concatenate((syndrome, np.zeros(M.shape[0], dtype=int)))
        )[:n]
        bulk_correction ^= correction
        corr_syndrome ^= H @ correction % 2

    return bulk_correction, corr_syndrome


class CSSCode(Protocol):
    hx: Any
    hz: Any
    lz: Any
    N: int


def get_BPOSD_failures(
    code: CSSCode,
    Mz: Union[np.ndarray, csr_matrix],
    pars: Sequence[int],
    noise_pars: Tuple[float, float, float],
    cycles: int,
    iters: int,
    seed: int,
) -> int:
    """Return number of logical failures using single-shot BP/LSD.

    Parameters
    - code: CSS code with ``hz`` and ``lz`` for Z errors/logicals.
    - Mz: constraint matrix for Z ancilla/time-like checks used in bulk pass.
    - pars: ``[bp_iters, osd_order]`` for decoders.
    - noise_pars: tuple ``(p1, p2, p_spam)`` for circuit noise.
    - cycles: number of repeated syndrome rounds.
    - iters: number of Monte Carlo shots.
    - seed: RNG seed for circuit generation.
    """
    H = code.hz
    logicals = code.lz
    m = H.shape[0]
    n = code.N
    p1, p2, p_spam = noise_pars

    # Final-pass decoder on the residual syndrome (after bulk pass).
    # bpd = BpOsdDecoder(H, error_rate=float(p2), max_iter=pars[0], ...)
    bpd = BpLsdDecoder(
        H,
        error_rate=float(5 * p2),
        max_iter=pars[0],
        bp_method="ms",
        osd_method="lsd_cs",
        osd_order=pars[1],
        schedule="serial",
    )

    c = circuit_utils.generate_full_circuit(
        code, rounds=cycles, p1=p1, p2=p2, p_spam=p_spam, seed=seed
    )
    sampler = c.compile_sampler()

    failures = 0
    outer_reps = iters // 256  # Stim samples a minimum of 256 shots at a time
    remainder = iters % 256
    for j in range(outer_reps + 1):
        num_shots = 256 if j < outer_reps else remainder
        if num_shots == 0:
            continue
        outputs = sampler.sample(shots=num_shots)
        for i in range(num_shots):
            output = outputs[i]

            # Build per-round Z-syndromes and terminal parity check.
            syndromes = np.zeros([cycles + 1, m], dtype=int)
            syndromes[:cycles] = output[:-n].reshape([cycles, m])
            syndromes[-1] = H @ output[-n:] % 2

            # Rolling bulk pass, then final correction on residual syndrome.
            bulk_correction, corr_syndrome = bulk_BPOSD_decode(
                syndromes, H, Mz, p2, pars, cycles
            )
            final_correction = bpd.decode(corr_syndrome ^ syndromes[-1])

            final_state = output[-n:] ^ bulk_correction ^ final_correction
            if (logicals @ final_state % 2).any():
                failures += 1

    return failures
