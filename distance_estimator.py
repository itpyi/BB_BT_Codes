"""Empirical minimum logical weight estimator.

Draws i.i.d. error patterns at rate ``p`` and decodes with BP+OSD.
If a logical flips, records the weight; returns the minimum observed.
"""

import numpy as np
from scipy.sparse import csr_matrix
from ldpc import BpOsdDecoder
from typing import Sequence, Protocol, Any


class CSSCode(Protocol):
    hx: Any
    hz: Any
    lx: Any
    lz: Any
    N: int


def get_min_logical_weight(
    code: CSSCode, p: float, pars: Sequence[int], iters: int, Ptype: int
) -> int:
    """Return minimum observed logical weight after decoding.

    Parameters
    - code: CSS code with ``hx``, ``hz``, ``lx``, ``lz``, and ``N``.
    - p: i.i.d. error rate for Bernoulli draws.
    - pars: ``[bp_iters, osd_order]`` for BP/OSD.
    - iters: number of random error samples to draw.
    - Ptype: 0 for X errors/constraints, 1 for Z errors/constraints.
    """
    if Ptype == 0:
        H = code.hx
        logicals = code.lx
    else:
        H = code.hz
        logicals = code.lz

    n = code.N
    bp_iters = pars[0]
    osd_order = pars[1]

    # Two decoders available; bposd is used below.
    bposd = BpOsdDecoder(
        H,
        error_rate=p,
        max_iter=bp_iters,
        bp_method="minimum_sum",
        osd_method="osd_cs",
        osd_order=osd_order,
    )

    min_weight = n
    # Pre-draw errors for vectorized sampling; errors[i] is the i-th pattern.
    errors = (np.random.rand(iters, n) < p).astype(int)
    for i in range(iters):
        state = errors[i]
        syndrome = H @ state % 2
        correction = bposd.decode(syndrome)
        final_state = state ^ correction
        if (logicals @ final_state % 2).any():
            weight = int(np.sum(final_state))
            if 0 < weight < min_weight:
                min_weight = weight
    return min_weight
