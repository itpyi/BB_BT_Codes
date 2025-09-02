"""Stim circuit construction utilities for repeated syndrome extraction.

Two entry points:
- generate_synd_circuit: builds a single X or Z stabilizer measurement layer
  with edge-colored CNOT scheduling and depolarizing noise.
- generate_full_circuit: composes repeated rounds of Z and X extraction followed
  by a final data measurement, injecting SPAM noise where applicable.
"""

import numpy as np
from edge_coloring import edge_color_bipartite
import stim
from networkx import relabel_nodes
from networkx.algorithms import bipartite
from typing import Iterable, Any, Protocol, Sequence, cast


def generate_synd_circuit(
    H: Any,
    checks: Sequence[int],
    stab_type: int,
    p1: float,
    p2: float,
    seed: int,
) -> stim.Circuit:
    """Return Stim circuit for one stabilizer extraction layer.

    Parameters
    - H: biadjacency matrix (CSR/array) mapping check-to-data for a CSS half.
    - checks: iterable of qubit indices for the ancilla (check) qubits.
    - stab_type: 0 for Z-stabilizers (no Hadamards), 1 for X-stabilizers
      (surround CNOTs with H on checks to rotate basis).
    - p1, p2: single/two-qubit depolarizing error rates applied after gates.
    - seed: RNG seed to shuffle color layers (0 means no shuffle).
    """
    m, n = H.shape

    # Build Tanner graph and relabel to match the global qubit index map used
    # by the full circuit (data first, then X-checks, then Z-checks).
    tanner_graph = bipartite.from_biadjacency_matrix(H)
    mapping = {i: checks[i] for i in range(m)}
    mapping.update({i: i - m for i in range(m, n + m)})
    tanner_graph = relabel_nodes(tanner_graph, mapping)

    # Edge-color the bipartite graph to schedule disjoint CNOT layers.
    coloring = edge_color_bipartite(tanner_graph)
    if seed != 0:
        rng = np.random.default_rng(seed=seed)
        # Shuffle in-place; cast to Any to satisfy mypy on numpy API.
        rng.shuffle(cast(Any, coloring), axis=0)

    c = stim.Circuit()

    # For X-stabilizers, rotate ancillas into X basis via H.
    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p1)

    for r in coloring:
        # Apply each edge-color class as one CNOT layer.
        data_qbts = set(np.arange(H.shape[1]))
        for g in r:
            # g = (data_idx, check_idx), consume data from the idle set.
            data_qbts.remove(g[0])
            targets = g[::-1] if stab_type else g
            c.append("CX", targets)
            c.append("DEPOLARIZE2", targets, p2)
        # Idle data qubits accrue single-qubit depolarizing noise this layer.
        c.append("DEPOLARIZE1", data_qbts, p1)

    # Undo the basis rotation for X-stabilizers.
    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p1)
    return c


# Only tracks Z syndrome measurements in the final state.
class CSSCode(Protocol):
    hx: Any
    hz: Any


def generate_full_circuit(
    code: CSSCode,
    rounds: int,
    p1: float,
    p2: float,
    p_spam: float,
    seed: int,
) -> stim.Circuit:
    """Return a Stim circuit for repeated X/Z extraction and final data readout.

    Parameters
    - code: CSS code with ``hx`` and ``hz``.
    - rounds: number of repeated extraction rounds.
    - p1, p2: single/two-qubit depolarizing error rates.
    - p_spam: SPAM (X) error rate for ancilla init, resets, and measurements.
    - seed: RNG seed relayed to sub-circuits.
    """
    mx, n = code.hx.shape
    mz = code.hz.shape[0]
    data_qubits = range(n)
    x_checks = range(n, n + mx)
    z_checks = range(n + mx, n + mx + mz)

    c = stim.Circuit()

    # Per-round Z then X extraction building blocks.
    z_synd_circuit = generate_synd_circuit(code.hz, z_checks, 0, p1, p2, seed)
    x_synd_circuit = generate_synd_circuit(code.hx, x_checks, 1, p1, p2, seed)

    # Ancilla initialization errors (SPAM) before rounds begin.
    c.append("X_ERROR", z_checks, p_spam)
    c.append("X_ERROR", x_checks, p_spam)

    # Repeat rounds of Z then X extraction; include mid-circuit SPAM effects.
    c_se = stim.Circuit()
    # Z syndrome measurement (measure and SPAM on checks)
    c_se += z_synd_circuit
    c_se.append("X_ERROR", z_checks, p_spam)
    c_se.append("MR", z_checks)
    c_se.append("X_ERROR", z_checks, p_spam)
    # X syndrome measurement (reset then SPAM on checks)
    c_se += x_synd_circuit
    c_se.append("R", x_checks)
    c_se.append("X_ERROR", x_checks, p_spam)

    c += c_se * rounds

    # Final transversal data measurement with SPAM.
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)
    return c
