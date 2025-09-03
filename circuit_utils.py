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
from scipy.sparse import csr_matrix, issparse


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
    # Ensure SciPy CSR for NetworkX biadjacency conversion
    H_csr = csr_matrix(H)
    m, n = H_csr.shape

    # No Idle error
    # p_idle = p1 / 100

    # Build Tanner graph and relabel to match the global qubit index map used
    # by the full circuit (data first, then X-checks, then Z-checks).
    tanner_graph = bipartite.from_biadjacency_matrix(H_csr)
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
    c.append("TICK")
    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p1)
        c.append("TICK")

    for r in coloring:
        # Apply each edge-color class as one CNOT layer.
        data_qbts = set(np.arange(n))
        for g in r:
            # g = (data_idx, check_idx), consume data from the idle set.
            data_qbts.remove(g[0])
            targets = g[::-1] if stab_type else g
            c.append("CX", targets)
            c.append("DEPOLARIZE2", targets, p2)
        # Idle data qubits accrue single-qubit depolarizing noise this layer.
        # 2025/9/3 Comment idle error
        # c.append("DEPOLARIZE1", data_qbts, p_idle)
        c.append("TICK")

    # Undo the basis rotation for X-stabilizers.
    if stab_type:
        c.append("H", checks)
        c.append("DEPOLARIZE1", checks, p1)
        c.append("TICK")
    return c


# Only tracks Z syndrome measurements in the final state.
class CSSCode(Protocol):
    hx: Any
    hz: Any
    lz: Any


def generate_full_circuit(
    code: CSSCode,
    rounds: int,
    p1: float,
    p2: float,
    p_spam: float,
    seed: int,
) -> stim.Circuit:
    """Return a Stim circuit with DETECTORs and correct round structure.

    Round 1: reset all qubits; measure only Z checks (no SPAM around MR).
    Rounds 2..n: measure Z checks (with SPAM around MR) and X checks (with SPAM
    around MR); add DETECTORs comparing current vs previous round for each check.
    Final: measure all data qubits in Z and include Z logicals as observables.
    """
    mx, n = code.hx.shape
    mz = code.hz.shape[0]
    data_qubits = list(range(n))
    x_checks = list(range(n, n + mx))
    z_checks = list(range(n + mx, n + mx + mz))

    c = stim.Circuit()

    # Build Z and X stabilizer extraction layers using graph scheduling.
    z_synd = generate_synd_circuit(code.hz, z_checks, 0, p1, p2, seed)
    x_synd = generate_synd_circuit(code.hx, x_checks, 1, p1, p2, seed)

    # Round 1: reset all qubits; build both Z and X layers; measure all checks once (no SPAM).
    c.append("R", data_qubits + x_checks + z_checks)
    c += z_synd
    c += x_synd
    all_checks = x_checks + z_checks  # Order defines MR record order

    c.append("X_ERROR", all_checks, p_spam)
    c.append("MR", all_checks)
    # DETECTORs for first round Z checks only (stabilizer expectation)
    for j in range(mz):
        # Current Z measurement records are the last mz of the combined MR
        c.append("DETECTOR", [stim.target_rec(-mz + j)])
    c.append("TICK")

    # Rounds 2..n: use a REPEAT block. Each iteration measures Z then X checks.
    if rounds > 1:
        body = stim.Circuit()
        # Build both Z and X layers, then measure all checks together with SPAM
        body += z_synd
        body += x_synd
        body.append("X_ERROR", all_checks, p_spam)
        body.append("MR", all_checks)
        # DETECTORs for Z: compare current vs previous Z records
        for j in range(mz):
            body.append(
                "DETECTOR",
                [
                    stim.target_rec(-mz + j),                       # current Z (last mz)
                    stim.target_rec(-(mx + mz) - mz + j),          # previous Z (last mz of prev block)
                ],
            )
        # DETECTORs for X: compare current vs previous X records
        for j in range(mx):
            body.append(
                "DETECTOR",
                [
                    stim.target_rec(-(mx + mz) + j),               # current X (first mx)
                    stim.target_rec(-2 * (mx + mz) + j),           # previous X (first mx of prev block)
                ],
            )

        # Wrap in a REPEAT block for rounds 2..n
        repeat_count = rounds - 1
        indented = "\n".join("    " + line for line in str(body).splitlines() if line.strip())
        c += stim.Circuit(f"REPEAT {repeat_count} {{\n{indented}\n}}\n")

    # Final: data Z measurement with SPAM, and observables from Z logicals.
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)
    # OBSERVABLE_INCLUDE for each Z logical operator row
    # Accept SciPy sparse or NumPy arrays; rows give data-qubit supports.
    lz_raw = code.lz
    if issparse(lz_raw):
        lz_csr = lz_raw if lz_raw.format == "csr" else csr_matrix(lz_raw)
        k = lz_csr.shape[0]
        for obs_idx in range(k):
            supp = lz_csr.getrow(obs_idx).indices.tolist()
            if supp:
                recs = [stim.target_rec(-n + q) for q in supp]
                c.append("OBSERVABLE_INCLUDE", recs, obs_idx)
    else:
        lz_arr = np.array(lz_raw)
        if lz_arr.ndim == 1:
            supp = np.nonzero(lz_arr)[0].tolist()
            if supp:
                recs = [stim.target_rec(-n + q) for q in supp]
                c.append("OBSERVABLE_INCLUDE", recs, 0)
        elif lz_arr.ndim == 2:
            k = lz_arr.shape[0]
            for obs_idx in range(k):
                supp = np.nonzero(lz_arr[obs_idx])[0].tolist()
                if supp:
                    recs = [stim.target_rec(-n + q) for q in supp]
                    c.append("OBSERVABLE_INCLUDE", recs, obs_idx)

    return c
