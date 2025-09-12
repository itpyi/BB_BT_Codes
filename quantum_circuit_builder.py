"""Stim circuit construction utilities for repeated syndrome extraction.

Entry point:
- generate_full_circuit: composes repeated rounds of Z and X extraction followed
  by a final data measurement, injecting SPAM noise where applicable.
"""

import numpy as np
from cnot_scheduling import generate_edge_colored_syndrome_circuit
import stim
from typing import Any, Protocol, List
from scipy.sparse import csr_matrix








def generate_edge_colored_syndrome_layer(
    code_hx: Any, 
    code_hz: Any,
    x_checks: List[int],
    z_checks: List[int], 
    p1: float,
    p2: float,
    seed: int
) -> stim.Circuit:
    """Return Stim circuit for syndrome extraction using edge-coloring scheduling.
    
    Args:
        code_hx: X stabilizer parity check matrix
        code_hz: Z stabilizer parity check matrix  
        x_checks: X check qubit indices
        z_checks: Z check qubit indices
        p1: Single-qubit depolarizing error rate
        p2: Two-qubit depolarizing error rate
        seed: Random seed for edge-coloring shuffling
    """
    c = stim.Circuit()
    
    # Generate Z stabilizer syndrome circuit (stab_type=0)
    z_circuit = generate_edge_colored_syndrome_circuit(
        code_hz, z_checks, stab_type=0, p1=p1, p2=p2, seed=seed
    )
    c += z_circuit
    
    # Generate X stabilizer syndrome circuit (stab_type=1)
    x_circuit = generate_edge_colored_syndrome_circuit(
        code_hx, x_checks, stab_type=1, p1=p1, p2=p2, seed=seed
    )
    c += x_circuit
    
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
    
    Args:
        code: CSS code with hx, hz, lz matrices
        rounds: Number of syndrome extraction rounds
        p1: Single-qubit depolarizing error rate
        p2: Two-qubit depolarizing error rate  
        p_spam: SPAM error rate for measurements
        seed: Random seed
    """
    mx, n = code.hx.shape
    mz = code.hz.shape[0]
    data_qubits = list(range(n))
    x_checks = list(range(n, n + mx))
    z_checks = list(range(n + mx, n + mx + mz))

    c = stim.Circuit()

    # Build syndrome extraction layer using edge-coloring scheduling.
    synd_circuit = generate_edge_colored_syndrome_layer(code.hx, code.hz, x_checks, z_checks, p1, p2, seed)

    # Round 1: initialize all qubits (data qubits to |0>, ancillas reset); build syndrome layer; measure all checks once (no SPAM).
    c.append("R", data_qubits + x_checks + z_checks)
    c += synd_circuit
    all_checks = x_checks + z_checks  # Order defines MR record order

    c.append("X_ERROR", all_checks, p_spam)
    c.append("MR", all_checks)
    # DETECTORs for first round Z checks only (stabilizer expectation)
    for j in range(mz):
        # Current Z measurement records are the last mz of the combined MR
        c.append("DETECTOR", [stim.target_rec(-mz + j)])
    
    
    c.append("TICK")

    # Rounds 2..n: use a REPEAT block. Each iteration performs syndrome measurement.
    if rounds > 1:
        body = stim.Circuit()
        # Build syndrome layer, then measure all checks together with SPAM
        body += synd_circuit
        body.append("X_ERROR", all_checks, p_spam)
        body.append("MR", all_checks)
        # DETECTORs for Z: compare current vs previous Z records
        for j in range(mz):
            body.append(
                "DETECTOR",
                [
                    stim.target_rec(-mz + j),  # current Z (last mz)
                    stim.target_rec(
                        -(mx + mz) - mz + j
                    ),  # previous Z (last mz of prev block)
                ],
            )
        # DETECTORs for X: compare current vs previous X records
        for j in range(mx):
            body.append(
                "DETECTOR",
                [
                    stim.target_rec(-(mx + mz) + j),  # current X (first mx)
                    stim.target_rec(
                        -2 * (mx + mz) + j
                    ),  # previous X (first mx of prev block)
                ],
            )
        

        # Wrap in a REPEAT block for rounds 2..n
        repeat_count = rounds - 1
        indented = "\n".join(
            "    " + line for line in str(body).splitlines() if line.strip()
        )
        c += stim.Circuit(f"REPEAT {repeat_count} {{\n{indented}\n}}\n")

    # Final: data Z measurement with SPAM, and observables from Z logicals.
    c.append("X_ERROR", data_qubits, p_spam)
    c.append("MR", data_qubits)

    # Add detectors comparing previous Z stabilizers with current data measurements
    # Each Z stabilizer check should be consistent with the data qubit measurements
    hz_csr = csr_matrix(code.hz)
    for z_check_idx in range(mz):
        # Get the data qubits that participate in this Z stabilizer
        data_qubits_in_check = hz_csr.getrow(z_check_idx).indices.tolist()

        if data_qubits_in_check:
            # Create detector: previous Z measurement XOR current data measurements
            detector_targets = []

            # Previous Z stabilizer measurement (from the last syndrome round)
            # After the REPEAT block, we append an MR over data qubits (n records).
            # The last Z-check results from the final syndrome MR are therefore
            # located at rec[-n - mz ... -n - 1].
            # HERE BUG is FIXED BY CODEX
            # The previous answer  stim.target_rec(-mx - mz + z_check_idx) is wrong.
            # BB code it works since n = mx
            # BT code it fails since n = 3*mx
            detector_targets.append(stim.target_rec(-n - mz + z_check_idx))

            # Current data qubit measurements (most recent n measurements)
            for data_idx in data_qubits_in_check:
                detector_targets.append(stim.target_rec(-n + data_idx))

            c.append("DETECTOR", detector_targets)


    # OBSERVABLE_INCLUDE for each Z logical; assume lz is sparse or coercible.
    # Coerce to CSR, orient so rows enumerate logical-Z operators.
    lz_csr = csr_matrix(code.lz)
    if lz_csr.shape[1] != n and lz_csr.shape[0] == n:
        lz_csr = lz_csr.T
    obs_idx = 0
    for r in range(lz_csr.shape[0]):
        # Readable row support extraction
        supp = lz_csr.getrow(r).indices.tolist()
        if supp:
            recs = [stim.target_rec(-n + q) for q in supp]
            c.append("OBSERVABLE_INCLUDE", recs, obs_idx)
            obs_idx += 1
    
    # diagram = c.diagram('detslice-with-ops-svg', tick=range(0, 5), filter_coords=['D265', 'D268', 'D278', 'D279'])
    # # Convert diagram object to string before writing
    # with open("circuit_debug_diagram.svg", "w") as f:
    #     f.write(str(diagram))

    # Optional heavy debug artifacts guarded by env flag
    import os as _os
    if _os.getenv("QEC_DEBUG_DIAGRAM"):
        diagram = c.diagram('timeline-svg')
        with open("circuit_debug_diagram.svg", "w") as f:
            f.write(str(diagram))
        c.to_file("circuit_debug_circuit.stim")
    print(f"mx={mx}, mz={mz}, n={n}, rounds={rounds}, total_ticks={len(c)}")

    return c
