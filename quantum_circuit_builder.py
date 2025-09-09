"""Stim circuit construction utilities for repeated syndrome extraction.

Entry point:
- generate_full_circuit: composes repeated rounds of Z and X extraction followed
  by a final data measurement, injecting SPAM noise where applicable.
"""

import numpy as np
from direct_cnot_scheduling import schedule_syndrome_cnots_only
import stim
from typing import Any, Protocol, List
from scipy.sparse import csr_matrix




def _add_meta_check_detectors(
    circuit: stim.Circuit, 
    h_meta: Any, 
    mz: int, 
    mx: int = 0, 
    is_first_round: bool = True
) -> None:
    """Add extra DETECTORs for BT meta checks (single shot capability).
    
    For BT codes, H_meta = [B^T, A^T, C^T] provides meta checks on Z stabilizers.
    Each row of H_meta defines a meta check that acts on specific Z stabilizers.
    
    Args:
        circuit: Stim circuit to add detectors to
        h_meta: Meta check matrix (r_meta x n_meta)
        mz: Number of Z checks (n_meta)
        mx: Number of X checks (for record offset calculation)
        is_first_round: If True, first round (no comparison with previous)
    """
    h_meta_csr = csr_matrix(h_meta)
    r_meta = h_meta_csr.shape[0]
    # print("=" * 40)
    # print(f"Adding {r_meta} meta check detectors (is_first_round={is_first_round})")
    # print("=" * 40)
    
    for meta_idx in range(r_meta):
        # Get which Z stabilizers this meta check acts on
        z_stab_indices = h_meta_csr.getrow(meta_idx).indices.tolist()
        
        if z_stab_indices:
            # I think the meta check only in the current round.
            # First round: meta check on current Z measurements
            # Z measurements are the last mz records
            meta_targets = [stim.target_rec(-mz + z_idx) for z_idx in z_stab_indices]
            # print("=" * 40)
            # print(meta_targets)
            # print("=" * 40)
            circuit.append("DETECTOR", meta_targets)


def _add_final_meta_check_detectors(
    circuit: stim.Circuit,
    h_meta: Any,
    hz: Any, 
    n: int,
    mz: int
) -> None:
    """Add meta check DETECTORs after final data measurement.
    
    For the final round, we only add detectors that compare the consistency
    of the final data measurements with the logical space structure.
    We don't add meta-check specific detectors here as they would create
    non-deterministic detector issues.
    
    Args:
        circuit: Stim circuit to add detectors to
        h_meta: Meta check matrix (r_meta x n_meta)  
        hz: Z stabilizer matrix (mz x n)
        n: Number of data qubits
        mz: Number of Z stabilizers
    """
    # For now, we'll skip the final meta check detectors to avoid
    # non-deterministic detector issues. The regular stabilizer-data
    # consistency checks are sufficient for the final round.
    # 
    # The meta check capability is already provided by the detectors
    # in the syndrome extraction rounds (1 through n).
    pass


def generate_direct_syndrome_circuit(
    code_hx: Any, 
    code_hz: Any,
    x_checks: List[int],
    z_checks: List[int], 
    p2: float
) -> stim.Circuit:
    """Return Stim circuit for syndrome extraction using direct parallel scheduling.
    
    Args:
        code_hx: X stabilizer parity check matrix
        code_hz: Z stabilizer parity check matrix  
        x_checks: X check qubit indices
        z_checks: Z check qubit indices
        p2: Two-qubit depolarizing error rate
    """
    # Convert matrices to stabilizer lists
    hx_csr = csr_matrix(code_hx)
    hz_csr = csr_matrix(code_hz)
    
    x_stabilizers = []
    for i in range(hx_csr.shape[0]):
        stabilizer = hx_csr.getrow(i).indices.tolist()
        if stabilizer:  # Only add non-empty stabilizers
            x_stabilizers.append(stabilizer)
    
    z_stabilizers = []
    for i in range(hz_csr.shape[0]):
        stabilizer = hz_csr.getrow(i).indices.tolist()
        if stabilizer:  # Only add non-empty stabilizers
            z_stabilizers.append(stabilizer)
    
    # Create circuit using direct scheduling
    c = stim.Circuit()
    c.append("TICK")
    
    # Use the direct scheduling approach (CNOTs only, no measurements)
    schedule_syndrome_cnots_only(
        c, z_stabilizers, z_checks, x_stabilizers, x_checks, p2
    )
    
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
    meta_check: bool = False,
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
        meta_check: If True, add extra DETECTORs for BT meta checks (single shot)
    """
    print(f"[DEBUG] generate_full_circuit called with meta_check={meta_check}, hasattr(code, 'h_meta')={hasattr(code, 'h_meta')}")
    mx, n = code.hx.shape
    mz = code.hz.shape[0]
    data_qubits = list(range(n))
    x_checks = list(range(n, n + mx))
    z_checks = list(range(n + mx, n + mx + mz))

    c = stim.Circuit()

    # Build syndrome extraction layer using direct parallel scheduling.
    synd_circuit = generate_direct_syndrome_circuit(code.hx, code.hz, x_checks, z_checks, p2)

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
    
    # Add meta check DETECTORs for first round (BT single shot)
    if meta_check and hasattr(code, 'h_meta'):
        _add_meta_check_detectors(c, code.h_meta, mz, is_first_round=True)
    
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
        
        # Add meta check DETECTORs for subsequent rounds (BT single shot)
        if meta_check and hasattr(code, 'h_meta'):
            _add_meta_check_detectors(body, code.h_meta, mz, mx, is_first_round=False)

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

    # Add meta check DETECTORs after final data measurement (BT single shot)
    if meta_check and hasattr(code, 'h_meta'):
        _add_final_meta_check_detectors(c, code.h_meta, code.hz, n, mz)

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
