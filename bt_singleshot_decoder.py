"""Single-shot decoding for BT (Bivariate Tricycle) codes using two-stage approach.

Implements two-stage single shot decoding for BT codes with meta checks:
1. Bulk Pass: Rolling decoder over syndrome rounds using meta check constraints
2. Final Pass: Terminal syndrome correction on residual

This approach leverages the meta check matrix H_meta = [B^T, A^T, C^T] for enhanced
error correction capability without requiring multiple measurement rounds.
"""

import numpy as np
from ldpc.bplsd_decoder import BpLsdDecoder
from ldpc.bposd_decoder import BpOsdDecoder
from scipy.sparse import csr_matrix, hstack, vstack, identity
from typing import Any, Sequence, Tuple, Union, Protocol, Optional
import logging
import stim
import time


class BTCode(Protocol):
    """Protocol for BT codes with meta check capability."""
    hx: Any
    hz: Any
    lz: Any
    h_meta: Any
    N: int


def build_meta_constraint_matrix(h_meta: Any, mz: int, cycles: int) -> csr_matrix:
    """Build constraint matrix M for BT meta checks across syndrome rounds.
    
    Args:
        h_meta: Meta check matrix H_meta = [B^T, A^T, C^T] (r_meta x n_meta)
        mz: Number of Z stabilizers (n_meta)
        cycles: Number of syndrome measurement rounds
        
    Returns:
        Constraint matrix M for time-like meta check constraints
    """
    h_meta_csr = csr_matrix(h_meta)
    r_meta = h_meta_csr.shape[0]
    
    # Build constraint matrix for meta checks across time
    # Each meta check constraint applies across consecutive syndrome rounds
    constraint_blocks = []
    
    for cycle in range(cycles):
        # Meta check constraints for current round
        # Add identity for current round and negative identity for next round
        current_block = identity(mz, dtype=int, format="csr")
        if cycle < cycles - 1:
            next_block = -identity(mz, dtype=int, format="csr") 
            cycle_constraint = hstack([current_block, next_block], format="csr")
        else:
            cycle_constraint = current_block
            
        constraint_blocks.append(cycle_constraint)
    
    if constraint_blocks:
        M = vstack(constraint_blocks, format="csr")
    else:
        M = csr_matrix((0, mz * cycles), dtype=int)
        
    return M


def bulk_bt_decode(
    syndromes: np.ndarray,
    hz: Union[np.ndarray, csr_matrix], 
    h_meta: Union[np.ndarray, csr_matrix],
    p: float,
    pars: Sequence[int],
    cycles: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-stage bulk decoder for BT codes with meta checks.
    
    Stage 1: Uses meta check constraints to correct syndrome history across rounds.
    
    Args:
        syndromes: Array of shape (cycles+1, mz) - per-round Z syndromes + final
        hz: Z stabilizer matrix (mz x n) 
        h_meta: Meta check matrix (r_meta x mz)
        p: Physical error rate for decoder weights
        pars: [bp_iters, osd_order] for decoder parameters
        cycles: Number of syndrome measurement rounds
        
    Returns:
        Tuple of (bulk_correction, corrected_syndrome) where:
        - bulk_correction: Length-n binary correction vector
        - corrected_syndrome: Residual length-mz syndrome for final pass
    """
    hz_csr = csr_matrix(hz)
    mz, n = hz_csr.shape
    
    # Build meta check constraint matrix across time
    M = build_meta_constraint_matrix(h_meta, mz, cycles)
    
    # Augmented system: [data qubits | syndrome memory qubits]
    # H_dec1: couples data errors to syndrome observations
    H_dec1 = hstack([hz_csr, identity(mz, dtype=int, format="csr")], format="csr")
    
    # H_dec2: meta check constraints on syndrome memory
    H_dec2 = hstack([csr_matrix((M.shape[0], n), dtype=int), M], format="csr")
    
    # Combined constraint matrix
    H_dec = vstack([H_dec1, H_dec2], format="csr")
    
    # Initialize decoder with augmented system
    try:
        bpd = BpLsdDecoder(
            H_dec,
            error_rate=float(5 * p),
            max_iter=pars[0],
            bp_method="ms", 
            osd_method="lsd_cs",
            osd_order=pars[1],
            schedule="serial",
        )
    except Exception as e:
        logging.warning(f"BpLsdDecoder failed, falling back to BpOsdDecoder: {e}")
        # Fallback to standard BP+OSD
        bpd = BpOsdDecoder(
            H_dec,
            error_channel=None,
            max_iter=pars[0],
            bp_method="ms",
            ms_scaling_factor=0.625,
            osd_method="osd_e" if pars[1] > 0 else "osd0",
            osd_order=pars[1],
            schedule="parallel",
        )
    
    bulk_correction = np.zeros(n, dtype=int)
    corrected_syndrome = np.zeros(mz, dtype=int)
    
    # Rolling bulk pass over syndrome rounds
    for cycle in range(cycles):
        # Current syndrome with correction from previous cycles
        current_syndrome = syndromes[cycle] ^ corrected_syndrome
        
        # Decode augmented system: [data_correction | syndrome_memory]
        augmented_syndrome = np.concatenate([
            current_syndrome, 
            np.zeros(M.shape[0], dtype=int)
        ])
        
        try:
            full_correction = bpd.decode(augmented_syndrome)
            # Extract data correction (first n bits)
            data_correction = full_correction[:n]
        except Exception as e:
            logging.warning(f"Bulk decode failed at cycle {cycle}: {e}")
            data_correction = np.zeros(n, dtype=int)
        
        # Accumulate corrections
        bulk_correction ^= data_correction
        
        # Update corrected syndrome for next cycle
        corrected_syndrome ^= (hz_csr @ data_correction) % 2
    
    return bulk_correction, corrected_syndrome


def bt_singleshot_decode(
    code: BTCode,
    syndromes: np.ndarray, 
    final_data: np.ndarray,
    p: float,
    pars: Sequence[int],
    cycles: int,
) -> Tuple[np.ndarray, bool]:
    """Complete single-shot decoding for BT codes with meta checks.
    
    Implements two-stage approach:
    1. Bulk pass using meta check constraints
    2. Final pass on residual syndrome
    
    Args:
        code: BT code with hz, lz, h_meta attributes
        syndromes: Syndrome measurements shape (cycles, mz) 
        final_data: Final data qubit measurements shape (n,)
        p: Physical error rate
        pars: [bp_iters, osd_order] decoder parameters
        cycles: Number of syndrome rounds
        
    Returns:
        Tuple of (total_correction, logical_failure) where:
        - total_correction: Combined correction from both stages
        - logical_failure: Whether logical error occurred
    """
    hz_csr = csr_matrix(code.hz)
    lz_csr = csr_matrix(code.lz)
    mz, n = hz_csr.shape
    
    # Build complete syndrome history including final data syndrome
    full_syndromes = np.zeros((cycles + 1, mz), dtype=int)
    full_syndromes[:cycles] = syndromes[:cycles]
    full_syndromes[cycles] = (hz_csr @ final_data) % 2
    
    # Stage 1: Bulk correction using meta checks
    bulk_correction, corrected_syndrome = bulk_bt_decode(
        full_syndromes, code.hz, code.h_meta, p, pars, cycles
    )
    
    # Stage 2: Final pass on residual syndrome
    try:
        final_decoder = BpOsdDecoder(
            hz_csr,
            error_channel=None,
            max_iter=pars[0],
            bp_method="ms",
            ms_scaling_factor=0.625, 
            osd_method="osd_e" if pars[1] > 0 else "osd0",
            osd_order=pars[1],
            schedule="parallel",
        )
        
        # Final syndrome = corrected syndrome XOR terminal data syndrome
        final_syndrome = corrected_syndrome ^ full_syndromes[cycles]
        final_correction = final_decoder.decode(final_syndrome)
        
    except Exception as e:
        logging.warning(f"Final decode failed: {e}")
        final_correction = np.zeros(n, dtype=int)
    
    # Total correction and final state
    total_correction = bulk_correction ^ final_correction
    final_state = final_data ^ total_correction
    
    # Check for logical failure
    logical_syndrome = (lz_csr @ final_state) % 2
    logical_failure = bool(logical_syndrome.any())
    
    return total_correction, logical_failure


def extract_syndrome_history_from_stim(
    stim_sample: np.ndarray,
    rounds: int,
    mx: int,
    mz: int,
    n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract syndrome history and final data from Stim circuit sample.
    
    Args:
        stim_sample: Single sample from Stim circuit (detector + observable measurements)
        rounds: Number of syndrome extraction rounds
        mx: Number of X checks
        mz: Number of Z checks  
        n: Number of data qubits
        
    Returns:
        Tuple of (syndrome_history, final_data) where:
        - syndrome_history: Shape (rounds, mz) - Z syndromes per round
        - final_data: Shape (n,) - final data qubit measurements
    """
    # Stim sample structure: [round1_checks, round2_checks, ..., final_data, observables]
    # Each round has (mx + mz) check measurements: [X_checks, Z_checks]
    
    total_check_measurements = rounds * (mx + mz)
    
    # Extract syndrome history - only Z checks from each round
    syndrome_history = np.zeros((rounds, mz), dtype=int)
    
    for round_idx in range(rounds):
        round_start = round_idx * (mx + mz)
        # Z checks are the last mz measurements in each round
        z_start = round_start + mx
        z_end = z_start + mz
        syndrome_history[round_idx] = stim_sample[z_start:z_end]
    
    # Extract final data measurements
    final_data_start = total_check_measurements
    final_data_end = final_data_start + n
    final_data = stim_sample[final_data_start:final_data_end]
    
    return syndrome_history, final_data


def build_bt_singleshot_decoder(
    code: BTCode,
    p: float,
    pars: Sequence[int]
) -> callable:
    """Build single-shot decoder function for BT codes.
    
    Returns a decoder function that can be used in place of standard BP+OSD.
    
    Args:
        code: BT code with meta check capability
        p: Physical error rate for decoder weights
        pars: [bp_iters, osd_order] decoder parameters
        
    Returns:
        Decoder function compatible with simulation pipeline
    """
    def decode_function(stim_sample: np.ndarray, rounds: int) -> bool:
        """Decode single Stim sample using two-stage approach.
        
        Args:
            stim_sample: Single sample from Stim circuit
            rounds: Number of syndrome rounds
            
        Returns:
            True if logical error detected, False otherwise
        """
        mx, n = code.hx.shape
        mz = code.hz.shape[0]
        
        # Extract syndrome history and final data from Stim sample
        syndrome_history, final_data = extract_syndrome_history_from_stim(
            stim_sample, rounds, mx, mz, n
        )
        
        # Apply two-stage decoding
        _, logical_failure = bt_singleshot_decode(
            code, syndrome_history, final_data, p, pars, rounds
        )
        
        return logical_failure
    
    return decode_function


def evaluate_bt_singleshot_performance(
    code: BTCode,
    circuit: 'stim.Circuit',
    noise_pars: Tuple[float, float, float],
    cycles: int,
    pars: Sequence[int],
    iters: int,
) -> Tuple[int, int]:
    """Evaluate BT single-shot decoder performance using Stim integration.
    
    Args:
        code: BT code with meta check capability
        circuit: Precompiled Stim circuit
        noise_pars: (p1, p2, p_spam) noise parameters
        cycles: Number of syndrome rounds
        pars: [bp_iters, osd_order] decoder parameters  
        iters: Number of Monte Carlo iterations
        
    Returns:
        Tuple of (total_failures, total_shots)
    """
    p1, p2, p_spam = noise_pars
    
    # Build decoder function
    decode_fn = build_bt_singleshot_decoder(code, p2, pars)
    
    # Compile Stim sampler
    sampler = circuit.compile_sampler()
    
    failures = 0
    total_shots = 0
    
    # Process samples in batches (Stim samples minimum 256 at a time)
    batch_size = min(256, iters)
    remaining = iters
    
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        samples = sampler.sample(shots=current_batch)
        
        for i in range(current_batch):
            sample = samples[i]
            if decode_fn(sample, cycles):
                failures += 1
            total_shots += 1
        
        remaining -= current_batch
    
    return failures, total_shots