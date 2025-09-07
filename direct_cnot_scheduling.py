"""Direct parallel CNOT scheduling for syndrome measurements.

Implements a straightforward approach where:
- For Z syndrome: Connect first qubit in Z stabilizer to corresponding Z-check qubit
- For X syndrome: Same but with swapped target/control
- Each layer realizes all CNOTs then applies depolarize2, followed by tick
"""

import stim
from typing import List, Set, Tuple, Dict


def schedule_z_syndrome_cnots(circuit: stim.Circuit, z_stabilizers: List[List[int]], 
                             z_check_qubits: List[int], noise_strength: float = 0.001) -> None:
    """Schedule CNOT gates for Z syndrome measurement.
    
    Args:
        circuit: Stim circuit to append operations to
        z_stabilizers: List of Z stabilizer qubit lists
        z_check_qubits: List of Z check qubit indices
        noise_strength: Depolarizing noise strength
    """
    if len(z_stabilizers) != len(z_check_qubits):
        raise ValueError("Number of Z stabilizers must match number of Z check qubits")
    
    # Determine maximum weight (number of layers needed)
    max_weight = max(len(stabilizer) for stabilizer in z_stabilizers) if z_stabilizers else 0
    
    # For each layer
    for layer in range(max_weight):
        cnots_in_layer = []
        
        # Collect all CNOTs for this layer
        for stab_idx, stabilizer in enumerate(z_stabilizers):
            if layer < len(stabilizer):
                # Z syndrome: data qubit is control, check qubit is target
                control = stabilizer[layer]
                target = z_check_qubits[stab_idx]
                cnots_in_layer.append((control, target))
        
        # Execute all CNOTs in parallel
        if cnots_in_layer:
            for control, target in cnots_in_layer:
                circuit.append("CNOT", [control, target])
            
            # Apply depolarizing noise to each CNOT pair
            for control, target in cnots_in_layer:
                circuit.append("DEPOLARIZE2", [control, target], noise_strength)
            
            # Add tick for this layer
            circuit.append("TICK")


def schedule_x_syndrome_cnots(circuit: stim.Circuit, x_stabilizers: List[List[int]], 
                             x_check_qubits: List[int], noise_strength: float = 0.001) -> None:
    """Schedule CNOT gates for X syndrome measurement.
    
    Args:
        circuit: Stim circuit to append operations to
        x_stabilizers: List of X stabilizer qubit lists
        x_check_qubits: List of X check qubit indices
        noise_strength: Depolarizing noise strength
    """
    if len(x_stabilizers) != len(x_check_qubits):
        raise ValueError("Number of X stabilizers must match number of X check qubits")
    
    # Determine maximum weight (number of layers needed)
    max_weight = max(len(stabilizer) for stabilizer in x_stabilizers) if x_stabilizers else 0
    
    # For each layer
    for layer in range(max_weight):
        cnots_in_layer = []
        
        # Collect all CNOTs for this layer
        for stab_idx, stabilizer in enumerate(x_stabilizers):
            if layer < len(stabilizer):
                # X syndrome: check qubit is control, data qubit is target (swapped)
                control = x_check_qubits[stab_idx]
                target = stabilizer[layer]
                cnots_in_layer.append((control, target))
        
        # Execute all CNOTs in parallel
        if cnots_in_layer:
            for control, target in cnots_in_layer:
                circuit.append("CNOT", [control, target])
            
            # Apply depolarizing noise to each CNOT pair
            for control, target in cnots_in_layer:
                circuit.append("DEPOLARIZE2", [control, target], noise_strength)
            
            # Add tick for this layer
            circuit.append("TICK")


def schedule_syndrome_cnots_only(circuit: stim.Circuit, 
                                z_stabilizers: List[List[int]], z_check_qubits: List[int],
                                x_stabilizers: List[List[int]], x_check_qubits: List[int],
                                noise_strength: float = 0.001) -> None:
    """Schedule CNOT operations for syndrome measurement without measurements.
    
    Args:
        circuit: Stim circuit to append operations to
        z_stabilizers: List of Z stabilizer qubit lists
        z_check_qubits: List of Z check qubit indices
        x_stabilizers: List of X stabilizer qubit lists
        x_check_qubits: List of X check qubit indices
        noise_strength: Depolarizing noise strength
    """
    # Initialize check qubits in |+> state for X measurements and |0> for Z measurements
    if x_check_qubits:
        circuit.append("H", x_check_qubits)
        circuit.append("DEPOLARIZE1", x_check_qubits, noise_strength)
        circuit.append("TICK")
    
    # Schedule Z syndrome CNOTs
    schedule_z_syndrome_cnots(circuit, z_stabilizers, z_check_qubits, noise_strength)
    
    # Schedule X syndrome CNOTs
    schedule_x_syndrome_cnots(circuit, x_stabilizers, x_check_qubits, noise_strength)
    
    # Rotate X basis back to Z basis for measurement
    if x_check_qubits:
        circuit.append("H", x_check_qubits)  
        circuit.append("DEPOLARIZE1", x_check_qubits, noise_strength)
        circuit.append("TICK")


def get_syndrome_layer_count(z_stabilizers: List[List[int]], x_stabilizers: List[List[int]]) -> int:
    """Get the total number of CNOT layers needed for syndrome measurement.
    
    Args:
        z_stabilizers: List of Z stabilizer qubit lists
        x_stabilizers: List of X stabilizer qubit lists
        
    Returns:
        Total number of CNOT layers
    """
    z_layers = max(len(stab) for stab in z_stabilizers) if z_stabilizers else 0
    x_layers = max(len(stab) for stab in x_stabilizers) if x_stabilizers else 0
    return z_layers + x_layers