# BB Code Simulation

Simulation framework for studying logical error rates of Bivariate Bicycle (BB) quantum error correction codes.

## Overview

This project simulates BB quantum LDPC codes to measure their logical error rates under different noise models. It supports both single-threaded and multiprocess simulations with comprehensive result visualization.

**Current**: BB code logical error rate simulation  
**Future**: Bivariate tricycle codes, trivariate tricycle codes, transversal CNOT gates

## Files

- `bivariate_bicycle_codes.py` - BB code construction
- `simulation_serial.py` - Single-threaded simulations  
- `simulation_multiprocess.py` - Multi-process simulations
- `results_parser_plotter.py` - Parse results and generate plots
- `quantum_circuit_builder.py` - Build quantum circuits
- `simulation_common.py` - Shared utilities
- `shared_utilities.py` - Helper functions

## Usage

### Install
```bash
pip install numpy scipy ldpc bposd stim networkx matplotlib pytest mypy
```

### Run simulation
```python
from simulation_serial import run_BB_serial_simulation

results = run_BB_serial_simulation(
    a_poly=[(2, 0), (1, 1), (0, 2)],  # x^2 + xy + y^2  
    b_poly=[(1, 0), (0, 1)],          # x + y
    l=4, m=5,
    p_min=0.001, p_max=0.01,
    num_points=5,
    rounds=3,
    max_shots=10000
)
```

### Command line
```bash
python simulation_multiprocess.py
python results_parser_plotter.py --resume results.csv --show
```

## Testing
```bash
python -m pytest test/
python -m mypy .
```

## References

- [ZSZ-codes-numerics](https://github.com/yifanhong/ZSZ-codes-numerics)
- [Stim](https://github.com/quantumlib/Stim) - Quantum circuit simulation
- [LDPC](https://github.com/quantumgizmos/ldpc) - BP+OSD decoding
- Bivariate Bicycle codes: [arXiv:2203.16958](https://arxiv.org/abs/2203.16958)
