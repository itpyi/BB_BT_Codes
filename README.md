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
- `qec_simulation_core.py` - Core QEC simulation functionality
- `file_io_utils.py` - File I/O and parsing utilities

## Usage

### Install
```bash
pip install numpy scipy ldpc bposd stim networkx matplotlib pytest mypy
```

### JSON Configuration (Recommended)
Create a JSON config file (see `config_examples/`):
```json
{
  "description": "Small BB code test",
  "a_poly": [[3, 0], [0, 1], [0, 2]],
  "b_poly": [[0, 3], [1, 0], [2, 0]],
  "l": 6, "m": 6,
  "p_range": {"min": 0.001, "max": 0.007, "num_points": 3},
  "rounds_list": [6, 8],
  "max_shots": 1000,
  "max_errors": 10
}
```

Run simulations:
```bash
# Serial simulation
python simulation_serial.py --config config_examples/bb_small_test.json --output-dir results

# Multiprocess simulation  
python simulation_multiprocess.py --config config_examples/bb_threshold_study.json --output-dir results
```

### Python API
```python
from simulation_serial import run_BB_serial_simulation

results = run_BB_serial_simulation(
    a_poly=[(2, 0), (1, 1), (0, 2)],  # x^2 + xy + y^2  
    b_poly=[(1, 0), (0, 1)],          # x + y
    l=4, m=5,
    p_list=[0.001, 0.003, 0.005],
    rounds_list=[6, 8],
    max_shots=10000
)
```

### Parse and Plot Results
```bash
python results_parser_plotter.py --resume results/bb_6_6_serial_resume.csv --show
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
- Bivariate Bicycle codes: [arXiv:2308.07915](https://arxiv.org/abs/2308.07915)
