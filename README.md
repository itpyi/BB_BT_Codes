# BB Code Simulation

Simulation framework for studying logical error rates of Bivariate Bicycle (BB) quantum error correction codes.

## Overview

This project simulates BB quantum LDPC codes to measure their logical error rates under different noise models. It supports both single-threaded and multiprocess simulations with comprehensive result visualization.

**Current**: BB code logical error rate simulation  
**Future**: Bivariate tricycle codes, trivariate tricycle codes, transversal CNOT gates

## Files

### Core QEC Framework
- `qec_simulation_core.py` - Core QEC simulation functionality (supports BB/BT/TT codes)
- `simulation_generic.py` - **NEW: Unified simulation runner for all code types**
- `quantum_circuit_builder.py` - Build quantum circuits (unified API)
- `results_parser_plotter.py` - Parse results and generate plots
- `file_io_utils.py` - File I/O and parsing utilities

### Code Construction  
- `bivariate_bicycle_codes.py` - BB (Bivariate Bicycle) code construction
- `bivariate_tricycle_codes.py` - BT (Bivariate Tricycle) code construction  
- `trivariate_tricycle_codes.py` - TT (Trivariate Tricycle) code construction

### Legacy Runners (BB codes only)
- `simulation_serial.py` - Single-threaded BB simulations
- `simulation_multiprocess.py` - Multi-process BB simulations

## Usage

### Install
```bash
pip install numpy scipy ldpc bposd stim networkx matplotlib pytest mypy
```

### Generic Framework (Recommended)
The new generic framework supports all code types through a unified API.

Create a JSON config file (see `config_examples/`):
```json
{
  "description": "Small BB code test",
  "code_type": "BB",
  "a_poly": [[3, 0], [0, 1], [0, 2]],
  "b_poly": [[0, 3], [1, 0], [2, 0]],
  "l": 6, "m": 6,
  "p_range": {"min": 0.001, "max": 0.007, "num_points": 3},
  "rounds_list": [6, 8],
  "max_shots": 1000,
  "max_errors": 10
}
```

**BT (Bivariate Tricycle) Code Example:**
```json
{
  "code_type": "BT",
  "a_poly": [[3, 0], [0, 1], [0, 2]], 
  "b_poly": [[0, 3], [1, 0], [2, 0]],
  "c_poly": [[1, 1]],
  "l": 3, "m": 3,
  "p_range": {"min": 0.001, "max": 0.01, "num_points": 3},
  "rounds_list": [4, 6]
}
```

**TT (Trivariate Tricycle) Code Example:**
```json
{
  "code_type": "TT",
  "a_poly": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "b_poly": [[2, 0, 0], [0, 2, 0], [0, 0, 2]], 
  "c_poly": [[1, 1, 0], [0, 1, 1]],
  "l": 2, "m": 2, "n": 3,
  "p_range": {"min": 0.001, "max": 0.01, "num_points": 3}
}
```

Run simulations:
```bash
# Generic framework - serial simulation (any code type)
python simulation_generic.py --config config_examples/bb_small_test.json --output-dir results

# Generic framework - multiprocess simulation (any code type)
python simulation_generic.py --config config_examples/bt_small_test.json --output-dir results --multiprocess

# Legacy runners (BB codes only)
python simulation_serial.py --config config_examples/bb_small_test.json --output-dir results
python simulation_multiprocess.py --config config_examples/bb_small_test.json --output-dir results
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

## Code structure

### Bivariate Bicycle Code

A, B, C are polynomial of x, y

- X Check: $H_x = [A, B]$
- Z Check: $H_Z = [B^T, A^T]$

### Bivariate Tricycle Code

A, B, C are polynomial of x, y

- X Check: $H_x = [A, B, C]$
- Z Check: $H_Z = [[C^T, 0, A^T], [0, C^T, B^T], [B^T, A^T, 0]]$
- meta Z check: $H_{meta} = [B^T, A^T, C^T]$


### Trivariate Tricycle Code

A, B, C are polynomial of x, y, z

- X Check: $H_x = [A, B, C]$
- Z Check: $H_Z = [[C^T, 0, A^T], [0, C^T, B^T], [B^T, A^T, 0]]$
- meta Z check: $H_{meta} = [B^T, A^T, C^T]$

## References

- [ZSZ-codes-numerics](https://github.com/yifanhong/ZSZ-codes-numerics)
- [Stim](https://github.com/quantumlib/Stim) - Quantum circuit simulation
- [LDPC](https://github.com/quantumgizmos/ldpc) - BP+OSD decoding
- Bivariate Bicycle codes: [arXiv:2308.07915](https://arxiv.org/abs/2308.07915)
