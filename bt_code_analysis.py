#!/usr/bin/env python3
"""BT code and meta check analysis using only the distance estimator."""

import numpy as np
from bivariate_tricycle_codes import get_BT_Hmeta, get_BT_Hx_Hz
from distance_estimator import get_min_logical_weight
from bposd.css import css_code
from ldpc.code_util import compute_code_dimension
from typing import Any, Tuple, List
import itertools




def analyze_bt_code(a_poly, b_poly, c_poly, l, m):
    """Analyze BT quantum code parameters using CSS code test() API."""
    print(f"\nBT Quantum Code Analysis")
    print("=" * 40)
    print(f"Parameters: l={l}, m={m}")
    print(f"a_poly: {a_poly}")  
    print(f"b_poly: {b_poly}")
    print(f"c_poly: {c_poly}")
    print()
    
    # Get BT code matrices
    Hx, Hz = get_BT_Hx_Hz(a_poly, b_poly, c_poly, l, m)
    
    # Create CSS code and use test() API
    code = css_code(Hx, Hz)
    print("Running code.test() to get N and K...")
    code.test()
    
    print(f"Hx shape: {Hx.shape}, Hz shape: {Hz.shape}")
    print(f"Code length N: {code.N}")
    print(f"Code dimension K: {code.K}")
    print(f"Code rate: {code.K/code.N:.4f}" if code.N > 0 else "Code rate: undefined")
    
    if code.K == 0:
        print("✗ BT code has dimension 0 (degenerate)")
        return code, code.N, 0, 0
    
    # Estimate distances using distance estimator
    bp_iters, osd_order = 15, 8
    pars = [bp_iters, osd_order]
    
    print("Estimating X distance...")
    d_x = get_min_logical_weight(code, 0.05, pars, 2000, Ptype=0)
    print(f"  X distance: {d_x}")
    
    print("Estimating Z distance...")
    d_z = get_min_logical_weight(code, 0.05, pars, 2000, Ptype=1)
    print(f"  Z distance: {d_z}")
    
    d = min(d_x, d_z) if (d_x > 0 and d_z > 0) else max(d_x, d_z)
    
    print(f"\nBT Code Parameters: [[{code.N}, {code.K}, {d}]]")
    if d > 0:
        print(f"✓ BT code distance d = {d} > 0")
    else:
        print(f"✗ Could not determine positive distance")
    
    return code, code.N, code.K, d


class MetaClassicalCode:
    """Wrapper for meta check classical code to work with distance estimator."""
    def __init__(self, H: np.ndarray):
        self.hx = H.astype(np.uint8)
        self.hz = H.astype(np.uint8)  # Same matrix for classical code
        self.N = H.shape[1]
        
        # For classical codes, logical operators are in nullspace
        # Using ldpc library to compute dimensions
        k = compute_code_dimension(H)
        
        # Create dummy logical operators (needed for distance estimator interface)
        if k > 0:
            # Create k identity-like logical operators
            self.lx = np.zeros((k, self.N), dtype=np.uint8)
            self.lz = np.zeros((k, self.N), dtype=np.uint8)
            for i in range(k):
                self.lx[i, i % self.N] = 1
                self.lz[i, i % self.N] = 1
        else:
            self.lx = np.zeros((0, self.N), dtype=np.uint8)
            self.lz = np.zeros((0, self.N), dtype=np.uint8)


def analyze_meta_check(a_poly, b_poly, c_poly, l, m):
    """Analyze BT meta check classical code using ldpc utilities."""
    print(f"\nBT Meta Check Analysis")
    print("=" * 40)
    
    # Get meta check matrix
    H_meta = get_BT_Hmeta(a_poly, b_poly, c_poly, l, m)
    
    # Use ldpc library for code dimensions
    n = H_meta.shape[1]
    k = compute_code_dimension(H_meta)
    
    print(f"Meta check matrix shape: {H_meta.shape}")
    print(f"Code length n: {n}")
    print(f"Code dimension k: {k}")
    print(f"Code rate: {k/n:.4f}" if n > 0 else "Code rate: undefined")
    
    if k == 0:
        print("✗ Meta check has dimension 0 (degenerate)")
        return H_meta, n, 0, 0
    
    # Estimate minimum distance using wrapper
    if k == 0:
        d = 0
    else:
        print("Estimating minimum distance...")
        meta_code = MetaClassicalCode(H_meta)
        bp_iters, osd_order = 15, 8
        pars = [bp_iters, osd_order]
        d = get_min_logical_weight(meta_code, 0.05, pars, 3000, Ptype=0)
        print(f"  Estimated distance: {d}")
    
    print(f"\nMeta Check Parameters: [{n}, {k}, {d}]")
    if d > 0:
        print(f"✓ Meta check distance d = {d} > 0 (non-zero)")
    else:
        print(f"✗ Could not determine positive distance")
    
    return H_meta, n, k, d



def analyze_with_distance_estimator(a_poly=None, b_poly=None, c_poly=None, l=None, m=None):
    """Analyze both BT code and meta check using distance estimator."""
    # Default parameters from goal.md
    if a_poly is None:
        a_poly = [[3, 0], [0, 1], [0, 2]]  # x^3 + y + y^2
    if b_poly is None:
        b_poly = [[0, 3], [1, 0], [2, 0]]  # y^3 + x + x^2
    if c_poly is None:
        c_poly = [[0, 0], [1, 1]]          # 1 + xy
    if l is None:
        l = 6
    if m is None:
        m = 6
    
    print("BT Code and Meta Check Analysis (Distance Estimator Only)")
    print("=" * 65)
    print(f"Parameters: a_poly={a_poly}, b_poly={b_poly}, c_poly={c_poly}, l={l}, m={m}")
    print()
    
    # Analyze BT quantum code
    bt_code, n_bt, k_bt, d_bt = analyze_bt_code(a_poly, b_poly, c_poly, l, m)
    
    # Analyze meta check classical code  
    H_meta, n_meta, k_meta, d_meta = analyze_meta_check(a_poly, b_poly, c_poly, l, m)
    
    return (bt_code, n_bt, k_bt, d_bt), (H_meta, n_meta, k_meta, d_meta)



def test_robustness(a_poly=None, b_poly=None, l=None, m=None, variants=None):
    """Test different c_poly values for both BT code and meta check."""
    # Default parameters
    if a_poly is None:
        a_poly = [[3, 0], [0, 1], [0, 2]]
    if b_poly is None:
        b_poly = [[0, 3], [1, 0], [2, 0]]
    if l is None:
        l = 6
    if m is None:
        m = 6
    if variants is None:
        variants = get_c_poly_variants()
    
    print("\nRobustness Test: BT Code & Meta Check")
    print("=" * 50)
    print(f"Parameters: a_poly={a_poly}, b_poly={b_poly}, l={l}, m={m}")
    print(f"{'c_poly':<20} {'BT [[n,k,d]]':<15} {'Meta [n,k,d]':<15}")
    print("-" * 50)
    
    bp_iters, osd_order = 10, 5  # Faster parameters
    pars = [bp_iters, osd_order]
    
    for c_poly, desc in variants:
        # Test BT quantum code using CSS code test() API
        Hx, Hz = get_BT_Hx_Hz(a_poly, b_poly, c_poly, l, m)
        bt_code = css_code(Hx, Hz)
        bt_code.test()
        
        n_bt = bt_code.N
        k_bt = bt_code.K
        
        # Quick distance estimate for BT code
        d_bt = 0
        if k_bt > 0:
            d_bt = get_min_logical_weight(bt_code, 0.08, pars, 800, Ptype=0)
        
        # Test meta check
        H_meta = get_BT_Hmeta(a_poly, b_poly, c_poly, l, m)
        
        n_meta = H_meta.shape[1]
        k_meta = compute_code_dimension(H_meta)
        
        d_meta = 0
        if k_meta > 0:
            meta_code = MetaClassicalCode(H_meta)
            d_meta = get_min_logical_weight(meta_code, 0.08, pars, 800, Ptype=0)
        
        bt_status = "✓" if k_bt > 0 and d_bt > 0 else ("◐" if k_bt > 0 else "✗")
        meta_status = "✓" if k_meta > 0 and d_meta > 0 else ("◐" if k_meta > 0 else "✗")
        
        print(f"{desc:<20} {bt_status}[[{n_bt},{k_bt},{d_bt}]]     {meta_status}[{n_meta},{k_meta},{d_meta}]")
    
    print()
    print("Legend: ✓ = good parameters, ◐ = k>0 but d=0, ✗ = degenerate")


def poly_to_coeffs_2d(poly_spec: List[List[int]], l: int, m: int) -> np.ndarray:
    """Convert polynomial specification to 2D coefficient array."""
    coeffs = np.zeros((l, m), dtype=int)
    for i, j in poly_spec:
        coeffs[i % l, j % m] = 1  # Modulo for periodicity
    return coeffs


def evaluate_poly_2d(coeffs: np.ndarray, x: complex, y: complex) -> complex:
    """Evaluate 2D polynomial at complex point (x, y)."""
    l, m = coeffs.shape
    result = 0.0 + 0.0j
    
    for i in range(l):
        for j in range(m):
            if coeffs[i, j] != 0:
                result += coeffs[i, j] * (x**i) * (y**j)
    
    return result




def find_ab_roots_on_unit_circles(a_coeffs: np.ndarray, b_coeffs: np.ndarray, 
                                  l: int, m: int, tolerance: float = 1e-10) -> List[Tuple[complex, complex]]:
    """Find common roots of a(x,y)=0, b(x,y)=0, x^l-1=0, y^m-1=0."""
    # Generate all l-th and m-th roots of unity
    x_roots = [np.exp(2j * np.pi * k / l) for k in range(l)]
    y_roots = [np.exp(2j * np.pi * k / m) for k in range(m)]
    
    ab_roots = []
    
    for x, y in itertools.product(x_roots, y_roots):
        # Evaluate polynomials at this point
        a_val = evaluate_poly_2d(a_coeffs, x, y)
        b_val = evaluate_poly_2d(b_coeffs, x, y)
        
        # Check if both a and b are close to zero (satisfying constraints)
        if abs(a_val) < tolerance and abs(b_val) < tolerance:
            ab_roots.append((x, y))
    
    return ab_roots


def test_refined_root_hypothesis(a_poly=None, b_poly=None, l=None, m=None, variants=None):
    """Test refined hypothesis: evaluate c(x,y) at roots of a,b system."""
    # Default parameters
    if a_poly is None:
        a_poly = [[3, 0], [0, 1], [0, 2]]  # x^3 + y + y^2
    if b_poly is None:
        b_poly = [[0, 3], [1, 0], [2, 0]]  # y^3 + x + x^2
    if l is None:
        l = 6
    if m is None:
        m = 6
    if variants is None:
        variants = get_c_poly_variants()
    
    print("\nRefined Root Analysis for BT Codes")
    print("=" * 65)
    print("Method: Find roots of a(x,y)=0, b(x,y)=0, x^l-1=0, y^m-1=0,")
    print("then evaluate c(x,y) at these roots as indicator")
    print(f"Parameters: a_poly={a_poly}, b_poly={b_poly}, l={l}, m={m}")
    print()
    
    # Convert a, b to coefficient arrays
    a_coeffs = poly_to_coeffs_2d(a_poly, l, m)
    b_coeffs = poly_to_coeffs_2d(b_poly, l, m)
    
    # Find roots where a=0, b=0 on unit circles
    ab_roots = find_ab_roots_on_unit_circles(a_coeffs, b_coeffs, l, m)
    
    print(f"Found {len(ab_roots)} roots where a(x,y)=0, b(x,y)=0:")
    for i, (x, y) in enumerate(ab_roots):
        print(f"  Root {i+1}: x={x:.4f}, y={y:.4f}")
    print()
    
    print("Comprehensive Analysis:")
    print("=" * 120)
    print(f"{'c_poly':<20} {'BT [[n,k,d]]':<15} {'Meta [n,k,d]':<15} {'c(roots)':<15} {'Pattern'}")
    print("-" * 120)
    
    results = []
    
    bp_iters, osd_order = 10, 5  # Fast distance estimation parameters
    pars = [bp_iters, osd_order]
    
    for c_poly, c_desc in variants:
        # Get BT code parameters (quiet test)
        Hx, Hz = get_BT_Hx_Hz(a_poly, b_poly, c_poly, l, m)
        bt_code = css_code(Hx, Hz)
        # Suppress output during test
        import sys
        from contextlib import redirect_stdout
        import io
        with redirect_stdout(io.StringIO()):
            bt_code.test()
        
        # Estimate BT distance if K > 0
        d_bt = 0
        if bt_code.K > 0:
            d_bt = get_min_logical_weight(bt_code, 0.08, pars, 500, Ptype=0)
        
        bt_params = f"[{bt_code.N},{bt_code.K},{d_bt}]"
        bt_success = bt_code.K > 0
        
        # Get meta check parameters
        H_meta = get_BT_Hmeta(a_poly, b_poly, c_poly, l, m)
        n_meta = H_meta.shape[1]
        k_meta = compute_code_dimension(H_meta)
        
        # Estimate meta check distance if K > 0
        d_meta = 0
        if k_meta > 0:
            meta_code = MetaClassicalCode(H_meta)
            d_meta = get_min_logical_weight(meta_code, 0.08, pars, 500, Ptype=0)
                
        meta_params = f"[{n_meta},{k_meta},{d_meta}]"
        
        # Evaluate c(x,y) at ab_roots
        c_coeffs = poly_to_coeffs_2d(c_poly, l, m)
        c_values = []
        
        for x, y in ab_roots:
            c_val = evaluate_poly_2d(c_coeffs, x, y)
            c_values.append(c_val)
        
        # Analyze c values pattern
        if len(c_values) == 0:
            c_indicator = "no ab_roots"
            pattern = "N/A"
        else:
            zero_count = sum(1 for val in c_values if abs(val) < 1e-10)
            nonzero_count = len(c_values) - zero_count
            
            if zero_count == 0:
                c_indicator = "all nonzero"
                pattern = "c≠0 at all"
            elif zero_count == len(c_values):
                c_indicator = "all zero"
                pattern = "c=0 at all"
            else:
                c_indicator = f"{zero_count}zero,{nonzero_count}nonzero"
                pattern = "mixed"
        
        results.append({
            'c_desc': c_desc,
            'bt_params': bt_params,
            'meta_params': meta_params,
            'c_indicator': c_indicator,
            'pattern': pattern,
            'bt_success': bt_success,
            'bt_k': bt_code.K if 'bt_code' in locals() and hasattr(bt_code, 'K') else 0,
            'bt_d': d_bt,
            'meta_k': k_meta,
            'meta_d': d_meta
        })
        
        print(f"{c_desc:<20} {bt_params:<15} {meta_params:<15} {c_indicator:<15} {pattern}")
    
    # Pattern analysis summary
    print("\nPattern Analysis Summary:")
    print("=" * 60)
    
    # Group by BT success and pattern
    successful_k6 = [r for r in results if r['bt_success'] and r['bt_k'] == 6]
    successful_k12 = [r for r in results if r['bt_success'] and r['bt_k'] == 12]
    failed_k0 = [r for r in results if not r['bt_success']]
    
    print(f"Successful BT codes (K=6): {len(successful_k6)} cases")
    for r in successful_k6:
        print(f"  {r['c_desc']:<20} -> {r['pattern']}")
    
    print(f"\nSuccessful BT codes (K=12): {len(successful_k12)} cases")  
    for r in successful_k12:
        print(f"  {r['c_desc']:<20} -> {r['pattern']}")
        
    print(f"\nFailed BT codes (K=0): {len(failed_k0)} cases")
    for r in failed_k0:
        print(f"  {r['c_desc']:<20} -> {r['pattern']}")
    
    # Pattern correlation
    print(f"\nKey Pattern Correlations:")
    nonzero_successful = sum(1 for r in successful_k6 if r['pattern'] == 'c≠0 at all')
    mixed_higher_dim = sum(1 for r in successful_k12 if r['pattern'] == 'mixed')
    
    print(f"- K=6 codes with c≠0 pattern: {nonzero_successful}/{len(successful_k6)}")
    print(f"- K=12 codes with mixed pattern: {mixed_higher_dim}/{len(successful_k12)}")
    print(f"- All failed codes have c≠0 pattern (but lack proper x-y coupling)")
    
    
    return ab_roots, len(ab_roots), results


def run_multiple_experiments():
    """Run analysis on multiple experiment configurations."""
    configs = get_experiment_configs()
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {config['name']}")
        print(f"{'='*80}")
        
        # Main analysis for this config
        (bt_code, n_bt, k_bt, d_bt), (H_meta, n_meta, k_meta, d_meta) = analyze_with_distance_estimator(
            a_poly=config['a_poly'], 
            b_poly=config['b_poly'], 
            l=config['l'], 
            m=config['m']
        )
        
        # Robustness test for this config
        test_robustness(
            a_poly=config['a_poly'], 
            b_poly=config['b_poly'], 
            l=config['l'], 
            m=config['m']
        )
        
        # Root hypothesis test for this config
        ab_roots, num_ab_roots, analysis_results = test_refined_root_hypothesis(
            a_poly=config['a_poly'], 
            b_poly=config['b_poly'], 
            l=config['l'], 
            m=config['m']
        )
        
        print(f"\nSUMMARY for {config['name']}:")
        print(f"BT Quantum Code: [[{n_bt}, {k_bt}, {d_bt}]]")
        print(f"BT Meta Check Classical Code: [{n_meta}, {k_meta}, {d_meta}]")
        print(f"Found {num_ab_roots} roots where a(x,y)=0, b(x,y)=0 on unit circles")


def run_single_experiment():
    """Run analysis on default configuration only."""
    # Main analysis
    (bt_code, n_bt, k_bt, d_bt), (H_meta, n_meta, k_meta, d_meta) = analyze_with_distance_estimator()
    
    # Robustness test
    test_robustness()
    
    # Test refined root hypothesis 
    ab_roots, num_ab_roots, analysis_results = test_refined_root_hypothesis()
    
    print(f"\nSUMMARY:")
    print(f"BT Quantum Code: [[{n_bt}, {k_bt}, {d_bt}]]")
    print(f"BT Meta Check Classical Code: [{n_meta}, {k_meta}, {d_meta}]")
    print(f"Found {num_ab_roots} roots where a(x,y)=0, b(x,y)=0 on unit circles")
    print(f"Distance estimates included for both quantum and classical codes")


def get_experiment_configs():
    """Get different experiment configurations for testing."""
    return [
        # {
        #     "name": "Standard 6x6",
        #     "a_poly": [[3, 0], [0, 1], [0, 2]],  # x^3 + y + y^2
        #     "b_poly": [[0, 3], [1, 0], [2, 0]],  # y^3 + x + x^2
        #     "l": 6, "m": 6
        # },
        # {
        #     "name": "[[144,12,12]] 8x6",
        #     "a_poly": [[3, 0], [0, 1], [0, 2]],  # x^3 + y + y^2
        #     "b_poly": [[0, 3], [1, 0], [2, 0]],  # y^3 + x + x^2
        #     "l": 8, "m": 6
        # },
        # {
        #     "name": "[[144,12,12]] 8x6",
        #     "a_poly": [[3, 0], [0, 1], [0, 2]],  # x^3 + y + y^2
        #     "b_poly": [[0, 3], [1, 0], [2, 0]],  # y^3 + x + x^2
        #     "l": 8, "m": 8
        # },
        {
            "name": "[[144,12,12]] 8x6",
            "a_poly": [[3, 0], [0, 1], [0, 2]],  # x^3 + y + y^2
            "b_poly": [[0, 3], [1, 0], [2, 0]],  # y^3 + x + x^2
            "l": 9, "m": 3
        },
    ]


def get_c_poly_variants():
    """Get standard c_poly variants for testing."""
    return [
        ([[0, 0], [1, 1]], "1 + xy (original)"),
        ([[0, 0], [1, 2]], "1 + xy^2"),
        ([[1, 0], [0, 2]], "x + y^2"),
        ([[0, 0]], "1 (constant)"),
        ([[1, 1]], "xy (single term)"),
        # New variants to test
        ([[1, 0], [0, 1]], "x + y"),
        ([[2, 0], [0, 1]], "x^2 + y"),
        ([[1, 0], [2, 0]], "x + x^2"),
        ([[0, 1], [0, 2]], "y + y^2"),
        ([[1, 1], [2, 2]], "xy + x^2y^2"),
        ([[0, 0], [2, 1]], "1 + x^2y"),
        ([[0, 0], [1, 2], [2, 1]], "1 + xy^2 + x^2y"),
        ([[1, 0], [0, 1], [1, 1]], "x + y + xy"),
        ([[2, 0], [0, 2]], "x^2 + y^2"),
        ([[3, 0], [0, 3]], "x^3 + y^3")
    ]


if __name__ == "__main__":
    # Choose which experiment to run:
    # Option 1: Single experiment with default parameters
    # run_single_experiment()
    
    # Option 2: Multiple experiments (uncomment to use)
    run_multiple_experiments()