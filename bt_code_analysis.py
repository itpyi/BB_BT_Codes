#!/usr/bin/env python3
"""BT code and meta check analysis using only the distance estimator."""

import numpy as np
from bivariate_tricycle_codes import get_BT_Hmeta, get_BT_Hx_Hz
from distance_estimator import get_min_logical_weight
from bposd.css import css_code
from ldpc.code_util import compute_code_dimension
from solve_f2root_multi_bc import solve_common_roots_multi_over_F2_with_BC, all_c_variants_over_F2
from std_monomial_basis_f2 import standard_monomial_basis_f2_simple
from bivariate_bicycle_codes import get_BB_Hx_Hz
from typing import Any, Tuple, List, Dict
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
    
    # Compute standard monomial basis for BT code polynomials
    print("\nComputing standard monomial basis...")
    std_basis_result = compute_bt_std_monomial_basis(a_poly, b_poly, c_poly, l, m)
    if "Error" not in std_basis_result:
        print(f"Standard monomial basis dimension: {std_basis_result['Dimension']}")
        print(f"Standard monomial basis length: {len(std_basis_result['StandardMonomials'])}")
    else:
        print(f"Standard monomial basis error: {std_basis_result['Error']}")
    
    if code.K == 0:
        print("✗ BT code has dimension 0 (degenerate)")
        return code, code.N, 0, 0, std_basis_result
    
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
    
    return code, code.N, code.K, d, std_basis_result


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


def compute_bt_std_monomial_basis(a_poly, b_poly, c_poly, l, m):
    """Compute standard monomial basis for BT code polynomials."""
    import sympy as sp
    
    def format_poly_terms(poly_terms):
        """Convert [[i,j], ...] format to sympy expression."""
        if not poly_terms:
            return 0
        x, y = sp.symbols('x y')
        expr = 0
        for i, j in poly_terms:
            expr += x**i * y**j
        return expr
    
    # Convert polynomials to sympy expressions
    x, y = sp.symbols('x y')
    a_expr = format_poly_terms(a_poly)
    b_expr = format_poly_terms(b_poly) 
    c_expr = format_poly_terms(c_poly)
    
    # Create generator list (a, b, c polynomials)
    gens = [a_expr, b_expr, c_expr]
    vars_symbols = [x, y]
    
    # Add periodic constraints
    periods = {x: l, y: m}
    
    try:
        result = standard_monomial_basis_f2_simple(gens, vars_symbols, periods)
        return result
    except Exception as e:
        return {"Error": f"Standard monomial basis computation failed: {str(e)}"}


def analyze_bb_code(a_poly, b_poly, l, m):
    """Analyze BB quantum code parameters using CSS code test() API."""
    print(f"\nBB Quantum Code Analysis")
    print("=" * 40)
    print(f"Parameters: l={l}, m={m}")
    print(f"a_poly: {a_poly}")  
    print(f"b_poly: {b_poly}")
    print()
    
    # Get BB code matrices
    Hx, Hz = get_BB_Hx_Hz(a_poly, b_poly, l, m)
    
    # Create CSS code and use test() API
    code = css_code(Hx, Hz)
    print("Running code.test() to get N and K...")
    code.test()
    
    print(f"Hx shape: {Hx.shape}, Hz shape: {Hz.shape}")
    print(f"Code length N: {code.N}")
    print(f"Code dimension K: {code.K}")
    print(f"Code rate: {code.K/code.N:.4f}" if code.N > 0 else "Code rate: undefined")
    
    # Compute standard monomial basis for BB code polynomials
    print("\nComputing standard monomial basis...")
    std_basis_result = compute_bb_std_monomial_basis(a_poly, b_poly, l, m)
    if "Error" not in std_basis_result:
        print(f"Standard monomial basis dimension: {std_basis_result['Dimension']}")
        print(f"Standard monomial basis length: {len(std_basis_result['StandardMonomials'])}")
    else:
        print(f"Standard monomial basis error: {std_basis_result['Error']}")
    
    if code.K == 0:
        print("✗ BB code has dimension 0 (degenerate)")
        return code, code.N, 0, 0, std_basis_result
    
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
    
    print(f"\nBB Code Parameters: [[{code.N}, {code.K}, {d}]]")
    if d > 0:
        print(f"✓ BB code distance d = {d} > 0")
    else:
        print(f"✗ Could not determine positive distance")
    
    return code, code.N, code.K, d, std_basis_result


def compute_bb_std_monomial_basis(a_poly, b_poly, l, m):
    """Compute standard monomial basis for BB code polynomials."""
    import sympy as sp
    
    def format_poly_terms(poly_terms):
        """Convert [[i,j], ...] format to sympy expression."""
        if not poly_terms:
            return 0
        x, y = sp.symbols('x y')
        expr = 0
        for i, j in poly_terms:
            expr += x**i * y**j
        return expr
    
    # Convert polynomials to sympy expressions
    x, y = sp.symbols('x y')
    a_expr = format_poly_terms(a_poly)
    b_expr = format_poly_terms(b_poly) 
    
    # Create generator list (a, b polynomials only for BB codes)
    gens = [a_expr, b_expr]
    vars_symbols = [x, y]
    
    # Add periodic constraints
    periods = {x: l, y: m}
    
    try:
        result = standard_monomial_basis_f2_simple(gens, vars_symbols, periods)
        return result
    except Exception as e:
        return {"Error": f"Standard monomial basis computation failed: {str(e)}"}


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
    bt_code, n_bt, k_bt, d_bt, bt_std_basis = analyze_bt_code(a_poly, b_poly, c_poly, l, m)
    
    # Analyze meta check classical code  
    H_meta, n_meta, k_meta, d_meta = analyze_meta_check(a_poly, b_poly, c_poly, l, m)
    
    return (bt_code, n_bt, k_bt, d_bt, bt_std_basis), (H_meta, n_meta, k_meta, d_meta)


def analyze_bb_with_distance_estimator(a_poly=None, b_poly=None, l=None, m=None):
    """Analyze BB code using distance estimator and standard monomial basis."""
    # Default parameters
    if a_poly is None:
        a_poly = [[3, 0], [0, 1], [0, 2]]  # x^3 + y + y^2
    if b_poly is None:
        b_poly = [[0, 3], [1, 0], [2, 0]]  # y^3 + x + x^2
    if l is None:
        l = 6
    if m is None:
        m = 6
    
    print("BB Code Analysis (Distance Estimator + Standard Monomial Basis)")
    print("=" * 70)
    print(f"Parameters: a_poly={a_poly}, b_poly={b_poly}, l={l}, m={m}")
    print()
    
    # Analyze BB quantum code
    bb_code, n_bb, k_bb, d_bb, bb_std_basis = analyze_bb_code(a_poly, b_poly, l, m)
    
    return bb_code, n_bb, k_bb, d_bb, bb_std_basis


def comprehensive_comparison_table(a_poly=None, b_poly=None, l=None, m=None):
    """Generate comprehensive comparison table for BT vs BB codes with standard monomial basis."""
    # Default parameters
    if a_poly is None:
        a_poly = [[3, 0], [0, 1], [0, 2]]  # x^3 + y + y^2
    if b_poly is None:
        b_poly = [[0, 3], [1, 0], [2, 0]]  # y^3 + x + x^2
    if l is None:
        l = 6
    if m is None:
        m = 6
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE BT vs BB CODE COMPARISON WITH STANDARD MONOMIAL BASIS")
    print("=" * 100)
    print(f"Base polynomials: a={a_poly}, b={b_poly}, grid={l}×{m}")
    print()
    
    # Analyze BB code (baseline - only uses a,b)
    print("BASELINE BB CODE (uses a,b polynomials only):")
    print("-" * 50)
    bb_code, n_bb, k_bb, d_bb, bb_std_basis = analyze_bb_code(a_poly, b_poly, l, m)
    bb_smb_dim = bb_std_basis.get('Dimension', 'N/A')
    bb_smb_len = len(bb_std_basis.get('StandardMonomials', [])) if 'StandardMonomials' in bb_std_basis else 'N/A'
    bb_smb_monomials = bb_std_basis.get('StandardMonomials', [])
    bb_smb_str = format_monomial_basis(bb_smb_monomials) if bb_smb_monomials else 'N/A'
    print(f"BB Code: [[{n_bb}, {k_bb}, {d_bb}]] | SMB Dim: {bb_smb_dim} | SMB: {bb_smb_str}")
    print()
    
    # Test BT codes with various c polynomials
    print("BT CODES (uses a,b,c polynomials with different c variants):")
    print("-" * 70)
    variants = get_standard_c_poly_variants()
    
    bt_results = test_robustness(a_poly, b_poly, l, m, variants)
    
    return {'bb_result': (n_bb, k_bb, d_bb, bb_smb_dim, bb_smb_len), 'bt_results': bt_results}



def test_robustness_with_generated_variants(a_poly=None, b_poly=None, l=None, m=None):
    """Test BT codes using c_poly variants generated from F₂ analysis of a,b system."""
    # Default parameters
    if a_poly is None:
        a_poly = [[3, 0], [0, 1], [0, 2]]
    if b_poly is None:
        b_poly = [[0, 3], [1, 0], [2, 0]]
    if l is None:
        l = 6
    if m is None:
        m = 6
    
    print(f"\nAdvanced Robustness Test: F₂-Generated C Variants")
    print("=" * 60)
    print(f"Generating c polynomials from a,b system using Gröbner basis analysis")
    print(f"Parameters: a_poly={a_poly}, b_poly={b_poly}, l={l}, m={m}")
    print()
    
    # Generate c variants from a,b system
    print("Generating c polynomial variants from a,b Gröbner basis...")
    generated_variants = generate_c_poly_variants_from_ab(a_poly, b_poly, l, m)
    
    if generated_variants:
        print(f"✓ Generated {len(generated_variants)} c polynomial variants:")
        for i, (c_poly, desc) in enumerate(generated_variants[:5]):  # Show first 5
            print(f"  {i+1}. {desc}")
        if len(generated_variants) > 5:
            print(f"  ... and {len(generated_variants)-5} more variants")
    else:
        print("✗ No variants generated, using standard set")
        generated_variants = get_standard_c_poly_variants()
        
    print()
    
    # Test the generated variants
    test_robustness(a_poly, b_poly, l, m, variants=generated_variants)


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
    
    print("\nRobustness Test: BT Code & Meta Check with Standard Monomial Basis")
    print("=" * 80)
    print(f"Parameters: a_poly={a_poly}, b_poly={b_poly}, l={l}, m={m}")
    print(f"{'c_poly':<20} {'BT [[n,k,d]]':<15} {'Meta [n,k,d]':<15} {'SMB':<8} {'Standard Monomial Basis':<30}")
    print("-" * 100)
    
    bp_iters, osd_order = 10, 5  # Faster parameters
    pars = [bp_iters, osd_order]
    
    summary_results = []
    
    for c_poly, desc in variants:
        # Test BT quantum code using CSS code test() API
        Hx, Hz = get_BT_Hx_Hz(a_poly, b_poly, c_poly, l, m)
        bt_code = css_code(Hx, Hz)
        # Suppress output during test
        import sys
        from contextlib import redirect_stdout
        import io
        with redirect_stdout(io.StringIO()):
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
        
        # Compute standard monomial basis for this c_poly variant
        std_basis_result = compute_bt_std_monomial_basis(a_poly, b_poly, c_poly, l, m)
        smb_dim = std_basis_result.get('Dimension', 'N/A')
        smb_len = len(std_basis_result.get('StandardMonomials', [])) if 'StandardMonomials' in std_basis_result else 'N/A'
        smb_monomials = std_basis_result.get('StandardMonomials', [])
        smb_str = format_monomial_basis(smb_monomials) if smb_monomials else 'N/A'
        
        bt_status = "✓" if k_bt > 0 and d_bt > 0 else ("◐" if k_bt > 0 else "✗")
        meta_status = "✓" if k_meta > 0 and d_meta > 0 else ("◐" if k_meta > 0 else "✗")
        
        print(f"{desc:<20} {bt_status}[[{n_bt},{k_bt},{d_bt}]]     {meta_status}[{n_meta},{k_meta},{d_meta}]     {smb_dim:<8} {smb_str:<30}")
        
        # Store results for summary table
        summary_results.append({
            'c_desc': desc,
            'bt_params': f"[[{n_bt},{k_bt},{d_bt}]]",
            'meta_params': f"[{n_meta},{k_meta},{d_meta}]",
            'smb_dim': smb_dim,
            'smb_len': smb_len,
            'smb_monomials': smb_monomials,
            'smb_str': smb_str,
            'bt_success': k_bt > 0 and d_bt > 0,
            'meta_success': k_meta > 0 and d_meta > 0,
            'bt_k': k_bt,
            'bt_d': d_bt
        })
    
    print()
    print("Legend: ✓ = good parameters, ◐ = k>0 but d=0, ✗ = degenerate")
    print("SMB = Standard Monomial Basis")
    
    # Generate comprehensive summary table
    generate_comprehensive_summary_table(summary_results, a_poly, b_poly, l, m)
    
    return summary_results


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




def find_ab_roots_over_f2(a_poly: List[List[int]], b_poly: List[List[int]], 
                         l: int, m: int) -> Dict[str, Any]:
    """Find common roots of a(x,y)=0, b(x,y)=0, x^l-1=0, y^m-1=0 over F₂ algebraic closure.
    
    Uses exact F₂ algebraic methods instead of complex approximations.
    """
    def format_poly_terms(poly_terms):
        """Convert [[i,j], ...] format to polynomial string."""
        if not poly_terms:
            return "0"
        terms = []
        for i, j in poly_terms:
            if i == 0 and j == 0:
                terms.append("1")
            elif i == 0:
                terms.append(f"y^{j}" if j > 1 else "y")
            elif j == 0:
                terms.append(f"x^{i}" if i > 1 else "x")
            else:
                x_part = f"x^{i}" if i > 1 else "x"
                y_part = f"y^{j}" if j > 1 else "y"
                terms.append(f"{x_part}*{y_part}")
        return " + ".join(terms)
    
    a_poly_str = format_poly_terms(a_poly)
    b_poly_str = format_poly_terms(b_poly)
    
    try:
        # Solve the system a(x,y)=0, b(x,y)=0 with boundary conditions
        result = solve_common_roots_multi_over_F2_with_BC(
            poly_strs=[a_poly_str, b_poly_str],
            var_names=["x", "y"],
            m=l,  # x^l - 1 = 0
            l=m,  # y^m - 1 = 0  
            order="lex"
        )
        
        return {
            'f2_analysis_successful': True,
            'has_common_root': result.has_common_root_with_bc,
            'groebner_basis_with_bc': result.groebner_basis_with_bc,
            'groebner_basis_no_bc': result.groebner_basis_no_bc,
            'univariate_eliminants': result.univariate_eliminants_no_bc,
            'univariate_factors': result.univariate_factors_no_bc,
            'triangular_assignments': result.triangular_assignments,
            'gcd_diagnostics': {
                'gcd_fx_xm1': result.gcd_fx_xm1,
                'gcd_fy_yl1': result.gcd_fy_yl1
            },
            'a_polynomial': a_poly_str,
            'b_polynomial': b_poly_str,
            'boundary_conditions': result.bc_polys
        }
        
    except Exception as e:
        return {
            'f2_analysis_successful': False,
            'error': str(e),
            'a_polynomial': a_poly_str,
            'b_polynomial': b_poly_str
        }


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
    
    # Analyze a, b system using F₂ algebraic methods
    ab_f2_analysis = find_ab_roots_over_f2(a_poly, b_poly, l, m)
    
    if ab_f2_analysis['f2_analysis_successful']:
        has_ab_roots = ab_f2_analysis['has_common_root']
        print(f"F₂ analysis of a(x,y)=0, b(x,y)=0 system:")
        print(f"  a(x,y) = {ab_f2_analysis['a_polynomial']}")
        print(f"  b(x,y) = {ab_f2_analysis['b_polynomial']}")
        print(f"  Has common roots over F₂: {has_ab_roots}")
        if ab_f2_analysis.get('triangular_assignments'):
            print(f"  Triangular relations: {ab_f2_analysis['triangular_assignments']}")
        if ab_f2_analysis.get('gcd_diagnostics'):
            gcd = ab_f2_analysis['gcd_diagnostics']
            if gcd.get('gcd_fx_xm1'):
                print(f"  GCD(f_x, x^{l}+1) = {gcd['gcd_fx_xm1']}")
            if gcd.get('gcd_fy_yl1'):
                print(f"  GCD(f_y, y^{m}+1) = {gcd['gcd_fy_yl1']}")
    else:
        has_ab_roots = False
        print(f"F₂ analysis failed: {ab_f2_analysis.get('error', 'unknown error')}")
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
        
        # Analyze c(x,y) constraint in the context of a,b system using F₂ methods
        if has_ab_roots:
            # If a,b system has roots, check if c(x,y) constraint is compatible
            try:
                # Analyze the full system: a(x,y)=0, b(x,y)=0, c(x,y)=0
                def format_poly_terms(poly_terms):
                    if not poly_terms:
                        return "0"
                    terms = []
                    for i, j in poly_terms:
                        if i == 0 and j == 0:
                            terms.append("1")
                        elif i == 0:
                            terms.append(f"y^{j}" if j > 1 else "y")
                        elif j == 0:
                            terms.append(f"x^{i}" if i > 1 else "x")
                        else:
                            x_part = f"x^{i}" if i > 1 else "x"
                            y_part = f"y^{j}" if j > 1 else "y"
                            terms.append(f"{x_part}*{y_part}")
                    return " + ".join(terms)
                
                c_poly_str = format_poly_terms(c_poly)
                a_poly_str = format_poly_terms(a_poly)
                b_poly_str = format_poly_terms(b_poly)
                
                # Check full system a=0, b=0, c=0
                full_system = solve_common_roots_multi_over_F2_with_BC(
                    poly_strs=[a_poly_str, b_poly_str, c_poly_str],
                    var_names=["x", "y"],
                    m=l, l=m, order="lex"
                )
                
                if full_system.has_common_root_with_bc:
                    c_indicator = "compatible"
                    pattern = "c compatible with ab"
                else:
                    c_indicator = "incompatible" 
                    pattern = "c blocks ab roots"
                    
            except Exception:
                c_indicator = "analysis_failed"
                pattern = "F₂ analysis error"
        else:
            c_indicator = "no_ab_roots"
            pattern = "no ab common roots"
        
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
    
    # Pattern correlation with F₂ analysis
    print(f"\nKey Pattern Correlations (F₂ Analysis):")
    compatible_successful = sum(1 for r in successful_k6 if 'compatible' in r['pattern'])
    incompatible_higher_dim = sum(1 for r in successful_k12 if 'blocks' in r['pattern'])
    
    print(f"- K=6 codes with c compatible pattern: {compatible_successful}/{len(successful_k6)}")  
    print(f"- K=12 codes with c incompatible pattern: {incompatible_higher_dim}/{len(successful_k12)}")
    print(f"- F₂ algebraic analysis provides exact root relationships over finite fields")
    
    
    return ab_f2_analysis, has_ab_roots, results


def run_multiple_experiments():
    """Run analysis on multiple experiment configurations."""
    configs = get_experiment_configs()
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {config['name']}")
        print(f"{'='*80}")
        
        # Main analysis for this config
        (bt_code, n_bt, k_bt, d_bt, bt_std_basis), (H_meta, n_meta, k_meta, d_meta) = analyze_with_distance_estimator(
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
        
        print(f"\nSUMMARY for {config['name']}:")
        print(f"BT Quantum Code: [[{n_bt}, {k_bt}, {d_bt}]]")
        print(f"BT Meta Check Classical Code: [{n_meta}, {k_meta}, {d_meta}]")


def run_single_experiment():
    """Run analysis on default configuration only."""
    # Main analysis
    (bt_code, n_bt, k_bt, d_bt, bt_std_basis), (H_meta, n_meta, k_meta, d_meta) = analyze_with_distance_estimator()
    
    # Robustness test
    test_robustness()
    
    print(f"\nSUMMARY:")
    print(f"BT Quantum Code: [[{n_bt}, {k_bt}, {d_bt}]]")
    print(f"BT Meta Check Classical Code: [{n_meta}, {k_meta}, {d_meta}]")
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
            "l": 6, "m": 12
        },
    ]


def generate_c_poly_variants_from_ab(a_poly: List[List[int]], b_poly: List[List[int]], 
                                    l: int, m: int) -> List[Tuple[List[List[int]], str]]:
    """Generate c polynomial variants using F₂ Gröbner basis analysis of a,b system.
    
    Args:
        a_poly: A polynomial as [[i,j], ...] format
        b_poly: B polynomial as [[i,j], ...] format  
        l, m: Grid dimensions
        
    Returns:
        List of (c_poly_terms, description) tuples
    """
    def format_poly_terms_to_string(poly_terms):
        """Convert [[i,j], ...] format to string."""
        if not poly_terms:
            return "0"
        terms = []
        for i, j in poly_terms:
            if i == 0 and j == 0:
                terms.append("1")
            elif i == 0:
                terms.append(f"y^{j}" if j > 1 else "y")
            elif j == 0:
                terms.append(f"x^{i}" if i > 1 else "x")
            else:
                x_part = f"x^{i}" if i > 1 else "x"
                y_part = f"y^{j}" if j > 1 else "y"
                terms.append(f"{x_part}{y_part}")
        return " + ".join(terms)
    
    def parse_sympy_to_poly_terms(sympy_expr):
        """Convert SymPy expression back to [[i,j], ...] format."""
        import sympy as sp
        x, y = sp.symbols('x y')
        
        if sympy_expr == 0:
            return []
            
        # Expand and collect terms
        expanded = sp.expand(sympy_expr)
        terms = []
        
        if hasattr(expanded, 'as_coefficients_dict'):
            for monomial, coeff in expanded.as_coefficients_dict().items():
                if coeff % 2 == 1:  # Only keep terms with odd coefficients (F₂)
                    if monomial == 1:
                        terms.append([0, 0])
                    else:
                        powers = monomial.as_powers_dict() if hasattr(monomial, 'as_powers_dict') else {monomial: 1}
                        x_exp = powers.get(x, 0)
                        y_exp = powers.get(y, 0)
                        terms.append([x_exp, y_exp])
        else:
            # Handle single term case
            if expanded == 1:
                terms.append([0, 0])
            elif expanded == x:
                terms.append([1, 0])
            elif expanded == y:
                terms.append([0, 1])
            # Add more cases as needed
        
        return terms
    
    # Convert to string format for all_c_variants_over_F2
    a_str = format_poly_terms_to_string(a_poly)
    b_str = format_poly_terms_to_string(b_poly)
    
    try:
        # Generate c variants using F₂ Gröbner basis analysis
        c_variants_dict = all_c_variants_over_F2(a_str, b_str, m=l, l=m)
        
        variants = []
        for name, sympy_expr in c_variants_dict.items():
            # Convert back to [[i,j], ...] format
            poly_terms = parse_sympy_to_poly_terms(sympy_expr)
            description = f"{name} ({sympy_expr})"
            variants.append((poly_terms, description))
            
        # Add some standard variants if none were generated
        if not variants:
            print("  No variants generated from Gröbner analysis, using standard set")
            variants = get_standard_c_poly_variants()
            
        return variants
        
    except Exception as e:
        print(f"  Error generating F₂ variants: {e}")
        print("  Falling back to standard variants")
        return get_standard_c_poly_variants()


def get_standard_c_poly_variants():
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

def format_monomial_basis(monomials):
    """Format standard monomial basis for display."""
    if not monomials:
        return "∅"
    
    # Convert sympy expressions to readable strings
    monomial_strs = []
    for mon in monomials:
        mon_str = str(mon)
        # Replace common patterns for better readability
        if mon_str == '1':
            monomial_strs.append('1')
        else:
            # Convert x**i*y**j to x^i*y^j format, handle single powers
            mon_str = mon_str.replace('**1', '').replace('*', '')  # Remove **1 and *
            mon_str = mon_str.replace('**', '^')  # Use ^ for powers
            monomial_strs.append(mon_str)
    
    return '{' + ', '.join(monomial_strs) + '}'


def generate_comprehensive_summary_table(results, a_poly, b_poly, l, m):
    """Generate a comprehensive summary table with standard monomial basis analysis."""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE SUMMARY TABLE WITH STANDARD MONOMIAL BASIS")
    print("=" * 120)
    print(f"Base polynomials: a={a_poly}, b={b_poly}, grid={l}×{m}")
    print()
    
    # Group results by BT code success and dimension
    successful_k6 = [r for r in results if r['bt_success'] and r['bt_k'] == 6]
    successful_k12 = [r for r in results if r['bt_success'] and r['bt_k'] == 12]
    failed_k0 = [r for r in results if not r['bt_success']]
    
    # Show successful K=6 codes
    if successful_k6:
        print(f"SUCCESSFUL BT CODES (K=6): {len(successful_k6)} variants")
        print(f"{'c_poly':<20} {'BT Code':<15} {'Meta Check':<15} {'SMB':<8} {'Standard Monomial Basis':<40}")
        print("-" * 120)
        for r in successful_k6:
            print(f"{r['c_desc']:<20} {r['bt_params']:<15} {r['meta_params']:<15} {r['smb_dim']:<8} {r['smb_str']:<40}")
        print()
    
    # Show successful K=12 codes  
    if successful_k12:
        print(f"SUCCESSFUL BT CODES (K=12): {len(successful_k12)} variants")
        print(f"{'c_poly':<20} {'BT Code':<15} {'Meta Check':<15} {'SMB':<8} {'Standard Monomial Basis':<40}")
        print("-" * 120)
        for r in successful_k12:
            print(f"{r['c_desc']:<20} {r['bt_params']:<15} {r['meta_params']:<15} {r['smb_dim']:<8} {r['smb_str']:<40}")
        print()
    
    # Show failed codes
    if failed_k0:
        print(f"FAILED BT CODES (K=0): {len(failed_k0)} variants")
        print(f"{'c_poly':<20} {'BT Code':<15} {'Meta Check':<15} {'SMB':<8} {'Standard Monomial Basis':<40}")
        print("-" * 120)
        for r in failed_k0:
            print(f"{r['c_desc']:<20} {r['bt_params']:<15} {r['meta_params']:<15} {r['smb_dim']:<8} {r['smb_str']:<40}")
        print()
    
    # Detailed monomial basis analysis
    print("STANDARD MONOMIAL BASIS ANALYSIS:")
    print("-" * 50)
    
    # Show unique monomial bases
    unique_bases = {}
    for r in results:
        if r['smb_str'] != 'N/A' and r['smb_str'] != '∅':
            if r['smb_str'] not in unique_bases:
                unique_bases[r['smb_str']] = []
            unique_bases[r['smb_str']].append((r['c_desc'], r['bt_k']))
    
    if unique_bases:
        print("Unique Standard Monomial Bases:")
        for i, (basis, codes) in enumerate(unique_bases.items(), 1):
            print(f"{i}. {basis}")
            code_info = ', '.join([f"{desc} (K={k})" for desc, k in codes])
            print(f"   Used by: {code_info}")
            print()
    
    # Statistical analysis
    k6_smb_dims = [r['smb_dim'] for r in successful_k6 if isinstance(r['smb_dim'], int)]
    k12_smb_dims = [r['smb_dim'] for r in successful_k12 if isinstance(r['smb_dim'], int)]
    failed_smb_dims = [r['smb_dim'] for r in failed_k0 if isinstance(r['smb_dim'], int)]
    
    if k6_smb_dims:
        print(f"• K=6 codes: SMB dimensions = {sorted(set(k6_smb_dims))} (most common: {max(set(k6_smb_dims), key=k6_smb_dims.count)})")
    if k12_smb_dims:
        print(f"• K=12 codes: SMB dimensions = {sorted(set(k12_smb_dims))} (most common: {max(set(k12_smb_dims), key=k12_smb_dims.count)})")
    if failed_smb_dims:
        print(f"• K=0 codes: SMB dimensions = {sorted(set(failed_smb_dims))} (most common: {max(set(failed_smb_dims), key=failed_smb_dims.count)})")
    
    print(f"\nTotal variants tested: {len(results)}")
    print(f"Successful codes: {len(successful_k6) + len(successful_k12)}")
    print(f"Failed codes: {len(failed_k0)}")












def reshape_to_grid(vector, l, m):
    """Reshape vector to l×m grid for visualization."""
    try:
        return vector.reshape(l, m)
    except:
        return None




def get_c_poly_variants():
    """Get standard c_poly variants for testing (backward compatibility)."""
    return get_standard_c_poly_variants()


if __name__ == "__main__":
    print("=" * 100)
    print("BT AND BB CODE ANALYSIS WITH STANDARD MONOMIAL BASIS")
    print("=" * 100)
    
    # Run comprehensive comparison
    comparison_results = comprehensive_comparison_table()
    
    print("\n" + "=" * 100)
    print("EXECUTION COMPLETED - Standard monomial basis analysis included for all code variants")
    print("=" * 100)
