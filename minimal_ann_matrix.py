#!/usr/bin/env python3

"""
Matrix-based implementation for computing Ann(f)/(g Ann(f)) using direct Gaussian elimination.

Algorithm:
1. Compute Ann(f) as matrix M_f where rows are generators in basis x^i y^j
2. Compute g*Ann(f) as matrix M_g
3. Find quotient M_f/M_g using Gaussian elimination:
   - Stack matrices vertically: [[M_f], [M_g]]
   - Use Gaussian elimination to find rows of M_f independent from M_g
   - Independent rows form basis for quotient space
"""

import numpy as np
import sympy as sp
from sympy import symbols, expand
import ldpc.mod2 as mod2
from typing import Any, Dict, List, Optional, Tuple

from bivariate_bicycle_codes import get_BB_Hx_Hz
from bposd.css import css_code

# Set up polynomial variables
x, y = symbols('x y')

# Cache for Groebner bases of the ambient ring relations A = <x^l+1, y^m+1>
_ring_groebner_cache: Dict[Tuple[int, int], sp.GroebnerBasis] = {}

def _get_ring_groebner(l: int, m: int) -> sp.GroebnerBasis:
    key = (l, m)
    gb = _ring_groebner_cache.get(key)
    if gb is None:
        gb = sp.groebner([x**l + 1, y**m + 1], [x, y], modulus=2)
        _ring_groebner_cache[key] = gb
    return gb


def _monomial_basis(l: int, m: int) -> List[sp.Expr]:
    """Return ordered basis [x^i y^j] with 0 <= i < l, 0 <= j < m."""
    return [x**i * y**j for i in range(l) for j in range(m)]

def setup_ring(l: int, m: int):
    """Setup polynomial ring GF(2)[x,y] with periodic boundary x^l+1, y^m+1"""
    monomials = []
    for i in range(l):
        for j in range(m):
            monomials.append(x**i * y**j)

    print(f"Ring: GF(2)[x,y] with {len(monomials)} monomials")
    print(f"Boundary conditions: x^{l} = 1, y^{m} = 1")
    return monomials

def apply_periodic_boundary(poly, l: int, m: int):
    """Reduce polynomial modulo A = <x^l+1, y^m+1> using Groebner remainder."""
    G_rel = _get_ring_groebner(l, m)
    _, rem = G_rel.reduce(expand(poly))
    return expand(rem)

def poly_to_vector(poly, monomials, l: int, m: int):
    """Convert polynomial to coefficient vector modulo A via Groebner reduction."""
    poly_reduced = apply_periodic_boundary(poly, l, m)
    vector = np.zeros(len(monomials), dtype=np.uint8)
    if poly_reduced == 0:
        return vector
    terms = sp.Add.make_args(poly_reduced) if poly_reduced.is_Add else [poly_reduced]
    for term in terms:
        if term.is_number:
            if int(term) % 2 == 1:
                try:
                    idx = monomials.index(1)
                    vector[idx] = 1
                except ValueError:
                    pass
        else:
            coeff, monom_part = term.as_coeff_Mul()
            if int(coeff) % 2 == 1:
                try:
                    idx = monomials.index(monom_part)
                    vector[idx] = 1
                except ValueError:
                    pass
    return vector

def vector_to_poly(vector, monomials):
    """Convert coefficient vector back to polynomial"""
    poly = 0
    for i, coeff in enumerate(vector):
        if coeff % 2 == 1:  # In GF(2)
            poly += monomials[i]
    return poly


def _row_space_basis(matrix: np.ndarray) -> np.ndarray:
    """Return a row-echelon basis for the row space of `matrix` over GF(2)."""

    if matrix.size == 0 or matrix.shape[0] == 0:
        return np.zeros((0, matrix.shape[1] if matrix.ndim == 2 else 0), dtype=np.uint8)

    matrix_uint8 = np.asarray(matrix, dtype=np.uint8)
    rref, _, _, _ = mod2.row_echelon(matrix_uint8, full=True)
    if not isinstance(rref, np.ndarray):
        rref = rref.toarray()

    basis_rows = [row.astype(np.uint8) % 2 for row in rref if np.any(row)]
    if not basis_rows:
        return np.zeros((0, matrix.shape[1]), dtype=np.uint8)

    return np.stack(basis_rows).astype(np.uint8)


def _principal_ideal_basis(
    poly: sp.Expr, monomials: List[sp.Expr], l: int, m: int
) -> Tuple[np.ndarray, List[sp.Expr]]:
    """Return basis vectors and polynomials spanning the principal ideal (poly)
    in R = GF(2)[x,y]/(x^l+1, y^m+1), using Gröbner reduction modulo the ring
    relations only (not including poly as a reducer).

    The ideal (poly) in R is the image of the linear map r -> r*poly (mod A),
    so we compute columns v_b = vec(b*poly) for each basis monomial b and then
    return a row-basis for the column space.
    """

    if poly == 0:
        zero_basis = np.zeros((0, len(monomials)), dtype=np.uint8)
        return zero_basis, []

    # Gröbner bases: ambient relations and full ideal with poly
    G_rel = _get_ring_groebner(l, m)
    I = sp.groebner([sp.expand(poly), x**l + 1, y**m + 1], [x, y], modulus=2)

    def reduce_mod_A(expr: sp.Expr) -> sp.Expr:
        _, rem = G_rel.reduce(sp.expand(expr))
        return sp.expand(rem)

    def poly_to_vec_gb(expr: sp.Expr) -> np.ndarray:
        rem = reduce_mod_A(expr)
        vec = np.zeros(len(monomials), dtype=np.uint8)
        if rem == 0:
            return vec
        terms = sp.Add.make_args(rem) if rem.is_Add else [rem]
        for term in terms:
            if term.is_number:
                if int(term) % 2 == 1:
                    try:
                        idx = monomials.index(1)
                        vec[idx] = 1
                    except ValueError:
                        pass
            else:
                coeff, mon_part = term.as_coeff_Mul()
                if int(coeff) % 2 == 1:
                    try:
                        idx = monomials.index(mon_part)
                        vec[idx] = 1
                    except ValueError:
                        pass
        return vec

    # Build a spanning set using the Groebner generators of I
    gens = [gp.as_expr() if hasattr(gp, 'as_expr') else gp for gp in I.polys]

    N = len(monomials)
    cols: List[np.ndarray] = []
    for mon in monomials:
        for ggen in gens:
            cols.append(poly_to_vec_gb(mon * ggen))

    M = np.stack(cols, axis=1) if cols else np.zeros((N, 0), dtype=np.uint8)

    basis_matrix = _row_space_basis(M.T)
    basis_polys = [vector_to_poly(row, monomials) for row in basis_matrix]
    return basis_matrix, basis_polys

def _ideal_span_from_generators(
    gens: List[sp.Expr], monomials: List[sp.Expr], l: int, m: int
) -> Tuple[np.ndarray, List[sp.Expr]]:
    """Given generators of an ideal in k[x,y], return the GF(2) row-basis of its
    image in R = GF(2)[x,y]/(x^l+1, y^m+1), by multiplying gens with the ring
    monomial basis and reducing by A only.
    """
    G_rel = _get_ring_groebner(l, m)

    def reduce_mod_A(expr: sp.Expr) -> sp.Expr:
        _, rem = G_rel.reduce(sp.expand(expr))
        return sp.expand(rem)

    def poly_to_vec_gb(expr: sp.Expr) -> np.ndarray:
        rem = reduce_mod_A(expr)
        vec = np.zeros(len(monomials), dtype=np.uint8)
        if rem == 0:
            return vec
        terms = sp.Add.make_args(rem) if rem.is_Add else [rem]
        for term in terms:
            if term.is_number:
                if int(term) % 2 == 1:
                    try:
                        idx = monomials.index(1)
                        vec[idx] = 1
                    except ValueError:
                        pass
            else:
                coeff, mon_part = term.as_coeff_Mul()
                if int(coeff) % 2 == 1:
                    try:
                        idx = monomials.index(mon_part)
                        vec[idx] = 1
                    except ValueError:
                        pass
        return vec

    N = len(monomials)
    cols: List[np.ndarray] = []
    for mon in monomials:
        for g in gens:
            cols.append(poly_to_vec_gb(mon * g))
    M = np.stack(cols, axis=1) if cols else np.zeros((N, 0), dtype=np.uint8)
    basis_matrix = _row_space_basis(M.T)
    basis_polys = [vector_to_poly(row, monomials) for row in basis_matrix]
    return basis_matrix, basis_polys

def _intersection_ideal_basis_via_elimination(
    f_poly: sp.Expr, g_poly: sp.Expr, monomials: List[sp.Expr], l: int, m: int
) -> Tuple[np.ndarray, List[sp.Expr]]:
    """Compute a basis for (I ∩ J) in R using elimination:
    I = ⟨f, x^l+1, y^m+1⟩, J = ⟨g, x^l+1, y^m+1⟩.
    In k[x,y,t], let K = ⟨t*I, (1+t)*J⟩ (since char=2, 1−t=1+t). Then
    (I ∩ J) = K ∩ k[x,y]. Extract polynomials from a Groebner basis of K that
    do not involve t, then span in R.
    """
    t = sp.symbols('t')
    gens_I = [sp.expand(f_poly), x**l + 1, y**m + 1]
    gens_J = [sp.expand(g_poly), x**l + 1, y**m + 1]
    K_gens = [t*gi for gi in gens_I] + [(1 + t)*gj for gj in gens_J]
    G = sp.groebner(K_gens, [t, x, y], order='lex', modulus=2)
    inter_gens = []
    for gp in G.polys:
        expr = gp.as_expr() if hasattr(gp, 'as_expr') else gp
        if not expr.has(t):
            inter_gens.append(expr)
    # Fallback: if no t-free generators are found, use simple product to avoid empty
    if not inter_gens:
        inter_gens = [sp.expand(f_poly)*sp.expand(g_poly)]
    return _ideal_span_from_generators(inter_gens, monomials, l, m)


def _subspace_intersection(
    A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Return row-space basis for the intersection of row spaces of A and B."""

    if A.size == 0 or B.size == 0:
        return np.zeros((0, A.shape[1] if A.size else B.shape[1]), dtype=np.uint8)

    A_uint8 = np.asarray(A, dtype=np.uint8)
    B_uint8 = np.asarray(B, dtype=np.uint8)

    N = A_uint8.shape[1]
    if B_uint8.shape[1] != N:
        raise ValueError("Subspaces must share the same ambient dimension")

    augmented = np.hstack([A_uint8.T, B_uint8.T])
    nullspace_vecs = mod2.nullspace(augmented)

    intersection_vectors: List[np.ndarray] = []

    for vec in nullspace_vecs:
        if hasattr(vec, "toarray"):
            gamma = vec.toarray().astype(np.uint8).flatten() % 2
        else:
            gamma = np.asarray(vec, dtype=np.uint8).flatten() % 2

        alpha = gamma[: A_uint8.shape[0]]
        if not np.any(alpha):
            continue

        combo = np.zeros(N, dtype=np.uint8)
        for idx, coeff in enumerate(alpha):
            if coeff % 2 == 1:
                combo ^= A_uint8[idx]

        if np.any(combo):
            intersection_vectors.append(combo)

    if not intersection_vectors:
        return np.zeros((0, N), dtype=np.uint8)

    return _row_space_basis(np.stack(intersection_vectors))

def compute_ann_f_matrix(f_poly, monomials, l: int, m: int):
    """Compute Ann(f) as matrix M_f where rows are generators"""

    print(f"\n=== Computing Ann(f) Matrix where f = {f_poly} ===")

    N = len(monomials)
    M = np.zeros((N, N), dtype=np.uint8)

    # Build matrix: M[i,j] = coefficient of monomial_i in (monomial_j * f)
    for j, monom_j in enumerate(monomials):
        product = expand(monom_j * f_poly)
        product_vec = poly_to_vector(product, monomials, l, m)
        M[:, j] = product_vec

    print(f"Built multiplication matrix M of size {M.shape}")
    print(f"Matrix rank: {mod2.rank(M)}")

    # Find nullspace: h such that h * f = 0
    nullspace_vecs = mod2.nullspace(M)

    # Convert to matrix format
    ann_f_rows: List[np.ndarray] = []
    ann_f_polys: List[sp.Expr] = []

    for vec in nullspace_vecs:
        if hasattr(vec, 'toarray'):
            coeffs = vec.toarray().flatten()
        else:
            coeffs = np.asarray(vec).flatten()
        coeffs = (coeffs.astype(np.uint8) % 2).reshape(1, -1)
        ann_f_rows.append(coeffs[0])
        poly = vector_to_poly(coeffs[0], monomials)
        if poly != 0:
            ann_f_polys.append(poly)

    if ann_f_rows:
        M_f = np.vstack(ann_f_rows).astype(np.uint8)
    else:
        M_f = np.zeros((0, len(monomials)), dtype=np.uint8)

    print(f"Ann(f) matrix M_f shape: {M_f.shape}")
    print(f"Ann(f) has {len(ann_f_polys)} non-zero generators:")
    for i, gen in enumerate(ann_f_polys):
        print(f"  Ann(f)[{i}]: {gen}")

    return M_f, ann_f_polys

def compute_g_ann_f_matrix(g_poly, M_f, ann_f_polys, monomials, l: int, m: int):
    """Compute g*Ann(f) as matrix M_g"""

    print(f"\n=== Computing g*Ann(f) Matrix where g = {g_poly} ===")

    g_ann_f_matrix: List[np.ndarray] = []
    g_ann_f_polys: List[sp.Expr] = []

    # For each generator h in Ann(f), compute g*h
    for i, h_poly in enumerate(ann_f_polys):
        gh_poly = expand(g_poly * h_poly)
        gh_vec = poly_to_vector(gh_poly, monomials, l, m)

        g_ann_f_matrix.append(gh_vec)
        g_ann_f_polys.append(gh_poly)

        print(f"  g*Ann(f)[{i}]: ({g_poly}) * ({h_poly}) = {gh_poly}")

    if g_ann_f_matrix:
        M_g = np.vstack(g_ann_f_matrix).astype(np.uint8)
    else:
        M_g = np.zeros((0, len(monomials)), dtype=np.uint8)

    print(f"g*Ann(f) matrix M_g shape: {M_g.shape}")

    return M_g, g_ann_f_polys

def compute_quotient_matrix(M_f, M_g, monomials, verbose: bool = True):
    """Compute quotient M_f/M_g using Gaussian elimination with proper sparse handling"""

    if verbose:
        print(f"\n=== Computing Quotient M_f/M_g using Gaussian elimination ===")
        print(f"M_f shape: {M_f.shape}")
        print(f"M_g shape: {M_g.shape}")

    if M_f.shape[0] == 0:
        if verbose:
            print("M_f is empty, quotient is trivial")
        return np.zeros((0, M_f.shape[1]), dtype=np.uint8), []

    if M_g.shape[0] == 0:
        if verbose:
            print("M_g is empty, quotient is just M_f")
        non_zero_rows = []
        non_zero_polys = []
        for row in M_f:
            if np.any(row):
                non_zero_rows.append(row)
                non_zero_polys.append(vector_to_poly(row, monomials))
        return np.array(non_zero_rows, dtype=np.uint8), non_zero_polys

    # Use simple rank-based approach to avoid sparse array complexity
    quotient_rows = []
    quotient_polys = []

    # Track the span of g*Ann(f) augmented with the quotient rows we keep
    # BUG FIXED
    # The quotient loop now keeps a mutable basis starting from M_g. Every time we
    # accept an Ann(f) row, it gets appended to this basis and the stored rank is
    # updated. Later rows are tested against the combined span rather than the
    # original M_g, so any component already generated by g·Ann(f) (or by quotient
    # rows we have already kept) is filtered out. This prevents overcounting
    # logical operators.
    basis_matrix = M_g.copy()
    rank_basis = mod2.rank(basis_matrix)
    if verbose:
        print(f"Rank of M_g: {rank_basis}")

    # For each row in M_f, check if it's linearly independent from current basis
    for i, f_row in enumerate(M_f):
        if not np.any(f_row):
            continue  # Skip zero rows

        # Test if adding f_row expands the current span
        test_matrix = np.vstack([basis_matrix, f_row.reshape(1, -1)])
        rank_test = mod2.rank(test_matrix)

        if rank_test > rank_basis:
            # f_row adds a new coset representative to the quotient basis
            quotient_rows.append(f_row)
            poly = vector_to_poly(f_row, monomials)
            quotient_polys.append(poly)
            if verbose:
                print(f"  Quotient[{len(quotient_rows)-1}]: {poly} (independent)")
            basis_matrix = test_matrix
            rank_basis = rank_test
        else:
            if verbose:
                poly = vector_to_poly(f_row, monomials)
                print(f"  Skipping M_f[{i}]: {poly} (dependent on M_g)")

    quotient_matrix = np.array(quotient_rows, dtype=np.uint8) if quotient_rows else np.zeros((0, M_f.shape[1]), dtype=np.uint8)

    if verbose:
        print(f"Final quotient matrix shape: {quotient_matrix.shape}")
        print(f"Quotient dimension: {len(quotient_polys)}")

    return quotient_matrix, quotient_polys


def compute_tor_1(
    f_str: str, g_str: str, l: int, m: int
) -> Dict[str, Any]:
    """Compute Tor_1^R(R/(f), R/(g)) ≅ (I ∩ J)/(I·J) for R = GF(2)[x,y]/(x^l+1, y^m+1).

    Uses Gröbner-based principal ideals for I = ⟨f, x^l+1, y^m+1⟩ and
    J = ⟨g, x^l+1, y^m+1⟩. The intersection and product spaces are handled in the
    finite-dimensional vector space over GF(2) induced by monomials {x^i y^j}.
    Verbose prints of intermediate basis sizes are removed.
    """

    f_poly = sp.sympify(f_str)
    g_poly = sp.sympify(g_str)
    monomials = _monomial_basis(l, m)

    # Principal ideals via Gröbner reduction
    ideal_f_matrix, ideal_f_polys = _principal_ideal_basis(f_poly, monomials, l, m)
    ideal_g_matrix, ideal_g_polys = _principal_ideal_basis(g_poly, monomials, l, m)

    # Intersection (f) ∩ (g) via elimination-based ideal intersection
    intersection_matrix, intersection_polys = _intersection_ideal_basis_via_elimination(
        f_poly, g_poly, monomials, l, m
    )

    # Product ideal (f)(g) = (f * g)
    product_poly = expand(f_poly * g_poly)
    product_matrix, product_polys = _principal_ideal_basis(product_poly, monomials, l, m)

    # Torsion quotient (I ∩ J)/(I·J)
    tor_matrix, tor_polys = compute_quotient_matrix(intersection_matrix, product_matrix, monomials, verbose=False)

    return {
        "ideal_f_matrix": ideal_f_matrix,
        "ideal_f_basis": ideal_f_polys,
        "ideal_g_matrix": ideal_g_matrix,
        "ideal_g_basis": ideal_g_polys,
        "intersection_matrix": intersection_matrix,
        "intersection_basis": intersection_polys,
        "product_matrix": product_matrix,
        "product_basis": product_polys,
        "tor_matrix": tor_matrix,
        "tor_basis": tor_polys,
        "dimension": len(tor_polys),
    }


def _polynomial_to_block_indicator(poly: sp.Expr, l: int, m: int) -> np.ndarray:
    """Return l x m binary array marking qubit positions touched by poly."""

    # WARNING Need Check

    block = np.zeros((l, m), dtype=np.uint8)
    if poly == 0:
        return block

    basis = _monomial_basis(l, m)
    vec = poly_to_vector(poly, basis, l, m)

    for idx in np.where(vec == 1)[0]:
        a = idx // m
        b = idx % m
        block[a, b] = 1

    return block


def _poly_to_exponent_pairs(poly: sp.Expr, l: int, m: int) -> List[Tuple[int, int]]:
    """Convert polynomial into list of exponent pairs (a, b) with coefficients in GF(2)."""

    basis = _monomial_basis(l, m)
    vec = poly_to_vector(poly, basis, l, m)
    pairs: List[Tuple[int, int]] = []
    for idx in np.where(vec == 1)[0]:
        a = int(idx // m)
        b = int(idx % m)
        pairs.append((a, b))
    return pairs


def build_qubit_logical_indicator(
    poly: sp.Expr,
    l: int,
    m: int,
    block: Any,
) -> Dict[str, Any]:
    """Map polynomial to flattened qubit indicator for block1, block2, or torsion."""

    qubit_tensor = np.zeros((2, l, m), dtype=np.uint8)
    indicator = _polynomial_to_block_indicator(poly, l, m)

    if block == 0:
        qubit_tensor[0] = indicator
        block_label: Any = 1
    elif block == 1:
        qubit_tensor[1] = indicator
        block_label = 2
    elif block == "torsion":
        qubit_tensor[0] = indicator
        qubit_tensor[1] = indicator
        block_label = "torsion"
    else:
        raise ValueError("block must be 0, 1, or 'torsion'")

    qubit_vector = qubit_tensor.reshape(-1)

    return {
        "poly": poly,
        "block": block_label,
        "tensor": qubit_tensor,
        "vector": qubit_vector,
    }


def compute_logical_qubit_operators(f_str: str, g_str: str, l: int, m: int) -> Dict[str, Any]:
    """Return logical Z operators as qubit occupancy vectors for both blocks."""

    result_f = compute_ann_quotient_matrix(f_str, g_str, l, m)
    result_g = compute_ann_quotient_matrix(g_str, f_str, l, m)

    block1_ops = []
    for idx, poly in enumerate(result_f["quotient_basis"]):
        entry = build_qubit_logical_indicator(poly, l, m, block=0)
        entry["index"] = idx
        block1_ops.append(entry)

    block2_ops = []
    for idx, poly in enumerate(result_g["quotient_basis"]):
        entry = build_qubit_logical_indicator(poly, l, m, block=1)
        entry["index"] = idx
        block2_ops.append(entry)

    torsion = compute_tor_1(f_str, g_str, l, m)
    torsion_ops = []
    for idx, poly in enumerate(torsion["tor_basis"]):
        entry = build_qubit_logical_indicator(poly, l, m, block="torsion")
        entry["index"] = idx
        torsion_ops.append(entry)

    # all_vectors = [op["vector"] for op in block1_ops + block2_ops + torsion_ops]
    all_vectors = [op["vector"] for op in block1_ops + block2_ops]
    if all_vectors:
        logical_matrix = np.vstack(all_vectors).astype(np.uint8)
        span_rank = mod2.rank(logical_matrix)
    else:
        logical_matrix = np.zeros((0, 2 * l * m), dtype=np.uint8)
        span_rank = 0

    return {
        "block1": block1_ops,
        "block2": block2_ops,
        "torsion": torsion_ops,
        "tor_details": torsion,
        "matrix": logical_matrix,
        "independence": {
            "matrix": logical_matrix,
            "rank": span_rank,
            "count": logical_matrix.shape[0],
            "independent": span_rank == logical_matrix.shape[0],
        },
    }


def verify_logical_z_equivalence(
    f_str: str,
    g_str: str,
    l: int,
    m: int,
    logicals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Show polynomial logical Zs match css_code logical Zs up to Z stabilizers."""

    if logicals is None:
        logicals = compute_logical_qubit_operators(f_str, g_str, l, m)

    f_poly = sp.sympify(f_str)
    g_poly = sp.sympify(g_str)
    f_terms = _poly_to_exponent_pairs(f_poly, l, m)
    g_terms = _poly_to_exponent_pairs(g_poly, l, m)

    Hx, Hz = get_BB_Hx_Hz(f_terms, g_terms, l, m)
    code = css_code(hx=Hx, hz=Hz, name=f"BB_{l}x{m}")

    lz_matrix = code.lz.toarray().astype(np.uint8)

    z_stab_sparse = mod2.row_basis(Hz)

    z_stab_basis = z_stab_sparse.toarray().astype(np.uint8)

    if z_stab_basis.size == 0:
        z_stab_basis = np.zeros((0, Hz.shape[1]), dtype=np.uint8)
    elif z_stab_basis.ndim == 1:
        z_stab_basis = z_stab_basis.reshape(1, -1)

    z_stab_rank = z_stab_basis.shape[0] if z_stab_basis.size else 0

    poly_entries = logicals["block1"] + logicals["block2"] + logicals.get("torsion", [])
    poly_matrix = (
        np.vstack([entry["vector"] for entry in poly_entries]).astype(np.uint8)
        if poly_entries
        else np.zeros((0, Hz.shape[1]), dtype=np.uint8)
    )

    css_stack = (
        np.vstack([z_stab_basis, lz_matrix])
        if lz_matrix.size or z_stab_basis.size
        else np.zeros((0, Hz.shape[1]), dtype=np.uint8)
    )
    poly_stack = (
        np.vstack([z_stab_basis, poly_matrix])
        if poly_matrix.size or z_stab_basis.size
        else np.zeros((0, Hz.shape[1]), dtype=np.uint8)
    )
    # rank(css ∪ Z)=42, rank(poly ∪ Z)=45, rank(union)=54, rank(Z stabilizer)=30
    # 2025/09/20
    # BUG! Why rank_poly - rank_Z is larger than the rank of the poly itself?
    rank_css = mod2.rank(css_stack)
    rank_poly = mod2.rank(poly_stack)
    rank_union = (
        mod2.rank(np.vstack([z_stab_basis, lz_matrix, poly_matrix]))
    )

    def in_z_stabilizer_span(vec: np.ndarray) -> bool:
        if z_stab_basis.size == 0:
            return not np.any(vec)
        stacked = np.vstack([z_stab_basis, vec])
        return mod2.rank(stacked) == mod2.rank(z_stab_basis)

    matches: List[Dict[str, Any]] = []
    unmatched_polynomial: List[Dict[str, Any]] = []

    for entry in poly_entries:
        vec = entry["vector"].astype(np.uint8)
        found = False
        for idx, css_vec in enumerate(lz_matrix):
            diff = vec ^ css_vec
            if in_z_stabilizer_span(diff):
                matches.append(
                    {
                        "poly_block": entry["block"],
                        "poly_index": entry["index"],
                        "css_lz_index": idx,
                    }
                )
                found = True
                break
        if not found:
            unmatched_polynomial.append(entry)

    unmatched_css = []
    for idx, css_vec in enumerate(lz_matrix):
        found = False
        for entry in poly_entries:
            diff = entry["vector"] ^ css_vec
            if in_z_stabilizer_span(diff):
                found = True
                break
        if not found:
            unmatched_css.append(idx)

    poly_not_in_css_span: List[int] = []
    css_not_in_poly_span: List[int] = []

    if css_stack.size:
        for idx, vec in enumerate(poly_matrix):
            stacked = np.vstack([css_stack, vec])
            if mod2.rank(stacked) > rank_css:
                poly_not_in_css_span.append(idx)

    if poly_stack.size:
        for idx, vec in enumerate(lz_matrix):
            stacked = np.vstack([poly_stack, vec])
            if mod2.rank(stacked) > rank_poly:
                css_not_in_poly_span.append(idx)

    return {
        "matches": matches,
        "unmatched_polynomial": unmatched_polynomial,
        "unmatched_css": unmatched_css,
        "poly_not_in_css_span": poly_not_in_css_span,
        "css_not_in_poly_span": css_not_in_poly_span,
        "rank_css_space": rank_css,
        "rank_poly_space": rank_poly,
        "rank_union_space": rank_union,
        "rank_z_stabilizer": z_stab_rank,
        "lz_matrix": lz_matrix,
        "z_stabilizer_basis": z_stab_basis,
    }


def _express_vector_in_basis(
    target: np.ndarray,
    basis: np.ndarray,
    labels: List[str],
) -> Optional[List[str]]:
    """Return labels whose XOR reproduces `target`, or None if outside the span."""

    target = target.astype(np.uint8, copy=False)
    if basis.size == 0:
        return [] if not np.any(target) else None

    stacked = np.vstack([basis, target])
    row_echelon_result, _, transform, _ = mod2.row_echelon(stacked, full=True)
    if np.any(row_echelon_result[-1]):
        return None

    coeffs = np.asarray(transform[-1, :-1]).astype(np.uint8) % 2
    return [labels[idx] for idx, val in enumerate(coeffs) if val]


def _express_with_preference(
    target: np.ndarray,
    primary_basis: np.ndarray,
    primary_labels: List[str],
    fallback_basis: np.ndarray,
    fallback_labels: List[str],
) -> Optional[List[str]]:
    """Prefer expressing `target` using `primary_basis` before falling back."""

    if primary_basis.size:
        primary_combo = _express_vector_in_basis(target, primary_basis, primary_labels)
        if primary_combo is not None:
            return primary_combo
    elif not np.any(target):
        return []

    return _express_vector_in_basis(target, fallback_basis, fallback_labels)


def _print_logical_equivalence_details(
    logicals: Dict[str, Any], equivalence: Dict[str, Any]
) -> None:
    """Print Z stabilizers and CSS↔polynomial logical relationships explicitly."""

    z_basis = equivalence["z_stabilizer_basis"].astype(np.uint8, copy=False)
    if z_basis.size:
        print("Z stabilizer basis (each row is a binary array):")
        for idx, row in enumerate(z_basis):
            print(f"  stabilizer_z[{idx}] = {row}")
    else:
        print("Z stabilizer basis (empty)")

    poly_entries = logicals["block1"] + logicals["block2"]
    if poly_entries:
        poly_matrix = np.vstack([entry["vector"] for entry in poly_entries]).astype(np.uint8)
    else:
        poly_matrix = np.zeros((0, 0), dtype=np.uint8)

    css_matrix = equivalence["lz_matrix"].astype(np.uint8, copy=False)

    poly_labels = [f"poly_logical[{idx}]" for idx in range(poly_matrix.shape[0])]
    css_labels = [f"css_logical_z[{idx}]" for idx in range(css_matrix.shape[0])]
    stab_labels = [f"stabilizer_z[{idx}]" for idx in range(z_basis.shape[0])]

    num_qubits = 0
    if poly_entries:
        num_qubits = poly_entries[0]["vector"].size
    elif z_basis.size:
        num_qubits = z_basis.shape[1]
    elif css_matrix.size:
        num_qubits = css_matrix.shape[1]

    css_basis = (
        np.vstack([poly_matrix, z_basis])
        if poly_matrix.size or z_basis.size
        else np.zeros((0, num_qubits), dtype=np.uint8)
    )
    css_basis_labels = poly_labels + stab_labels

    poly_basis = (
        np.vstack([css_matrix, z_basis])
        if css_matrix.size or z_basis.size
        else np.zeros((0, num_qubits), dtype=np.uint8)
    )
    poly_basis_labels = css_labels + stab_labels

    if css_matrix.size:
        print("CSS logical Z expressed via polynomial logicals and Z stabilizers:")
        for idx, vec in enumerate(css_matrix):
            combo = _express_with_preference(
                vec,
                poly_matrix,
                poly_labels,
                css_basis,
                css_basis_labels,
            )
            if combo is None:
                print(
                    "  css_logical_z[{idx}] cannot be expressed via polynomial logicals and Z stabilizers".format(
                        idx=idx
                    )
                )
            else:
                rhs = " + ".join(combo) if combo else "0"
                print(f"  css_logical_z[{idx}] = {rhs}")

    if poly_matrix.size:
        print("Polynomial logical Z expressed via css_code logicals and Z stabilizers:")
        for idx, vec in enumerate(poly_matrix):
            combo = _express_with_preference(
                vec,
                css_matrix,
                css_labels,
                poly_basis,
                poly_basis_labels,
            )
            if combo is None:
                print(
                    "  poly_logical[{idx}] cannot be expressed via css_code logicals and Z stabilizers".format(
                        idx=idx
                    )
                )
            else:
                rhs = " + ".join(combo) if combo else "0"
                print(f"  poly_logical[{idx}] = {rhs}")


def compute_ann_quotient_matrix(f_str: str, g_str: str, l: int, m: int):
    """Main function to compute Ann(f)/(g Ann(f)) using matrix Gaussian elimination"""

    print(f"=== Computing Ann({f_str})/(g Ann({f_str})) using Matrix Method ===")
    print(f"Ring: GF(2)[x,y]/(x^{l}+1, y^{m}+1)")
    print()

    # Setup
    monomials = setup_ring(l, m)

    # Parse polynomials
    f_poly = sp.sympify(f_str)
    g_poly = sp.sympify(g_str)

    print(f"f = {f_poly}")
    print(f"g = {g_poly}")

    # Step 1: Compute Ann(f) as matrix M_f
    M_f, ann_f_polys = compute_ann_f_matrix(f_poly, monomials, l, m)

    # Step 2: Compute g*Ann(f) as matrix M_g
    M_g, g_ann_f_polys = compute_g_ann_f_matrix(g_poly, M_f, ann_f_polys, monomials, l, m)

    # Step 3: Compute quotient M_f/M_g using Gaussian elimination
    quotient_matrix, quotient_polys = compute_quotient_matrix(M_f, M_g, monomials)

    print(f"\n=== Final Result ===")
    print(f"Dimension of Ann({f_str})/(g Ann({f_str})): {len(quotient_polys)}")
    print(f"Quotient basis:")
    for i, poly in enumerate(quotient_polys):
        print(f"  QuotientBasis[{i}]: {poly}")

    return {
        "M_f": M_f,
        "M_g": M_g,
        "quotient_matrix": quotient_matrix,
        "quotient_basis": quotient_polys,
        "dimension": len(quotient_polys)
    }

def compute_ann_quotient_symmetric(f_str: str, g_str: str, l: int, m: int):
    """Compute both Ann(f)/(g Ann(f)) and Ann(g)/(f Ann(g))"""

    print(f"=== Computing Both Quotients: Ann(f)/(g Ann(f)) and Ann(g)/(f Ann(g)) ===")
    print(f"Ring: GF(2)[x,y]/(x^{l}+1, y^{m}+1)")
    print(f"f = {f_str}, g = {g_str}")
    print()

    # Compute Ann(f)/(g Ann(f))
    print(">>> PART 1: Computing Ann(f)/(g Ann(f)) <<<")
    result_f = compute_ann_quotient_matrix(f_str, g_str, l, m)

    print("\n" + "="*80)

    # Compute Ann(g)/(f Ann(g))
    print(">>> PART 2: Computing Ann(g)/(f Ann(g)) <<<")
    result_g = compute_ann_quotient_matrix(g_str, f_str, l, m)

    print(f"\n=== Summary ===")
    print(f"Dimension of Ann({f_str})/(g Ann({f_str})): {result_f['dimension']}")
    print(f"Dimension of Ann({g_str})/(f Ann({g_str})): {result_g['dimension']}")

    return {
        "ann_f_quotient": result_f,
        "ann_g_quotient": result_g
    }

def run_test_examples():
    """Run test examples"""

    test_cases = [
        ("1 + x", "1 + y", 3, 3),
        ("1 + x + x*y", "1 + y + x*y", 3, 3),
        # ("1 + x + x*y", "1 + y + x*y", 9, 9),
        ("x^3 + y + y^2", "y^3 + x + x^2", 6, 6),
    ]

    print("=== Running Test Examples ===")

    for i, (f, g, l, m) in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: f={f}, g={g}, ring=GF(2)[x,y]/(x^{l}+1,y^{m}+1) ---")
        try:
            # quotient_summary = compute_ann_quotient_symmetric(f, g, l, m)
            # print(f"Ann(f)/(g Ann(f)) dimension = {quotient_summary['ann_f_quotient']['dimension']}")
            # print(f"Ann(g)/(f Ann(g)) dimension = {quotient_summary['ann_g_quotient']['dimension']}")

            logicals = compute_logical_qubit_operators(f, g, l, m)

            print("Logical Z operators on block 1 (Ann(f)/(g Ann(f))):")
            for entry in logicals["block1"]:
                print(f"  index {entry['index']}, poly {entry['poly']}")
                print("    tensor=", np.array2string(entry["tensor"], separator=", "))
                print("    vector=", entry["vector"])

            print("Logical Z operators on block 2 (Ann(g)/(f Ann(g))):")
            for entry in logicals["block2"]:
                print(f"  index {entry['index']}, poly {entry['poly']}")
                print("    tensor=", np.array2string(entry["tensor"], separator=", "))
                print("    vector=", entry["vector"])

            print("Logical Z torsion operators (Tor_1):")
            for entry in logicals["torsion"]:
                print(f"  index {entry['index']}, poly {entry['poly']}")
                print("    tensor=", np.array2string(entry["tensor"], separator=", "))
                print("    vector=", entry["vector"])

            indep = logicals["independence"]
            print(
                f"Logical operator rank check: rank={indep['rank']} count={indep['count']} independent={indep['independent']}"
            )
            print("Stacked logical Z matrix shape:", logicals["matrix"].shape)
            print(logicals["matrix"])

            equivalence = verify_logical_z_equivalence(f, g, l, m, logicals)
            print(
                f"Polynomial ↔ css_code logical Z matches: {len(equivalence['matches'])}; "
                f"unmatched polynomial (pairwise)={len(equivalence['unmatched_polynomial'])}; "
                f"unmatched css (pairwise)={len(equivalence['unmatched_css'])}"
            )
            print(
                "  rank(css ∪ Z)={rank_css}, rank(poly ∪ Z)={rank_poly}, rank(union)={rank_union}, rank(Z stabilizer)={rank_z}".format(
                    rank_css=equivalence["rank_css_space"],
                    rank_poly=equivalence["rank_poly_space"],
                    rank_union=equivalence["rank_union_space"],
                    rank_z=equivalence["rank_z_stabilizer"],
                )
            )
            _print_logical_equivalence_details(logicals, equivalence)
            if equivalence["poly_not_in_css_span"]:
                print(
                    "  ⚠ Polynomial vectors outside span(css ∪ Z) indices:",
                    equivalence["poly_not_in_css_span"],
                )
            if equivalence["css_not_in_poly_span"]:
                print(
                    "  ⚠ css_code vectors outside span(polynomials ∪ Z) indices:",
                    equivalence["css_not_in_poly_span"],
                )

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 80)

if __name__ == "__main__":
    # Test with simple example
    # print("=== Single Test Example ===")
    # print("Testing with f = x + 1, g = y + 1, in GF(2)[x,y]/(x^3+1, y^3+1)")
    # result = compute_ann_quotient_matrix("x + 1", "y + 1", 3, 3)
    # print("\n" + "="*80)

    # Run test suite
    run_test_examples()
