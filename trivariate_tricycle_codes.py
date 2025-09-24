"""Utilities for constructing Trivariate Tricycle (TT) codes.

This module provides helpers to build the TT parity-check matrices (Hx, Hz)
from trivariate polynomial specifications. TT codes use three polynomials A, B, C
in three variables x, y, z with the same check structure as BT codes.
"""

from __future__ import annotations

from typing import List, Tuple, Sequence, Union

import numpy as np


def _shift_mat(n: int, k: int) -> np.ndarray:
    """Return n x n cyclic left shift-by-k matrix over GF(2) as uint8.
    
    The definition is consistent with the polynomial to qubit mapping.
    """
    return np.roll(np.identity(n, dtype=np.uint8), -int(k) % n, axis=1)


def _parse_trivariate_terms(
    spec: Union[Sequence[Tuple[int, int, int]], np.ndarray],
) -> List[Tuple[int, int, int]]:
    """Normalize polynomial spec into list of (i, j, k) tuples for x^i y^j z^k.

    Accepts:
    - list of tuples: [(i, j, k), ...] e.g., [(2,0,1), (1,1,0), (0,2,1)] 
    - ndarray shape (n, 3) with integer exponents
    - empty list [] for zero polynomial
    """
    if isinstance(spec, np.ndarray):
        arr = np.asarray(spec, dtype=np.uint8)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return [(int(i), int(j), int(k)) for i, j, k in arr]
        elif arr.size == 0:  # Handle empty ndarray
            return []
        raise ValueError("ndarray spec must have shape (n, 3)")

    if len(spec) == 0:  # Handle empty list (zero polynomial)
        return []
    
    if isinstance(spec[0], (list, tuple)) and len(spec[0]) == 3:
        return [(int(p[0]), int(p[1]), int(p[2])) for p in spec]

    raise ValueError("Unsupported polynomial spec format for trivariate terms")


def get_TT_matrix(
    a: Union[Sequence[Tuple[int, int, int]], np.ndarray], l: int, m: int, n: int
) -> np.ndarray:
    """Return the (l*m*n) x (l*m*n) binary matrix for polynomial a(x, y, z).

    The polynomial is specified by exponent tuples for monomials x^i y^j z^k. For
    each (i, j, k), contribute kron(kron(shift_x(i), shift_y(j)), shift_z(k)).
    """
    terms = _parse_trivariate_terms(a)
    A = np.zeros((l * m * n, l * m * n), dtype=np.uint8)
    for ix, iy, iz in terms:
        # Triple Kronecker product: kron(kron(X, Y), Z)
        xy_term = np.kron(_shift_mat(l, ix), _shift_mat(m, iy))
        xyz_term = np.kron(xy_term, _shift_mat(n, iz)).astype(np.uint8)
        A += xyz_term  # XOR accumulates modulo-2 for binary matrices in {0,1}
    return A % 2


def get_TT_Hx_Hz(
    a: Union[Sequence[Tuple[int, int, int]], np.ndarray],
    b: Union[Sequence[Tuple[int, int, int]], np.ndarray],
    c: Union[Sequence[Tuple[int, int, int]], np.ndarray],
    l: int,
    m: int,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build TT code Hx, Hz from trivariate polynomials a(x,y,z), b(x,y,z), c(x,y,z).

    TT Code Structure (same as BT but with trivariate polynomials):
    - X Check: Hx = [A, B, C]
    - Z Check: Hz = [[C^T, 0, A^T], [0, C^T, B^T], [B^T, A^T, 0]]

    Accepts list of exponent tuples [(i, j, k), ...] or an ndarray of shape (p, 3).
    """
    A = get_TT_matrix(a, l, m, n)
    B = get_TT_matrix(b, l, m, n)
    C = get_TT_matrix(c, l, m, n)
    
    # X Check: [A, B, C] - horizontal concatenation
    Hx = np.concatenate((A, B, C), axis=1).astype(np.uint8)
    
    # Z Check: 3x3 block structure
    # [[C^T, 0, A^T], [0, C^T, B^T], [B^T, A^T, 0]]
    zero_block = np.zeros_like(A, dtype=np.uint8)
    
    # Build the 3x3 block matrix row by row
    hz_row1 = np.concatenate((C.T, zero_block, A.T), axis=1)
    hz_row2 = np.concatenate((zero_block, C.T, B.T), axis=1)
    hz_row3 = np.concatenate((B.T, A.T, zero_block), axis=1)
    
    Hz = np.concatenate((hz_row1, hz_row2, hz_row3), axis=0).astype(np.uint8)
    
    return Hx, Hz


def get_TT_Hmeta(
    a: Union[Sequence[Tuple[int, int, int]], np.ndarray],
    b: Union[Sequence[Tuple[int, int, int]], np.ndarray],
    c: Union[Sequence[Tuple[int, int, int]], np.ndarray],
    l: int,
    m: int,
    n: int,
) -> np.ndarray:
    """Build TT meta Z check: Hmeta = [B^T, A^T, C^T].

    This is an auxiliary check matrix that may be useful for certain analyses.
    """
    A = get_TT_matrix(a, l, m, n)
    B = get_TT_matrix(b, l, m, n)
    C = get_TT_matrix(c, l, m, n)
    
    return np.concatenate((B.T, A.T, C.T), axis=1).astype(np.uint8)


__all__ = ["get_TT_matrix", "get_TT_Hx_Hz", "get_TT_Hmeta"]