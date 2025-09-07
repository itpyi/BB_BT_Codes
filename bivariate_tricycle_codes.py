"""Utilities for constructing Bivariate Tricycle (BT) codes.

This module provides helpers to build the BT parity-check matrices (Hx, Hz)
from bivariate polynomial specifications. BT codes use three polynomials A, B, C
with a more complex Z-check structure than BB codes.
"""

from __future__ import annotations

from typing import List, Tuple, Sequence, Union

import numpy as np


def _shift_mat(n: int, k: int) -> np.ndarray:
    """Return n x n cyclic shift-by-k matrix over GF(2) as uint8."""
    return np.roll(np.identity(n, dtype=np.uint8), int(k) % n, axis=1)


def _parse_bivariate_terms(
    spec: Union[Sequence[Tuple[int, int]], np.ndarray],
) -> List[Tuple[int, int]]:
    """Normalize polynomial spec into list of (i, j) pairs for x^i y^j.

    Accepts:
    - list of pairs: [(i, j), ...] e.g., [(2,0), (1,1), (0,2)] for x^2 + xy + y^2
    - ndarray shape (k, 2) with integer exponents
    - empty list [] for zero polynomial
    """
    if isinstance(spec, np.ndarray):
        arr = np.asarray(spec, dtype=np.uint8)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [(int(i), int(j)) for i, j in arr]
        elif arr.size == 0:  # Handle empty ndarray
            return []
        raise ValueError("ndarray spec must have shape (k, 2)")

    if len(spec) == 0:  # Handle empty list (zero polynomial)
        return []
    
    if isinstance(spec[0], (list, tuple)) and len(spec[0]) == 2:
        return [(int(p[0]), int(p[1])) for p in spec]

    raise ValueError("Unsupported polynomial spec format for bivariate terms")


def get_BT_matrix(
    a: Union[Sequence[Tuple[int, int]], np.ndarray], l: int, m: int
) -> np.ndarray:
    """Return the (l*m) x (l*m) binary matrix for polynomial a(x, y).

    The polynomial is specified by exponent pairs for monomials x^i y^j. For
    each (i, j), contribute kron(shift_x(i), shift_y(j)).
    """
    terms = _parse_bivariate_terms(a)
    A = np.zeros((l * m, l * m), dtype=np.uint8)
    for ix, iy in terms:
        term = np.kron(_shift_mat(l, ix), _shift_mat(m, iy)).astype(np.uint8)
        A += term  # XOR accumulates modulo-2 for binary matrices in {0,1}
    return A % 2


def get_BT_Hx_Hz(
    a: Union[Sequence[Tuple[int, int]], np.ndarray],
    b: Union[Sequence[Tuple[int, int]], np.ndarray],
    c: Union[Sequence[Tuple[int, int]], np.ndarray],
    l: int,
    m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build BT code Hx, Hz from bivariate polynomials a(x,y), b(x,y), c(x,y).

    BT Code Structure:
    - X Check: Hx = [A, B, C]
    - Z Check: Hz = [[C^T, 0, A^T], [0, C^T, B^T], [B^T, A^T, 0]]

    Accepts list of exponent pairs [(i, j), ...] or an ndarray of shape (k, 2).
    """
    A = get_BT_matrix(a, l, m)
    B = get_BT_matrix(b, l, m)
    C = get_BT_matrix(c, l, m)
    
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


def get_BT_Hmeta(
    a: Union[Sequence[Tuple[int, int]], np.ndarray],
    b: Union[Sequence[Tuple[int, int]], np.ndarray],
    c: Union[Sequence[Tuple[int, int]], np.ndarray],
    l: int,
    m: int,
) -> np.ndarray:
    """Build BT meta Z check: Hmeta = [B^T, A^T, C^T].

    This is an auxiliary check matrix that may be useful for certain analyses.
    """
    A = get_BT_matrix(a, l, m)
    B = get_BT_matrix(b, l, m)
    C = get_BT_matrix(c, l, m)
    
    return np.concatenate((B.T, A.T, C.T), axis=1).astype(np.uint8)


__all__ = ["get_BT_matrix", "get_BT_Hx_Hz", "get_BT_Hmeta"]