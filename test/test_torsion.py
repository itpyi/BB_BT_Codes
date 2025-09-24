"""Unit tests for torsion computation in minimal_ann_matrix."""

from __future__ import annotations

import numpy as np
import pytest

from ldpc import mod2

from minimal_ann_matrix import compute_tor_1


@pytest.mark.parametrize(
    "f_str, g_str, l, m",
    [
        ("1 + x", "1 + y", 3, 3),
        ("1 + x + x*y", "1 + y + x*y", 4, 4),
    ],
)
def test_torsion_trivial_cases(f_str: str, g_str: str, l: int, m: int) -> None:
    """When (I ∩ J) and (I·J) coincide the torsion should vanish."""

    tor_data = compute_tor_1(f_str, g_str, l, m)

    intersection_rank = mod2.rank(tor_data["intersection_matrix"])
    product_rank = mod2.rank(tor_data["product_matrix"])

    # Expect identical ranks and no torsion vectors returned.
    assert intersection_rank == product_rank
    assert tor_data["dimension"] == 0
    assert tor_data["tor_matrix"].size == 0


def test_torsion_nontrivial_span_properties() -> None:
    """The torsion basis should span (I ∩ J)/(I·J)."""

    f_str = "x^3 + y + y^2"
    g_str = "y^3 + x + x^2"
    l = m = 6

    tor_data = compute_tor_1(f_str, g_str, l, m)

    intersection = tor_data["intersection_matrix"]
    product = tor_data["product_matrix"]
    torsion = tor_data["tor_matrix"]

    intersection_rank = mod2.rank(intersection)
    product_rank = mod2.rank(product)
    torsion_rank = mod2.rank(torsion)

    # Dimension matches the rank difference between intersection and product.
    assert torsion_rank == tor_data["dimension"]
    assert torsion_rank == intersection_rank - product_rank

    # Each torsion vector lies inside the intersection span.
    for row in torsion:
        augmented = np.vstack([intersection, row])
        assert mod2.rank(augmented) == intersection_rank

    # And it contributes a new coset relative to the product span.
    for row in torsion:
        augmented = np.vstack([product, row])
        assert mod2.rank(augmented) > product_rank

    # Together with the product generators, torsion vectors rebuild the intersection span.
    combined_rank = mod2.rank(np.vstack([product, torsion]))
    assert combined_rank == intersection_rank
