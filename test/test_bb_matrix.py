import unittest
import numpy as np

from bivariate_bicycle_codes import get_BB_matrix, get_BB_Hx_Hz


def shift_mat(n: int, k: int) -> np.ndarray:
    return np.roll(np.eye(n, dtype=int), k % n, axis=1)


class TestBBMatrix(unittest.TestCase):
    def test_cross_term_xy(self):
        l, m = 4, 5
        # x^2 + x y + y^2
        pairs = [(2, 0), (1, 1), (0, 2)]
        A = get_BB_matrix(pairs, l, m)

        expected = (
            np.kron(shift_mat(l, 2), np.eye(m, dtype=int))
            + np.kron(shift_mat(l, 1), shift_mat(m, 1))
            + np.kron(np.eye(l, dtype=int), shift_mat(m, 2))
        ) % 2

        self.assertEqual(A.shape, (l * m, l * m))
        self.assertTrue(np.array_equal(A, expected))

    def test_hx_hz_shapes(self):
        l, m = 3, 4
        a = [(1, 0), (0, 1)]  # x + y
        b = [(2, 0), (0, 2)]  # x^2 + y^2
        Hx, Hz = get_BB_Hx_Hz(a, b, l, m)
        lm = l * m
        # Hx = [A | B], Hz = [B^T | A^T]
        self.assertEqual(Hx.shape, (lm, 2 * lm))
        self.assertEqual(Hz.shape, (lm, 2 * lm))


if __name__ == "__main__":
    unittest.main()
