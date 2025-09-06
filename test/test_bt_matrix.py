import unittest
import numpy as np

from bivariate_tricycle_codes import get_BT_matrix, get_BT_Hx_Hz, get_BT_Hmeta


def shift_mat(n: int, k: int) -> np.ndarray:
    return np.roll(np.eye(n, dtype=int), k % n, axis=1)


class TestBTMatrix(unittest.TestCase):
    def test_bt_matrix_construction(self):
        """Test BT individual matrix construction."""
        l, m = 3, 3
        # Simple polynomial: x + y
        pairs = [(1, 0), (0, 1)]
        A = get_BT_matrix(pairs, l, m)

        expected = (
            np.kron(shift_mat(l, 1), np.eye(m, dtype=int))
            + np.kron(np.eye(l, dtype=int), shift_mat(m, 1))
        ) % 2

        self.assertEqual(A.shape, (l * m, l * m))
        self.assertTrue(np.array_equal(A, expected))

    def test_bt_cross_term_xy(self):
        """Test BT matrix with cross term xy."""
        l, m = 4, 3
        # x^2 + xy + y^2
        pairs = [(2, 0), (1, 1), (0, 2)]
        A = get_BT_matrix(pairs, l, m)

        expected = (
            np.kron(shift_mat(l, 2), np.eye(m, dtype=int))
            + np.kron(shift_mat(l, 1), shift_mat(m, 1))
            + np.kron(np.eye(l, dtype=int), shift_mat(m, 2))
        ) % 2

        self.assertEqual(A.shape, (l * m, l * m))
        self.assertTrue(np.array_equal(A, expected))

    def test_bt_hx_hz_shapes(self):
        """Test BT Hx and Hz matrix shapes."""
        l, m = 3, 4
        a = [(1, 0), (0, 1)]  # x + y
        b = [(2, 0), (0, 2)]  # x^2 + y^2
        c = [(1, 1)]          # xy
        
        Hx, Hz = get_BT_Hx_Hz(a, b, c, l, m)
        lm = l * m
        
        # Hx = [A | B | C] - horizontal concatenation of 3 blocks
        self.assertEqual(Hx.shape, (lm, 3 * lm))
        # Hz is 3x3 block matrix, so 3*lm x 3*lm
        self.assertEqual(Hz.shape, (3 * lm, 3 * lm))

    def test_bt_hz_block_structure(self):
        """Test BT Hz has correct 3x3 block structure."""
        l, m = 2, 2
        a = [(1, 0)]  # x
        b = [(0, 1)]  # y  
        c = [(1, 1)]  # xy
        
        Hx, Hz = get_BT_Hx_Hz(a, b, c, l, m)
        
        # Get individual matrices for comparison
        A = get_BT_matrix(a, l, m)
        B = get_BT_matrix(b, l, m)
        C = get_BT_matrix(c, l, m)
        
        lm = l * m
        zero_block = np.zeros((lm, lm), dtype=np.uint8)
        
        # Check structure: [[C^T, 0, A^T], [0, C^T, B^T], [B^T, A^T, 0]]
        np.testing.assert_array_equal(Hz[:lm, :lm], C.T)           # Top-left: C^T
        np.testing.assert_array_equal(Hz[:lm, lm:2*lm], zero_block) # Top-middle: 0
        np.testing.assert_array_equal(Hz[:lm, 2*lm:], A.T)        # Top-right: A^T
        
        np.testing.assert_array_equal(Hz[lm:2*lm, :lm], zero_block) # Middle-left: 0
        np.testing.assert_array_equal(Hz[lm:2*lm, lm:2*lm], C.T)   # Middle-middle: C^T
        np.testing.assert_array_equal(Hz[lm:2*lm, 2*lm:], B.T)    # Middle-right: B^T
        
        np.testing.assert_array_equal(Hz[2*lm:, :lm], B.T)        # Bottom-left: B^T
        np.testing.assert_array_equal(Hz[2*lm:, lm:2*lm], A.T)    # Bottom-middle: A^T
        np.testing.assert_array_equal(Hz[2*lm:, 2*lm:], zero_block) # Bottom-right: 0

    def test_bt_hmeta_shape(self):
        """Test BT meta check matrix shape."""
        l, m = 3, 4
        a = [(1, 0)]  # x
        b = [(0, 1)]  # y
        c = [(1, 1)]  # xy
        
        Hmeta = get_BT_Hmeta(a, b, c, l, m)
        lm = l * m
        
        # Hmeta = [B^T | A^T | C^T] - horizontal concatenation
        self.assertEqual(Hmeta.shape, (lm, 3 * lm))

    def test_bt_binary_matrices(self):
        """Test that all BT matrices are binary (0 or 1)."""
        l, m = 3, 3
        a = [(2, 1), (1, 2)]
        b = [(1, 0), (0, 1)]
        c = [(2, 0), (0, 2)]
        
        A = get_BT_matrix(a, l, m)
        B = get_BT_matrix(b, l, m)
        C = get_BT_matrix(c, l, m)
        
        # Check all matrices are binary
        self.assertTrue(np.all((A == 0) | (A == 1)))
        self.assertTrue(np.all((B == 0) | (B == 1)))
        self.assertTrue(np.all((C == 0) | (C == 1)))
        
        Hx, Hz = get_BT_Hx_Hz(a, b, c, l, m)
        self.assertTrue(np.all((Hx == 0) | (Hx == 1)))
        self.assertTrue(np.all((Hz == 0) | (Hz == 1)))
        
        Hmeta = get_BT_Hmeta(a, b, c, l, m)
        self.assertTrue(np.all((Hmeta == 0) | (Hmeta == 1)))

    def test_bt_ndarray_input(self):
        """Test BT matrix construction with ndarray input."""
        l, m = 2, 3
        # Use ndarray instead of list of tuples
        pairs_array = np.array([[1, 0], [0, 1], [1, 1]], dtype=int)
        A = get_BT_matrix(pairs_array, l, m)
        
        # Compare with list input
        pairs_list = [(1, 0), (0, 1), (1, 1)]
        A_list = get_BT_matrix(pairs_list, l, m)
        
        self.assertTrue(np.array_equal(A, A_list))

    def test_bt_empty_polynomial(self):
        """Test BT matrix with empty polynomial (zero matrix)."""
        l, m = 2, 2
        pairs = []  # Empty polynomial
        A = get_BT_matrix(pairs, l, m)
        
        expected = np.zeros((l * m, l * m), dtype=np.uint8)
        self.assertTrue(np.array_equal(A, expected))


if __name__ == "__main__":
    unittest.main()