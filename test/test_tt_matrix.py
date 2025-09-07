import unittest
import numpy as np

from trivariate_tricycle_codes import get_TT_matrix, get_TT_Hx_Hz, get_TT_Hmeta


def shift_mat(n: int, k: int) -> np.ndarray:
    return np.roll(np.eye(n, dtype=int), k % n, axis=1)


class TestTTMatrix(unittest.TestCase):
    def test_tt_matrix_construction(self):
        """Test TT individual matrix construction."""
        l, m, n = 2, 2, 2
        # Simple polynomial: x + y + z
        pairs = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        A = get_TT_matrix(pairs, l, m, n)

        expected = (
            np.kron(np.kron(shift_mat(l, 1), np.eye(m, dtype=int)), np.eye(n, dtype=int))
            + np.kron(np.kron(np.eye(l, dtype=int), shift_mat(m, 1)), np.eye(n, dtype=int))
            + np.kron(np.kron(np.eye(l, dtype=int), np.eye(m, dtype=int)), shift_mat(n, 1))
        ) % 2

        self.assertEqual(A.shape, (l * m * n, l * m * n))
        self.assertTrue(np.array_equal(A, expected))

    def test_tt_cross_terms(self):
        """Test TT matrix with cross terms xy, yz, xz."""
        l, m, n = 2, 2, 2
        # xy + yz + xz
        pairs = [(1, 1, 0), (0, 1, 1), (1, 0, 1)]
        A = get_TT_matrix(pairs, l, m, n)

        expected = (
            np.kron(np.kron(shift_mat(l, 1), shift_mat(m, 1)), np.eye(n, dtype=int))
            + np.kron(np.kron(np.eye(l, dtype=int), shift_mat(m, 1)), shift_mat(n, 1))
            + np.kron(np.kron(shift_mat(l, 1), np.eye(m, dtype=int)), shift_mat(n, 1))
        ) % 2

        self.assertEqual(A.shape, (l * m * n, l * m * n))
        self.assertTrue(np.array_equal(A, expected))

    def test_tt_hx_hz_shapes(self):
        """Test TT Hx and Hz matrix shapes."""
        l, m, n = 2, 2, 3
        a = [(1, 0, 0), (0, 1, 0)]  # x + y
        b = [(0, 0, 1), (1, 1, 0)]  # z + xy
        c = [(2, 0, 0), (0, 0, 2)]  # x^2 + z^2
        
        Hx, Hz = get_TT_Hx_Hz(a, b, c, l, m, n)
        lmn = l * m * n
        
        # Hx = [A | B | C] - horizontal concatenation of 3 blocks
        self.assertEqual(Hx.shape, (lmn, 3 * lmn))
        # Hz is 3x3 block matrix, so 3*lmn x 3*lmn
        self.assertEqual(Hz.shape, (3 * lmn, 3 * lmn))

    def test_tt_hz_block_structure(self):
        """Test TT Hz has correct 3x3 block structure."""
        l, m, n = 2, 2, 2
        a = [(1, 0, 0)]  # x
        b = [(0, 1, 0)]  # y
        c = [(0, 0, 1)]  # z
        
        Hx, Hz = get_TT_Hx_Hz(a, b, c, l, m, n)
        
        # Get individual matrices for comparison
        A = get_TT_matrix(a, l, m, n)
        B = get_TT_matrix(b, l, m, n)
        C = get_TT_matrix(c, l, m, n)
        
        lmn = l * m * n
        zero_block = np.zeros((lmn, lmn), dtype=np.uint8)
        
        # Check structure: [[C^T, 0, A^T], [0, C^T, B^T], [B^T, A^T, 0]]
        np.testing.assert_array_equal(Hz[:lmn, :lmn], C.T)           # Top-left: C^T
        np.testing.assert_array_equal(Hz[:lmn, lmn:2*lmn], zero_block) # Top-middle: 0
        np.testing.assert_array_equal(Hz[:lmn, 2*lmn:], A.T)        # Top-right: A^T
        
        np.testing.assert_array_equal(Hz[lmn:2*lmn, :lmn], zero_block) # Middle-left: 0
        np.testing.assert_array_equal(Hz[lmn:2*lmn, lmn:2*lmn], C.T)   # Middle-middle: C^T
        np.testing.assert_array_equal(Hz[lmn:2*lmn, 2*lmn:], B.T)    # Middle-right: B^T
        
        np.testing.assert_array_equal(Hz[2*lmn:, :lmn], B.T)        # Bottom-left: B^T
        np.testing.assert_array_equal(Hz[2*lmn:, lmn:2*lmn], A.T)    # Bottom-middle: A^T
        np.testing.assert_array_equal(Hz[2*lmn:, 2*lmn:], zero_block) # Bottom-right: 0

    def test_tt_hmeta_shape(self):
        """Test TT meta check matrix shape."""
        l, m, n = 2, 3, 2
        a = [(1, 0, 0)]  # x
        b = [(0, 1, 0)]  # y
        c = [(0, 0, 1)]  # z
        
        Hmeta = get_TT_Hmeta(a, b, c, l, m, n)
        lmn = l * m * n
        
        # Hmeta = [B^T | A^T | C^T] - horizontal concatenation
        self.assertEqual(Hmeta.shape, (lmn, 3 * lmn))

    def test_tt_binary_matrices(self):
        """Test that all TT matrices are binary (0 or 1)."""
        l, m, n = 2, 2, 2
        a = [(2, 1, 0), (1, 0, 2)]
        b = [(1, 1, 1), (0, 2, 0)]
        c = [(2, 0, 1), (0, 1, 2)]
        
        A = get_TT_matrix(a, l, m, n)
        B = get_TT_matrix(b, l, m, n)
        C = get_TT_matrix(c, l, m, n)
        
        # Check all matrices are binary
        self.assertTrue(np.all((A == 0) | (A == 1)))
        self.assertTrue(np.all((B == 0) | (B == 1)))
        self.assertTrue(np.all((C == 0) | (C == 1)))
        
        Hx, Hz = get_TT_Hx_Hz(a, b, c, l, m, n)
        self.assertTrue(np.all((Hx == 0) | (Hx == 1)))
        self.assertTrue(np.all((Hz == 0) | (Hz == 1)))
        
        Hmeta = get_TT_Hmeta(a, b, c, l, m, n)
        self.assertTrue(np.all((Hmeta == 0) | (Hmeta == 1)))

    def test_tt_ndarray_input(self):
        """Test TT matrix construction with ndarray input."""
        l, m, n = 2, 2, 2
        # Use ndarray instead of list of tuples
        pairs_array = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=int)
        A = get_TT_matrix(pairs_array, l, m, n)
        
        # Compare with list input
        pairs_list = [(1, 0, 0), (0, 1, 0), (1, 1, 1)]
        A_list = get_TT_matrix(pairs_list, l, m, n)
        
        self.assertTrue(np.array_equal(A, A_list))

    def test_tt_empty_polynomial(self):
        """Test TT matrix with empty polynomial (zero matrix)."""
        l, m, n = 2, 2, 2
        pairs = []  # Empty polynomial
        A = get_TT_matrix(pairs, l, m, n)
        
        expected = np.zeros((l * m * n, l * m * n), dtype=np.uint8)
        self.assertTrue(np.array_equal(A, expected))

    def test_tt_triple_kronecker_product(self):
        """Test that triple Kronecker product works correctly for single term."""
        l, m, n = 2, 3, 2
        # Single term: x^1 y^2 z^1
        pairs = [(1, 2, 1)]
        A = get_TT_matrix(pairs, l, m, n)
        
        # Manual computation of triple Kronecker product
        X = shift_mat(l, 1)  # x^1 shift matrix
        Y = shift_mat(m, 2)  # y^2 shift matrix  
        Z = shift_mat(n, 1)  # z^1 shift matrix
        expected = np.kron(np.kron(X, Y), Z) % 2
        
        self.assertTrue(np.array_equal(A, expected))

    def test_tt_dimension_consistency(self):
        """Test TT matrices have consistent dimensions for different l,m,n."""
        test_cases = [(2, 2, 2), (3, 2, 4), (2, 3, 3)]
        
        for l, m, n in test_cases:
            a = [(1, 0, 0)]
            b = [(0, 1, 0)] 
            c = [(0, 0, 1)]
            
            A = get_TT_matrix(a, l, m, n)
            Hx, Hz = get_TT_Hx_Hz(a, b, c, l, m, n)
            Hmeta = get_TT_Hmeta(a, b, c, l, m, n)
            
            lmn = l * m * n
            self.assertEqual(A.shape, (lmn, lmn))
            self.assertEqual(Hx.shape, (lmn, 3 * lmn))
            self.assertEqual(Hz.shape, (3 * lmn, 3 * lmn))
            self.assertEqual(Hmeta.shape, (lmn, 3 * lmn))


if __name__ == "__main__":
    unittest.main()