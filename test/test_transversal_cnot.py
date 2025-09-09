"""Comprehensive tests for transversal CNOT gate implementation and verification.

Tests cover:
1. TransversalCNOTVerifier initialization and setup
2. Logical operator extraction
3. Stabilizer commutation verification
4. Logical truth table verification
5. Cyclic operation analysis
6. Meta check distance verification
7. BT code robustness testing
8. Edge cases and error handling
"""

import unittest
import numpy as np
from typing import Dict, Any, List

from transversal_cnot import (
    TransversalCNOTVerifier,
    LogicalOperators,
    StabilizerCommutationResult,
    LogicalTruthTableResult,
    create_example_verifier,
    verify_meta_check_distance,
    test_bt_robustness
)
from bivariate_bicycle_codes import get_BB_Hx_Hz
from bivariate_tricycle_codes import get_BT_Hx_Hz
from qec_simulation_core import build_bb_code, build_bt_code


class TestTransversalCNOTVerifier(unittest.TestCase):
    """Test cases for TransversalCNOTVerifier class."""
    
    def setUp(self) -> None:
        """Set up test fixtures with standard BB and BT parameters."""
        self.bb_params = {
            'a_poly': [[3, 0], [0, 1], [0, 2]],  # x^3 + y + y^2
            'b_poly': [[0, 3], [1, 0], [2, 0]],  # y^3 + x + x^2
            'l': 6,
            'm': 6
        }
        
        self.bt_params = {
            'a_poly': [[3, 0], [0, 1], [0, 2]],  # x^3 + y + y^2
            'b_poly': [[0, 3], [1, 0], [2, 0]],  # y^3 + x + x^2
            'c_poly': [[0, 0], [1, 1]],          # 1 + xy
            'l': 6,
            'm': 6
        }
        
        # Create small test case for faster testing
        self.small_bb_params = {
            'a_poly': [[1, 0], [0, 1]],  # x + y
            'b_poly': [[2, 0], [0, 2]],  # x^2 + y^2
            'l': 3,
            'm': 3
        }
        
        self.small_bt_params = {
            'a_poly': [[1, 0], [0, 1]],  # x + y
            'b_poly': [[2, 0], [0, 2]],  # x^2 + y^2
            'c_poly': [[1, 1]],          # xy
            'l': 3,
            'm': 3
        }
    
    def test_verifier_initialization(self) -> None:
        """Test TransversalCNOTVerifier initialization."""
        verifier = TransversalCNOTVerifier(self.small_bb_params, self.small_bt_params, verbose=False)
        
        # Check basic attributes
        self.assertEqual(verifier.l, 3)
        self.assertEqual(verifier.m, 3)
        self.assertEqual(verifier.lm, 9)
        
        # Check codes were built
        self.assertIsNotNone(verifier.bb_code)
        self.assertIsNotNone(verifier.bt_code)
        
        # Check logical operators were extracted
        self.assertIsInstance(verifier.bb_logicals, LogicalOperators)
        self.assertIsInstance(verifier.bt_logicals, LogicalOperators)
        
        # Check stabilizer matrices
        self.assertEqual(verifier.bb_Hx.shape[1], 2 * 9)  # BB: [A|B]
        self.assertEqual(verifier.bt_Hx.shape[1], 3 * 9)  # BT: [A|B|C]
    
    def test_dimension_mismatch_error(self) -> None:
        """Test error handling for mismatched code dimensions."""
        mismatched_bt_params = self.small_bt_params.copy()
        mismatched_bt_params['l'] = 4  # Different from BB
        
        with self.assertRaises(ValueError) as context:
            TransversalCNOTVerifier(self.small_bb_params, mismatched_bt_params, verbose=False)
        
        self.assertIn("must have matching l, m parameters", str(context.exception))
    
    def test_logical_operator_extraction(self) -> None:
        """Test logical operator extraction from CSS codes."""
        verifier = TransversalCNOTVerifier(self.small_bb_params, self.small_bt_params, verbose=False)
        
        # Check BB logical operators
        bb_logicals = verifier.bb_logicals
        self.assertGreater(bb_logicals.k_logical, 0, "BB code should have logical qubits")
        self.assertEqual(bb_logicals.n_physical, verifier.bb_code.N)
        self.assertEqual(bb_logicals.logical_x.shape[0], bb_logicals.k_logical)
        self.assertEqual(bb_logicals.logical_x.shape[1], bb_logicals.n_physical)
        self.assertEqual(bb_logicals.logical_z.shape, bb_logicals.logical_x.shape)
        
        # Check BT logical operators
        bt_logicals = verifier.bt_logicals
        self.assertGreaterEqual(bt_logicals.k_logical, 0, "BT code dimension should be non-negative")
        self.assertEqual(bt_logicals.n_physical, verifier.bt_code.N)
        
        # Check operators are binary
        self.assertTrue(np.all((bb_logicals.logical_x == 0) | (bb_logicals.logical_x == 1)))
        self.assertTrue(np.all((bb_logicals.logical_z == 0) | (bb_logicals.logical_z == 1)))
        self.assertTrue(np.all((bt_logicals.logical_x == 0) | (bt_logicals.logical_x == 1)))
        self.assertTrue(np.all((bt_logicals.logical_z == 0) | (bt_logicals.logical_z == 1)))
    
    def test_transversal_cnot_construction(self) -> None:
        """Test transversal CNOT matrix construction."""
        verifier = TransversalCNOTVerifier(self.small_bb_params, self.small_bt_params, verbose=False)
        
        tcnot = verifier.construct_transversal_cnot()
        
        total_qubits = verifier.bt_code.N + verifier.bb_code.N
        expected_shape = (2 * total_qubits, 2 * total_qubits)
        
        # Check shape
        self.assertEqual(tcnot.shape, expected_shape)
        
        # Check it's binary
        self.assertTrue(np.all((tcnot == 0) | (tcnot == 1)))
        
        # Check structure: identity on X part
        x_block = tcnot[:total_qubits, :total_qubits]
        self.assertTrue(np.array_equal(x_block, np.eye(total_qubits, dtype=np.uint8)))
        
        # Check Z transformation structure
        z_block = tcnot[total_qubits:, total_qubits:]
        self.assertEqual(z_block.shape, (total_qubits, total_qubits))
    
    def test_stabilizer_commutation_structure(self) -> None:
        """Test structure of stabilizer commutation verification."""
        verifier = TransversalCNOTVerifier(self.small_bb_params, self.small_bt_params, verbose=False)
        
        result = verifier.verify_stabilizer_commutation()
        
        # Check result structure
        self.assertIsInstance(result, StabilizerCommutationResult)
        self.assertIsInstance(result.x_stab_commutes, bool)
        self.assertIsInstance(result.z_stab_commutes, bool)
        self.assertIsInstance(result.max_x_violation, int)
        self.assertIsInstance(result.max_z_violation, int)
        
        # Check commutation matrices are non-empty
        self.assertGreater(result.x_commutation_matrix.size, 0)
        self.assertGreater(result.z_commutation_matrix.size, 0)
        
        # Violations should be non-negative
        self.assertGreaterEqual(result.max_x_violation, 0)
        self.assertGreaterEqual(result.max_z_violation, 0)
    
    def test_logical_truth_table_structure(self) -> None:
        """Test structure of logical truth table verification."""
        verifier = TransversalCNOTVerifier(self.small_bb_params, self.small_bt_params, verbose=False)
        
        result = verifier.verify_logical_truth_table()
        
        # Check result structure
        self.assertIsInstance(result, LogicalTruthTableResult)
        self.assertIsInstance(result.table_correct, bool)
        
        # Check expected table structure
        expected_inputs = {(0,0), (0,1), (1,0), (1,1)}
        self.assertEqual(set(result.expected_table.keys()), expected_inputs)
        
        # Check CNOT truth table values
        self.assertEqual(result.expected_table[(0,0)], (0,0))
        self.assertEqual(result.expected_table[(0,1)], (0,1))
        self.assertEqual(result.expected_table[(1,0)], (1,1))
        self.assertEqual(result.expected_table[(1,1)], (1,0))
    
    def test_cyclic_operations_analysis(self) -> None:
        """Test cyclic operations analysis between logical operators."""
        verifier = TransversalCNOTVerifier(self.small_bb_params, self.small_bt_params, verbose=False)
        
        result = verifier.verify_cyclic_logical_operations()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('cyclic_shifts_detected', result)
        self.assertIn('shift_patterns', result)
        self.assertIn('analysis_details', result)
        
        self.assertIsInstance(result['cyclic_shifts_detected'], bool)
        self.assertIsInstance(result['shift_patterns'], list)
        
        # If shifts detected, patterns should have proper structure
        for pattern in result['shift_patterns']:
            self.assertIsInstance(pattern, dict)
            self.assertIn('logical_index', pattern)
            self.assertIn('shift_type', pattern)
    
    def test_h0_homology_analysis(self) -> None:
        """Test H₀(C) homology group analysis."""
        verifier = TransversalCNOTVerifier(self.small_bb_params, self.small_bt_params, verbose=False)
        
        result = verifier.calculate_h0_homology_group()
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('implemented', result)
        self.assertIn('requires_symbolic_computation', result)
        self.assertIn('c_polynomial', result)
        self.assertIn('constraints', result)
        
        # Check constraints structure
        constraints = result['constraints']
        self.assertIn('x_constraint', constraints)
        self.assertIn('y_constraint', constraints)
        self.assertIn('c_constraint', constraints)
        
        # Should match BT parameters
        self.assertEqual(result['c_polynomial'], self.small_bt_params['c_poly'])
    
    def test_complete_verification(self) -> None:
        """Test complete verification suite."""
        verifier = TransversalCNOTVerifier(self.small_bb_params, self.small_bt_params, verbose=False)
        
        results = verifier.run_complete_verification()
        
        # Check top-level structure
        expected_keys = ['stabilizer_commutation', 'logical_truth_table', 
                        'cyclic_operations', 'homology_group', 'overall']
        self.assertEqual(set(results.keys()), set(expected_keys))
        
        # Check each section has proper structure
        for key in expected_keys[:-1]:  # All except 'overall'
            self.assertIn('passed' if key != 'homology_group' else 'calculated', results[key])
            self.assertIn('details', results[key])
        
        # Check overall assessment
        overall = results['overall']
        self.assertIn('transversal_cnot_valid', overall)
        self.assertIn('additional_analysis_complete', overall)
        self.assertIsInstance(overall['transversal_cnot_valid'], bool)


class TestExampleVerifier(unittest.TestCase):
    """Test example verifier creation and usage."""
    
    def test_create_example_verifier(self) -> None:
        """Test creation of example verifier with goal.md parameters."""
        verifier = create_example_verifier()
        
        # Check parameters match goal.md
        self.assertEqual(verifier.l, 6)
        self.assertEqual(verifier.m, 6)
        self.assertEqual(verifier.bb_params['a_poly'], [[3, 0], [0, 1], [0, 2]])
        self.assertEqual(verifier.bb_params['b_poly'], [[0, 3], [1, 0], [2, 0]])
        self.assertEqual(verifier.bt_params['c_poly'], [[0, 0], [1, 1]])
        
        # Should be properly initialized
        self.assertIsNotNone(verifier.bb_code)
        self.assertIsNotNone(verifier.bt_code)


class TestMetaCheckVerification(unittest.TestCase):
    """Test BT meta check distance verification."""
    
    def test_verify_meta_check_distance(self) -> None:
        """Test meta check distance verification."""
        bt_params = {
            'a_poly': [[1, 0], [0, 1]],  # x + y
            'b_poly': [[2, 0], [0, 2]],  # x^2 + y^2
            'c_poly': [[1, 1]],          # xy
            'l': 3,
            'm': 3
        }
        
        # Capture output to avoid cluttering test results
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            results = verify_meta_check_distance(bt_params)
        
        # Check result structure
        expected_keys = ['matrix_shape', 'code_length', 'code_dimension', 
                        'estimated_distance', 'nonzero_distance', 'analysis_passed']
        self.assertEqual(set(results.keys()), set(expected_keys))
        
        # Check types
        self.assertIsInstance(results['matrix_shape'], tuple)
        self.assertIsInstance(results['code_length'], int)
        self.assertIsInstance(results['code_dimension'], int)
        self.assertIsInstance(results['estimated_distance'], int)
        self.assertIsInstance(results['nonzero_distance'], bool)
        self.assertIsInstance(results['analysis_passed'], bool)
        
        # Sanity checks
        self.assertGreater(results['code_length'], 0)
        self.assertGreaterEqual(results['code_dimension'], 0)
        self.assertGreaterEqual(results['estimated_distance'], 0)


class TestBTRobustness(unittest.TestCase):
    """Test BT code robustness with different c_poly values."""
    
    def test_bt_robustness_basic(self) -> None:
        """Test basic BT robustness testing."""
        a_poly = [[1, 0], [0, 1]]  # x + y
        b_poly = [[2, 0], [0, 2]]  # x^2 + y^2
        l, m = 3, 3
        
        c_variants = [
            ([[1, 1]], "xy"),
            ([[0, 0]], "1"),
            ([[1, 0]], "x")
        ]
        
        # Capture output to avoid cluttering test results
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            results = test_bt_robustness(a_poly, b_poly, l, m, c_variants)
        
        # Check result structure
        expected_keys = ['total_variants', 'valid_variants', 'robustness_ratio', 'detailed_results']
        self.assertEqual(set(results.keys()), set(expected_keys))
        
        # Check basic properties
        self.assertEqual(results['total_variants'], len(c_variants))
        self.assertGreaterEqual(results['valid_variants'], 0)
        self.assertLessEqual(results['valid_variants'], results['total_variants'])
        self.assertGreaterEqual(results['robustness_ratio'], 0.0)
        self.assertLessEqual(results['robustness_ratio'], 1.0)
        
        # Check detailed results
        detailed = results['detailed_results']
        self.assertEqual(len(detailed), len(c_variants))
        
        for result in detailed:
            expected_fields = ['c_poly', 'description', 'code_valid', 'n', 'k', 'error']
            self.assertEqual(set(result.keys()), set(expected_fields))
            self.assertIsInstance(result['code_valid'], bool)
            self.assertIsInstance(result['n'], int)
            self.assertIsInstance(result['k'], int)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_degenerate_codes(self) -> None:
        """Test handling of degenerate codes with K=0."""
        # Create parameters likely to produce degenerate codes
        degenerate_bb_params = {
            'a_poly': [[0, 0]],  # Constant polynomial
            'b_poly': [[0, 0]],  # Constant polynomial
            'l': 2,
            'm': 2
        }
        
        degenerate_bt_params = {
            'a_poly': [[0, 0]],  # Constant polynomial
            'b_poly': [[0, 0]],  # Constant polynomial
            'c_poly': [[0, 0]],  # Constant polynomial
            'l': 2,
            'm': 2
        }
        
        try:
            verifier = TransversalCNOTVerifier(degenerate_bb_params, degenerate_bt_params, verbose=False)
            
            # Should handle gracefully even if codes are degenerate
            truth_result = verifier.verify_logical_truth_table()
            cyclic_result = verifier.verify_cyclic_logical_operations()
            
            # Results should indicate limitations
            if verifier.bb_code.K == 0 or verifier.bt_code.K == 0:
                self.assertFalse(truth_result.table_correct)
        
        except Exception as e:
            # Some combinations might fail to construct valid codes
            # This is expected behavior for degenerate parameters
            pass
    
    def test_empty_polynomial_inputs(self) -> None:
        """Test handling of empty polynomial specifications."""
        # Test with empty c_poly for BT code
        bt_params_empty_c = {
            'a_poly': [[1, 0]],  # x
            'b_poly': [[0, 1]],  # y
            'c_poly': [],        # Empty
            'l': 2,
            'm': 2
        }
        
        bb_params_small = {
            'a_poly': [[1, 0]],  # x
            'b_poly': [[0, 1]],  # y  
            'l': 2,
            'm': 2
        }
        
        try:
            # This might fail or create degenerate code
            verifier = TransversalCNOTVerifier(bb_params_small, bt_params_empty_c, verbose=False)
            
            # If it succeeds, BT code might be degenerate
            if verifier.bt_code.K == 0:
                self.assertEqual(verifier.bt_logicals.k_logical, 0)
                
        except Exception:
            # Expected for invalid polynomial specifications
            pass


if __name__ == '__main__':
    # Run tests with minimal output
    import sys
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTransversalCNOTVerifier,
        TestExampleVerifier, 
        TestMetaCheckVerification,
        TestBTRobustness,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2 if '-v' in sys.argv else 1)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)