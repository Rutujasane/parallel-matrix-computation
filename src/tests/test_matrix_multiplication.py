import unittest
import numpy as np
from src.Matrix_operations.parallel_matrix_multiplication import serial_matrix_multiply
from src.Matrix_operations.parallel_matrix_multiplication import parallel_matrix_multiply


class TestMatrixMultiplication(unittest.TestCase):
    def function_that_raises():
        raise ValueError('Incompatible dimensions.')

    def test_series_multiply(self):
        m1 = [[1, 2], [3, 4]]
        m2 = [[5, 6], [7, 8]]
        r = [[19, 22], [43, 50]]
        np.testing.assert_array_equal(serial_matrix_multiply(m1, m2), r)

        a = [[-1, -2], [-3, -4]]
        with self.assertRaises(ValueError) as context:
            serial_matrix_multiply(a, [[5]])
        self.assertEqual(str(context.exception), "Incompatible dimensions.")

        np.testing.assert_array_equal(serial_matrix_multiply([[0]], [[5, 7]]), ([[0, 0]]))

    def test_parallel_matrix_multiply(self):
        m1 = [[1, 2], [3, 4]]
        m2 = [[5, 6], [7, 8]]
        r = [[19, 22], [43, 50]]
        np.testing.assert_array_equal(parallel_matrix_multiply(m1, m2), r)

        a = [[-1, -2], [-3, -4]]
        with self.assertRaises(ValueError) as context:
            parallel_matrix_multiply(a, [[5]])
        self.assertEqual(str(context.exception), "Incompatible dimensions.")

        np.testing.assert_array_equal(parallel_matrix_multiply([[0]], [[5, 7]]), ([[0, 0]]))


if __name__ == '__main__':
    unittest.main()
    