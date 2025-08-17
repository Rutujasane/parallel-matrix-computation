import unittest
import numpy as np
from src.Matrix_operations.parallel_matrix_inversion import serial_matrix_inverse
from src.Matrix_operations.parallel_matrix_inversion import parallel_matrix_inverse


class TestMatrixInversion(unittest.TestCase):
    def test_series_inverse(self):
        a = [[1, 2], [3, 4]]
        r = [[-2.0, 1.0], [1.5, -0.5]]
        np.testing.assert_array_equal(serial_matrix_inverse(a), np.array(r))

        with np.testing.assert_raises(ValueError) as context:
            serial_matrix_inverse([[5, 7]])
        self.assertEqual(str(context.exception), "Non-square matrix")

        b = [[0, 6, 5], [0, 7, 8], [0, 7, 2]]
        with np.testing.assert_raises(ValueError) as context:
            serial_matrix_inverse(b)
        self.assertEqual(str(context.exception), "Singular Matrix")


    def test_parallel_matrix_inverse(self):
        a = [[1, 2], [3, 4]]
        r = [[-2.0, 1.0], [1.5, -0.5]]
        np.testing.assert_array_equal(parallel_matrix_inverse(a), np.array(r))

        with np.testing.assert_raises(ValueError) as context:
            parallel_matrix_inverse([[5, 7]])
        self.assertEqual(str(context.exception), "Non-square matrix")

        b = [[0, 6, 5], [0, 7, 8], [0, 7, 2]]
        with np.testing.assert_raises(ValueError) as context:
            parallel_matrix_inverse(b)
        self.assertEqual(str(context.exception), "Singular Matrix")


if __name__ == '__main__':
    unittest.main()