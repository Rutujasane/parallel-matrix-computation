"""
This file contains the function to compute multiplication of matrices.
"""

import multiprocessing
import numpy as np


def serial_matrix_multiply(matrix1, matrix2):
    """
    Compute the multiplication of matrices using serial processing.

    Args:
        matrix1, matrix2: two matrices to be multiplied.

    Returns:
        result: Resultant matrix after multiplication.

    Example:
    >>> serial_matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    [[19, 22], [43, 50]]
    """
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Incompatible dimensions.")

    result = [[0 for i in range(len(matrix2[0]))] for j in range(len(matrix1))]

    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result


def multiply_part(matrix1, matrix2, result, start, end, col):
    """
    Compute the multiplication of part of matrices.

    Args:
        matrix1: first matrix.
        matrix2: second matrix.
        result: the result matrix.
        start: start index.
        end: end index.
        col: number of columns of result matrix.
    Returns:
        None

    Example:
    >>> multiply_part([[1, 2], [3, 4]], [[5], [7]], [0, 0], 0, 1, 1)
    """
    result[col*start:col*end] = (np.dot(matrix1[start:end], matrix2)).flatten()


def parallel_matrix_multiply(matrix1, matrix2):
    """
    Compute the multiplication of matrices using parallel processing.

    Args:
        matrix1: first matrix.
        matrix2: second matrix.

    Returns:
        result: Resultant matrix after multiplication.

    Example:
    >>> parallel_matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    [[19, 22], [43, 50]]

    """
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Incompatible dimensions.")

    row = np.shape(matrix1)[0]
    col = np.shape(matrix2)[1]

    n_processes = multiprocessing.cpu_count()

    if row < n_processes:
        return serial_matrix_multiply(matrix1, matrix2)

    step = row//n_processes      # number of rows to be processed by each process

    # shared array for storing the result
    result = multiprocessing.Array('f', [0]*(row*col))

    processes = []
    for i in range(n_processes-1):     # for the first n-1 processes
        args = [matrix1, matrix2, result, i*step, (i+1)*step, col]
        p = multiprocessing.Process(target=multiply_part, args=args)
        p.start()
        processes.append(p)

    i = n_processes-1  # for the last process to not miss any rows
    args = [matrix1, matrix2, result, i*step, row, col]
    p = multiprocessing.Process(target=multiply_part, args=args)
    p.start()
    processes.append(p)

    for process in processes:
        process.join()

    # make a 2d array from the shared array of size row x col
    result = np.array(result[:]).reshape(row, col)
    return result
