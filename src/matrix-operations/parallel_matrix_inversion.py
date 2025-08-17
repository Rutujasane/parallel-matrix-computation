"""
This file contains the functions to find the inverse of a matrix.
Using serial and parallel processing.
References:
https://cse.buffalo.edu/faculty/miller/Courses/CSE633/thanigachalam-Spring-2014-CSE633.pdf
"""

import multiprocessing
import numpy as np


def serial_matrix_inverse(matrix):
    """
    Compute the inverse of a matrix using python loops

    Args:
        matrix: matrix to be inverted.

    Returns:
        inverse: Inverse of the matrix.

    Example:
    >>> serial_matrix_inverse([[1, 2], [3, 4]])
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    """

    if len(matrix) != len(matrix[0]):
        raise ValueError("Non-square matrix")

    def swap_rows(matrix, i, j):    # To swap rows in case of 0 in diagonal
        temp = matrix[i].copy()
        matrix[i] = matrix[j]
        matrix[j] = temp

    n = len(matrix)

    error = None
    augmented_matrix = np.concatenate((matrix, np.identity(n)), axis=1, dtype=float)

    for i in range(n):
        if augmented_matrix[i][i] == 0.0:   # To check if diagonal element is 0
            error = 1
            for j in range(i+1, n):
                if augmented_matrix[j][i] != 0.0:
                    matrix = swap_rows(augmented_matrix, i, j)
                    error = 0
                    break

        if error == 1:  # If diagonal element is still 0 - Matrix is singular
            raise ValueError("Singular Matrix")

    # Perform row operations to make the matrix an identity matrix
    for i in range(n):
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i][i]

        for j in range(i+1, n):
            augmented_matrix[j] = augmented_matrix[j] - augmented_matrix[j][i] / augmented_matrix[i][i] * augmented_matrix[i]

    # Back substitution to make the matrix an identity matrix
    for i in range(n-1, 0, -1):
        for j in range(i-1, -1, -1):
            augmented_matrix[j] = augmented_matrix[j] - augmented_matrix[j][i] / augmented_matrix[i][i] * augmented_matrix[i]

    inverse = augmented_matrix[:, n:]
    return inverse


def parallel_row_operation(matrix, i, j, shape, lock):
    """
    Perform row operations to make the matrix an identity matrix.

    Args:
        matrix: matrix to be inverted.
        i: row index.
        j: row index.

    Returns:
        None

    Example:
    >>> parallel_row_operation(multiprocessing.Array('d', [1, 2, 3, 4]),
    0, 1, (2, 2), multiprocessing.Lock())
    """
    with lock:
        # Convert shared memory to numpy array
        matrix = np.frombuffer(matrix.get_obj()).reshape(shape)
        matrix[j] = matrix[j] - matrix[j][i] / matrix[i][i] * matrix[i]  # Perform row operation


def parallel_matrix_inverse(matrix):
    """
    Compute the inverse of a matrix using parallel processing.

    Args:
        matrix: matrix to be inverted.

    Returns:
        inverse: Inverse of the matrix.

    Example:
    >>> parallel_matrix_inverse([[1, 2], [3, 4]])
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    """

    if len(matrix) != len(matrix[0]):
        raise ValueError("Non-square matrix")

    def swap_rows(matrix, i, j):    # To swap rows in case of 0 in diagonal
        temp = matrix[i].copy()
        matrix[i] = matrix[j]
        matrix[j] = temp

    n = len(matrix)

    error = None
    augmented_matrix = np.concatenate((matrix, np.identity(n)), axis=1,
                                      dtype=float)

    for i in range(n):
        if augmented_matrix[i][i] == 0.0:   # To check if diagonal element is 0
            error = 1
            for j in range(i+1, n):
                if augmented_matrix[j][i] != 0.0:
                    matrix = swap_rows(augmented_matrix, i, j)
                    error = 0
                    break

        if error == 1:  # If diagonal element is still 0 - Matrix is singular
            raise ValueError("Singular Matrix")

    # shared augmented_matrix
    augmented_matrix = multiprocessing.Array('d', augmented_matrix.flatten())

    # Convert shared memory to numpy array
    augmented_matrix_np = np.frombuffer(augmented_matrix.get_obj()).reshape(n, 2*n)
    lock = multiprocessing.Lock()

    # Perform row operations to make the matrix an identity matrix
    for i in range(n):
        # Make diagonal element 1
        augmented_matrix_np[i] = augmented_matrix_np[i] / augmented_matrix_np[i][i]

        processes = []
        for j in range(i+1, n):   # Perform row operations in parallel
            process = multiprocessing.Process(target=parallel_row_operation,
                                              args=(augmented_matrix, i, j, augmented_matrix_np.shape, lock))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    # Back substitution to make the matrix an identity matrix
    processes = []
    for i in range(n-1, 0, -1):
        for j in range(i-1, -1, -1):
            process = multiprocessing.Process(target=parallel_row_operation,
                                              args=(augmented_matrix, i, j, augmented_matrix_np.shape, lock))
            processes.append(process)
            process.start()

    for process in processes:
        process.join()

    inverse = augmented_matrix_np[:, n:]    # Extract the inverse matrix
    return inverse