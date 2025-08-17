
---

# Parallel and GPU-Accelerated Matrix Computations

This project contains two primary modules: a suite of tools for parallel matrix operations using Python's multiprocessing library, and a CUDA-based solver for differential equations, available in both a C++/CUDA and a Python/PyCUDA implementation.

## Table of Contents
1.  [Parallel Matrix Computations (Python)](#parallel-matrix-computations-python)
    *   [Overview](#overview-python)
    *   [Modules](#modules)
    *   [How to Run Tests](#how-to-run-tests)
2.  [GPU-Accelerated ODE Solver (CUDA)](#gpu-accelerated-ode-solver-cuda)
    *   [Overview](#overview-cuda)
    *   [Files](#files)
    *   [How to Run](#how-to-run)

---

## Parallel Matrix Computations (Python)

### Overview
This part of the project provides Python scripts that use the `multiprocessing` module to perform matrix multiplication and inversion in parallel. This is particularly effective for large matrices where computational tasks can be distributed across multiple CPU cores to improve performance.

### Modules
*   `parallel_matrix_multiplication.py`: Contains functions for performing matrix multiplication using both standard serial processing and parallel processing.
*   `parallel_matrix_inversion.py`: Provides functions for inverting matrices using serial and parallel methods based on Gauss-Jordan elimination.
*   `test_matrix_multiplication.py` & `test_matrix_inverse.py`: Unit tests to verify the correctness of the parallel and serial matrix operations.

### How to Run Tests
To ensure the implementations are correct, you can run the provided unit tests from the root directory of the project.

```bash
python -m unittest src/tests/test_matrix_multiplication.py
python -m unittest src/tests/test_matrix_inverse.py
```
---

## GPU-Accelerated ODE Solver (CUDA)

### Overview
This module is designed to solve second-order linear ordinary differential equations (ODEs) using the finite difference method, accelerated with NVIDIA CUDA. It demonstrates how to leverage GPU parallelism to speed up matrix inversion and matrix-vector multiplication, which are the core components of the solver.

### Files
*   **Python (PyCUDA) Implementation:**
    *   `solver.py`: A Python script that uses PyCUDA to dynamically compile and run CUDA kernels. It handles data initialization, orchestrates the GPU computations, and verifies the results.

*   **C++/CUDA Implementation:**
    *   `matrix_inversion.cu`: Contains the core CUDA C++ kernels for performing parallel matrix inversion using Gauss-Jordan elimination.
    *   `solver.cu`: The main CUDA C++ file that sets up the problem, manages GPU memory, and calls the kernels to solve the ODE.

### How to Run

#### Prerequisites:
*   NVIDIA GPU with CUDA support.
*   NVIDIA CUDA Toolkit (for `nvcc` compiler).
*   For the Python version: A Python environment with `numpy` and `pycuda`.

#### Execution:

1.  **Python (PyCUDA) Version:**
    Run the Python script from the command line. The script will compile the CUDA code within it, execute the solver, and print the results.
    ```bash
    python solver.py
    ```

2.  **C++/CUDA Version:**
    First, compile the `.cu` files using the `nvcc` compiler, then run the resulting executable.
    ```bash
    # Navigate to the CUDA solver directory
    cd cuda/differential_solver

    # Compile the solver
    nvcc solver.cu -o differential_solver

    # Run the executable
    ./differential_solver
    ```