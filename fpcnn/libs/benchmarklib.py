"""Benchmarking utilities."""

# external
import numpy as np


def get_acc(A, B):
    """
    Compute accuracy.

    Args:
        A (ndarray): array to be evaluated
        B (ndarray): array to be evaulated

    Returns:
        acc (float): percentage accuracy
    """
    assert type(A) == np.ndarray, f"Array is not ndarray (A:{type(A)})"
    assert type(B) == np.ndarray, f"Array is not ndarray (B:{type(B)})"
    assert A.size == B.size, f"Mistmatched array sizes (A:{A.size}, B:{B.size})"
    assert A.ndim == 1, f"Array dim is not 1 (A:{A.ndim})"
    assert B.ndim == 1, f"Array dim is not 1 (B:{B.ndim})"

    n = A.size
    score = 0

    for i in range(n):
        a = A[i].item()
        b = B[i].item()

        if a == b:
            score += 1

    acc = (score / n) * 100

    return acc


def get_mae(A, B):
    """
    Compute the mean absolute error.

    Args:
        A (ndarray): array to be evaluated
        B (ndarray): array to be evaulated

    Returns:
        mae (float): mean absolute error
    """
    assert type(A) == np.ndarray, f"Array is not ndarray (A:{type(A)})"
    assert type(B) == np.ndarray, f"Array is not ndarray (B:{type(B)})"
    assert A.size == B.size, f"Mistmatched array sizes (A:{A.size}, B:{B.size})"
    assert A.ndim == 1, f"Array dim is not 1 (A:{A.ndim})"
    assert B.ndim == 1, f"Array dim is not 1 (B:{B.ndim})"

    n = A.size
    error_sum = 0

    for i in range(n):
        a = A[i].item()
        b = B[i].item()

        error = abs(a - b)
        error_sum += error

    mae = error_sum / n

    return mae


def get_msae(A, B):
    """
    Compute the mean squared absolute error.

    Args:
        A (ndarray): array to be evaluated
        B (ndarray): array to be evaulated

    Returns:
        msae (float): mean squared absolute error
    """
    assert type(A) == np.ndarray, f"Array is not ndarray (A:{type(A)})"
    assert type(B) == np.ndarray, f"Array is not ndarray (B:{type(B)})"
    assert A.size == B.size, f"Mistmatched array sizes (A:{A.size}, B:{B.size})"
    assert A.ndim == 1, f"Array dim is not 1 (A:{A.ndim})"
    assert B.ndim == 1, f"Array dim is not 1 (B:{B.ndim})"

    n = A.size
    error_sum = 0

    for i in range(n):
        a = A[i].item()
        b = B[i].item()

        error = abs(a - b) ** 2
        error_sum += error

    msae = error_sum / n

    return msae


def get_mape(A, B):
    """
    Compute the mean absolute percentage error.

    Args:
        A (ndarray): expected values
        B (ndarray): observed values

    Returns:
        mape (float): mean squared absolute percentage error
    """
    assert type(A) == np.ndarray, f"Array is not ndarray (A:{type(A)})"
    assert type(B) == np.ndarray, f"Array is not ndarray (B:{type(B)})"
    assert A.size == B.size, f"Mistmatched array sizes (A:{A.size}, B:{B.size})"
    assert A.ndim == 1, f"Array dim is not 1 (A:{A.ndim})"
    assert B.ndim == 1, f"Array dim is not 1 (B:{B.ndim})"

    n = A.size
    error_sum = 0

    for i in range(n):
        a = A[i].item()
        b = B[i].item()

        error = abs(a - b) / a if a != 0 else 0
        error_sum += error

    mape = (error_sum / n) * 100

    return mape


def get_diff(A, B):
    """Compute difference between A and B.

    Args:
        A (ndarray): 1D truth array
        B (ndarray): 1D reference array

    Returns:
        errors (list): list containing difference sequence
    """
    assert type(A) == np.ndarray, f"Array is not ndarray (A:{type(A)})"
    assert type(B) == np.ndarray, f"Array is not ndarray (B:{type(B)})"
    assert A.size == B.size, f"Mistmatched array sizes (A:{A.size}, B:{B.size})"
    assert A.ndim == 1, f"Array dim is not 1 (A:{A.ndim})"
    assert B.ndim == 1, f"Array dim is not 1 (B:{B.ndim})"

    n = A.size
    diffs = np.zeros(shape=(n), dtype=np.float64, order="C")

    for i in range(n):
        a = A[i].item()
        b = B[i].item()

        diff = b - a
        diffs[i] = diff

    return diffs


def get_compression_factor(A, B):
    """Compute the compression factor between two arrays.

    Args:
        A (ndarray): Original array
        B (ndarray): Compressed array

    TODO:
        Return compression factor instead of printing
    """
    assert type(A) == np.ndarray, f"Array is not ndarray (A:{type(A)})"
    assert type(B) == np.ndarray, f"Array is not ndarray (B:{type(B)})"

    bpv_A = A.nbytes * 8 / A.size
    bpv_B = B.size / A.size

    cf = bpv_A / bpv_B

    print("Compression Stats:")
    print(f"BPV (A):\t{bpv_A}")
    print(f"BPV (B):\t{bpv_B}")
    print(f"CF:\t{cf}")


def print_error(A, B, title=""):
    """Print various error statistics.

    Args:
        A (ndarray): truth array
        B (ndarray): reference array
    """
    acc = get_acc(A=A, B=B)
    mae = get_mae(A=A, B=B)
    msae = get_msae(A=A, B=B)
    mape = get_mape(A=A, B=B)

    print(f"Error Measures: {title}")
    print(f"ACC:\t{acc}%")
    print(f"MAE:\t{mae}")
    print(f"MSAE:\t{msae}")
    print(f"MAPE:\t{mape}%")
    print()
