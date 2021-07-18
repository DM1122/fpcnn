"""Benchmarking utilities."""

# external
import numpy as np


def get_mae(A, B):
    """
    Compute the mean absolute error.

    Args:
        A (ndarray): array to be evaluated
        B (ndarray): array to be evaulated

    Returns:
        mae (float): mean absolute error
    """
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


def get_bpc(a):
    """Calculates the bits per component of an array. BPC is a measure of information
        density.

    Args:
        a (numpy.ndarray): Data array.

    Returns:
        float: Bits per component of array.
    """
    bpc = a.nbytes * 8 / a.size
    return bpc


def get_bpc_bitstream(a, size):
    """For a given bitstream, computes the bits per component in accordance to its
        original number of components (size).

    Args:
        a (ndarray): An array of ones and zeros.
        size (int): Original number of components in bitstream.

    Returns:
        bpc (float): bits per component
    """
    bpc = a.size / size
    return bpc


def get_cr(a, b):
    """Get compression ratio between a compressed array and the original.

    Args:
        a (numpy.ndarray): Uncompressed data.
        b (numpy.ndarray): Compressed data.

    Returns:
        float: Compression ratio.
    """
    cr = a.nbytes / b.nbytes
    return cr


def get_cr_bitstream(a, b):
    """Calculates the compression ratio between a compressed bitstream and the orignal
        data.


    Args:
        a (numpy.ndarray): Uncompressed data.
        b (list): A python list of ones and zeros.

    Returns:
        float: Compression ratio.
    """
    cr = a.nbytes * 8 / b.size
    return cr
