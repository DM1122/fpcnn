"""FPCNN pipeline functions."""

# external
import numpy as np
from libs import mathlib


def map_residuals(data):
    """Map residuals to positive values using overlap and interleave scheme.

    Args:
        data (ndarray): 1D ndarray of residuals

    Returns:
        output (ndarray): Mapped values
    """
    assert type(data) == np.ndarray, f"Array is not a numpy array ({type(data)})"
    assert data.ndim == 1, f"Array does not have ndim=1 ({data.ndim})"

    n = data.size
    output = np.zeros(shape=(n), dtype="uint16", order="C")

    for i in range(n):
        x = data[i].item()

        if x >= 0:
            x = 2 * x
        else:
            x = -2 * x - 1

        output[i] = x

    return output


def grc_encode(data, m):
    """Goulomb-Rice encoder.

    Each codeword is structured as
    <quotient code><remainder code>. The parameter 'm' defines
    via M=2^m the Golomb slope of the distribution, that is,
    the typical number of consecutive 0s expected in the encoding.
    Returns a self-delimiting variable length encoded bit sequence.
    Useful for sparse data with many 0s and few 1s.

    Args:
        data (ndarray): Array of intergers to be encoded
        m (int): Goulomb parameter

    Returns:
        code (ndarray): Encoded array of bits
    """
    M = 2 ** m

    code = []
    for n in data:
        # get quotient and remainder (can be implemented with bit masks in C)
        q, r = (n // M, n % M)

        # quotient code
        u = [1] * q  # unary encoding of q
        u.append(0)  # delimiter

        # remainder code
        c = 2 ** (m + 1) - M
        if r < c:  # some fancy logic (?)
            v = mathlib.dec_to_bin(d=r, width=m)
        elif r >= c:
            v = mathlib.dec_to_bin(d=r + c, width=m + 1)

        code += u + v  # array concatenation

    code = np.array(code, dtype="uint8", order="C")

    return code


def grc_decode(code, m):
    """Goulomb-Rice decoder.

    Each codeword is structured as <quotient code><remainder code>.
    The parameter 'm' defines via M=2^m the Golomb slope of the
    distribution, that is, the typical number of consecutive 0s expected
    in the encoding. Returns a self-delimiting variable length encoded
    bit sequence. Useful for sparse data with many 0s and few 1s.

    Args:
        data (ndarray): Bitstream to be decoded
        m (int): Goulomb parameter. Must match the value used at
        the encoder stage

    Returns:
        data (ndarray): Decoded values
    """
    M = 2 ** m

    data = []
    q = 0
    i = 0
    while i < len(code):
        if code[i] == 1:  # quotient code
            q += 1
            i += 1
        elif code[i] == 0:  # remainder code
            i += 1  # don't need to read delimiter
            v = code[i : i + m]
            r = mathlib.bin_to_dec(v)

            n = q * M + r
            data.append(n)

            # reset and move to next code word
            q = 0
            i += m

    data = np.array(data, dtype="uint16", order="C")

    return data


def remap_residuals(data):
    """Remap residuals by reversing overlap and interleave.

    Args:
        data (ndarray): Array of mapped values

    Returns:
        output (ndarray): Remapped values
    """
    assert type(data) == np.ndarray, f"Array is not a numpy array ({type(data)})"
    assert data.ndim == 1, f"Array does not have ndim=1 ({data.ndim})"

    n = data.size
    output = np.zeros(shape=(n), dtype="int16", order="C")

    for i in range(n):
        x = data[i].item()

        if x % 2 == 0:
            x = x / 2
        else:
            x = (x + 1) / -2

        output[i] = x

    return output
