"""FPCNN encoding functions."""

# stdlib
import logging

# external
import numpy as np

# project
from fpcnn.libs import mathlib

LOG = logging.getLogger(__name__)


def map_residuals(data):
    """Map negative and positive residuals to positive values using overlap and
        interleave scheme.

    Args:
        data (numpy.ndarray): Array of residuals.

    Returns:
        numpy.ndarray: Mapped positive values.
    """
    return np.where(data >= 0, 2 * data, -2 * data - 1)


def grc_encode(data, m):
    """Goulomb-Rice encoding.

    Each codeword is structured as
    <quotient code><remainder code>. The parameter 'm' defines
    via M=2^m the Golomb slope of the distribution, that is,
    the typical number of consecutive 0s expected in the encoding.
    Returns a self-delimiting variable length encoded bit sequence.
    Useful for sparse data with many 0s and few 1s.

    Args:
        data (numpy.ndarray): Array of positive intergers to be encoded.
        m (int): Goulomb parameter.

    Returns:
        numpy.ndarray: 1D Encoded array of bits.
    """
    M = 2 ** m

    data_flat = data.flatten()

    code = []
    for n in data_flat:
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

    code = np.array(object=code, dtype="uint8")

    return code


def grc_decode(code, m):
    """Goulomb-Rice decoding.

    Each codeword is structured as <quotient code><remainder code>.
    The parameter 'm' defines via M=2^m the Golomb slope of the
    distribution, that is, the typical number of consecutive 0s expected
    in the encoding. Returns a self-delimiting variable length encoded
    bit sequence. Useful for sparse data with many 0s and few 1s.

    Args:
        data (numpy.ndarray): Bitstream to be decoded.
        m (int): Goulomb parameter. Must match the value used at
            the encoder stage.

    Returns:
        numpy.ndarray: Decoded 1D array of values.
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

    data = np.array(data)

    return data


def remap_residuals(data):
    """Remap residuals by undoing overlap and interleave.

    Args:
        data (numpy.ndarray): Array of positive mapped values.

    Returns:
        numpy.ndarray: Remapped negative and positive values.
    """
    return np.where(data % 2 == 0, data / 2, (data + 1) / -2)
