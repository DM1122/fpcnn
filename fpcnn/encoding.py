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


def encode_weights_biases(weights_biases):
    """Encodes weights and biases to end of bitstream

    Args:
        weights_biases (list{ndarray}): List of ndarrays defining weights.

    Returns:
        toAppend numpy.ndarray: numpy.ndarray: 1D Encoded array of bits..
    """

    floating = []

    for a in weights_biases:
        for b in a:
            if isinstance(b, np.float32):
                floating.append(mathlib.float_to_bin(b))

            else:
                for c in b:
                    floating.append(mathlib.float_to_bin(c))
    # Convert long integers into 1s and 0s
    toAppend = []
    for i in floating:
        for bit in i:
            toAppend.append(int(bit))

    # Create array of bits representing weights and biases to append to bitstream
    toAppend = np.array(toAppend)
    toAppend = toAppend.astype(np.uint8)
    # Get length of bits that represent Ws & Bs,
    # PROBABLY UNEEDED
    lenOfwb = len(toAppend)
    lenOfwb = mathlib.dec_to_bin(lenOfwb, 16)
    lenOfwb = np.array(lenOfwb)
    lenOfwb = lenOfwb.astype(np.uint8)
    # Append length of Ws and Bs to the bit representation of Ws and Bs
    toAppend = np.append(toAppend, lenOfwb)
    return toAppend


def decode_bitstream(stream):
    """Removes weights and biases from end of bitstream

    Args:
        stream (numpy.ndarray): Array of encoded cube, weights and biases.

    Returns:
        weights (list{ndarray}): List of ndarrays defining weights.
    """
    # TODO: restore origianl bitstream to be decoded back into cube
    lengthOfWB = ""
    for i in range(len(stream) - 16, len(stream)):

        lengthOfWB += str(stream[i])

    lengthOfWB = mathlib.bin_to_dec(lengthOfWB)

    f = ""

    a = np.empty(shape=(12, 5), dtype="float32")
    b = np.empty(shape=(5), dtype="float32")
    c = np.empty(shape=(4, 5), dtype="float32")
    d = np.empty(shape=(5), dtype="float32")
    e = np.empty(shape=(10, 5), dtype="float32")
    f = np.empty(shape=(5), dtype="float32")
    g = np.empty(shape=(5, 1), dtype="float32")
    h = np.empty(shape=(1), dtype="float32")
    weights = [a, b, c, d, e, f, g, h]

    currentBit = len(stream) - 4848
    for arr in weights:
        for l in range(0, arr.shape[0]):
            if arr.ndim > 1:
                for e in range(0, arr.shape[1]):
                    f = ""
                    for bit in range(currentBit, currentBit + 32):
                        f += str(stream[bit])
                    currentBit += 32
                    arr[l, e] = np.float32(float(mathlib.bin_to_float(f)))
            else:
                f = ""
                for bit in range(currentBit, currentBit + 32):
                    f += str(stream[bit])
                currentBit += 32
                arr[l] = np.float32(float(mathlib.bin_to_float(f)))

    return weights
