"""Math-related helper functions."""

# stdlib
import struct


def dec_to_bin(d, width):
    """Convert interger to binary list representation.

    Args:
        d (int): Some decimal number
        width (int): Number of bits to use in representation

    Returns:
        b (list): Binary list representation
    """
    b = [int(x) for x in "{:0{size}b}".format(d, size=width)]

    return b


def bin_to_float(binary):
    """Converts binary string represenation to float.

    Args:
        binary (str): Some floating-point number

    Returns:
        f (str): Original float
    """

    f = struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

    return f


def bin_to_dec(b):
    """Convert binary list to decimal representation.

    Args:
        b (list): Binary list of bits

    Returns:
        d (int): Decimal representation
    """
    d = int("".join(str(x) for x in b), 2)

    return d


def float_to_bin(num):
    """Convert float to binary list representation.

    Args:
        num (float): Some floating-point number

    Returns:
        b (list): Binary list representation
    """
    b = []
    strBin = format(struct.unpack("!I", struct.pack("!f", num))[0], "032b")
    for bit in strBin:
        b.append(int(bit))

    return b


def clamp(x, mn, mx):
    """Clamp a number between an upper and lower bound.

    Args:
        x (num): A number.
        mn (num): Lower bound.
        mx (num): Upper bound.

    Returns:
        num: The clamped value.
    """
    return max(mn, min(x, mx))
