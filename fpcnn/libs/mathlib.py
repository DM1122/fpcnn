"""Math-related helper functions."""


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


def bin_to_dec(b):
    """Convert binary list to decimal representation.

    Args:
        b (list): Binary list of bits

    Returns:
        d (int): Decimal representation
    """
    d = int("".join(str(x) for x in b), 2)

    return d


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
