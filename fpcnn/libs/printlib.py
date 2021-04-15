"""Printing utilities."""

# external
import numpy as np
import progress.spinner
from progress.bar import Bar


def print_ndarray_stats(array, title=""):
    """Print array statistics.

    Args:
        array (ndarray): Data array
        title (str, optional): Title. Defaults to None.
    """
    print(f"Ndarray Stats: {title}")
    print(f"Shape:\t{array.shape}")
    print(f"Size:\t{array.size}")
    print(f"Dims:\t{array.ndim}")
    print(f"Type:\t{array.dtype}")
    print(f"Bytes:\t{array.nbytes} ({round(array.nbytes*10**-6,2)}MB)")
    print(f"Range:\t{array.min()},{array.max()} ({array.max()-array.min()})")
    print(f"Mean:\t{round(array.mean(),2)}")
    print(f"Median:\t{round(np.median(array),2)}")
    print(f"σ:\t{round(array.std(),2)}")
    print(f"σ²:\t{round(array.var(),2)}")

    if array.ndim == 1:
        print(f"{array[:5]}...{array[-5:]}")

    print()


class ProgressBar(Bar):
    """Custom progressbar class."""

    suffix = (
        "%(index)d/%(max)d (%(percent)d%%) - %(freq)d it/s - %(elapsed)ds:%(total)ds"
    )

    @property
    def total(self):
        """Progress bar total remaining time property."""
        return self.elapsed + self.eta

    @property
    def freq(self):
        """Progress bar frequency property."""
        return 1 / self.avg


class ProgressSpinner(progress.spinner.PieSpinner):
    """Custom progress spinner class."""

    suffix = "%(freq)d it/s - %(elapsed)ds"
