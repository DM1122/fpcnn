"""Printing utilities."""

import numpy as np
import progress
import progress.spinner
# from progress.spinner import PieSpinner


def print_ndarray_stats(array, title=""):
    """Print array statistics.

    Args:
        array (ndarray): Data array
        title (str, optional): Title. Defaults to None.
    """
    # title = "" if title is None else title
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
    print(f"{array[:5]}...{array[-5:]}") if array.ndim == 1 else None
    print()



class ProgressBar(progress.bar.Bar):
    suffix = "%(index)d/%(max)d (%(percent)d%%) - %(freq)d it/s - %(elapsed)ds:%(total)ds"

    @property
    def total(self):
        return self.elapsed + self.eta
    
    @property
    def freq(self):
        return 1 / self.avg


class ProgressSpinner(progress.spinner.PieSpinner):
    suffix = "%(freq)d it/s - %(elapsed)ds"