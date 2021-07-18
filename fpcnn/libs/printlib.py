"""Printing utilities."""

# external
import numpy as np

# project
from fpcnn.libs import benchmarklib


def get_tf_model_summary(model):
    """Gets a Tensorflow model's summary as a string to enable logger compatibility.

    Args:
        model (tensorflow.python.keras.engine.functional.Functional): Tensorflow model.

    Returns:
        string: Model summary.
    """
    print(type(model))
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))
    summary = "\n".join(lines)

    return summary


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
    print(f"BPC:\t{benchmarklib.get_bpc(a=array)}")
    print(f"Range:\t{array.min()},{array.max()} ({array.max()-array.min()})")
    print(f"Mean:\t{round(array.mean(),2)}")
    print(f"Median:\t{round(np.median(array),2)}")
    print(f"σ:\t{round(array.std(),2)}")
    print(f"σ²:\t{round(array.var(),2)}")

    if array.ndim == 1:
        print(f"{array[:5]}...{array[-5:]}")

    print()
