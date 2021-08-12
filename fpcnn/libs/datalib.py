"""Utilities to load hyperspectral image data."""

# stdlib
import logging

# external
import h5py
import scipy.io

LOG = logging.getLogger(__name__)


def load_data_hdf5(path, header):
    """Load hyperspectral image data from hdf5 files.

    Args:
        path (str): Path to file.
        header (str): The key to access the actual data from the file.

    Returns:
        numpy.ndarray: Hyperspectral datacube, typically with shape (x, y, lambda)
    """
    try:
        dataset = h5py.File(path, "r")
    except OSError:
        LOG.warning("Dataset .mat file version < 7.3. Importing with scipy")
        dataset = scipy.io.loadmat(path)

    LOG.debug(f"Dataset info:\nType: {type(dataset)}\nKeys: {[*dataset]}")

    data = dataset[header]

    return data
