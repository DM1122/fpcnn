"""Utilities to load hyperspectral image data."""

# external
import h5py
import scipy.io


def load_data_hdf5(path, header):
    """Load hyperspectral image data from hdf5 files.

    Args:
        path (str): path to file
        header (str): the key to access the actual data from the file

    Returns:
        data (ndarray): array containing hyperspectral data
    """
    try:
        dataset = h5py.File(path, "r")
    except OSError:
        print("Dataset .mat file version < 7.3. Importing with scipy...")
        dataset = scipy.io.loadmat(path)

    data = dataset[header]

    return data
