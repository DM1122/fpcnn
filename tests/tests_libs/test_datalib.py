"""Tests for datalib"""


# stdlib
import logging
from pathlib import Path

# project
from fpcnn.libs import datalib

LOG = logging.getLogger(__name__)


def test_load_data_hdf5():
    """Test data loader method for HDF5 files"""
    data_path = Path("data/indian_pines.mat")
    header = "indian_pines"

    LOG.info(f"Loading data from '{data_path}'")
    data = datalib.load_data_hdf5(path=data_path, header=header)
    LOG.info(f"Data ({data.shape}, {data.dtype}):\n{data}")
