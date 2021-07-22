"""Tests for encoding functions."""

# stdlib
import logging

# external
import numpy as np
import pytest

# project
from fpcnn import encoding

LOG = logging.getLogger(__name__)


def test_map_residuals():
    """Test for residual mapping."""

    rng = np.random.default_rng()
    data = rng.integers(low=-32768, high=32767, size=32)
    LOG.info(f"Data ({data.shape}, {data.dtype}):\n{data}")

    LOG.info("Mapping residuals")
    data_mapped = encoding.map_residuals(data)
    LOG.info(f"Data mapped ({data_mapped.shape}, {data_mapped.dtype}):\n{data_mapped}")

    assert np.all(data_mapped < 0) is False, "Non-positive value found"


def test_grc_encode():
    """Test for Goulomb Rice Code encoding."""

    rng = np.random.default_rng()
    data = rng.integers(low=0, high=64, size=8)
    LOG.info(f"Data ({data.shape}, {data.dtype}):\n{data}")

    LOG.info("Encoding")
    data_encoded = encoding.grc_encode(data, m=2)
    LOG.info(
        f"Data encoded ({data_encoded.shape}, {data_encoded.dtype}):\n{data_encoded}"
    )


def test_grc_decode():
    """Test for Goulomb Rice Code decoding."""

    rng = np.random.default_rng()
    data = rng.integers(low=0, high=1, size=32)
    LOG.info(f"Data ({data.shape}, {data.dtype}):\n{data}")

    LOG.info("Decoding")
    data_decoded = encoding.grc_decode(data, m=2)
    LOG.info(
        f"Data decoded ({data_decoded.shape}, {data_decoded.dtype}):\n{data_decoded}"
    )


@pytest.mark.star
def test_remap_residuals():
    """Test for residual remapping."""

    rng = np.random.default_rng()
    data = rng.integers(low=0, high=65535, size=32)
    LOG.info(f"Data ({data.shape}, {data.dtype}):\n{data}")

    LOG.info("Remapping residuals")
    data_remapped = encoding.remap_residuals(data)
    LOG.info(
        f"Data remapped ({data_remapped.shape}, {data_remapped.dtype}):\n"
        f"{data_remapped}"
    )
