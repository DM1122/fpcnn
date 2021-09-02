"""Tests for encoding functions."""

# stdlib
import logging
import random

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

    assert np.all(data_mapped > 0), "Non-positive value found"


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


def test_encode_weights_biases():
    """Test for encoding weights and biases."""
    # pylint: disable=no-member
    weights_biases = []
    i = 0
    while i < random.randint(1, 10):
        weights_biases.append(
            np.random.rand(random.randint(1, 5), random.randint(1, 15))
        )
        i += 1
    LOG.info(
        f"Data (Number of np arrays: {len(weights_biases)}, {weights_biases[0].dtype}):\n{weights_biases}"  # pylint: disable=line-too-long
    )

    LOG.info("Encoding")
    data_encoded = encoding.encode_weights_biases(weights_biases)
    LOG.info(
        f"Encoded weights & biases ({data_encoded.shape}, {data_encoded.dtype}, Number of bits: {len(data_encoded)}):\n{data_encoded}"  # pylint: disable=line-too-long
    )


def test_decode_bitstream():
    """Test for decoding weights and biases."""

    rng = np.random.default_rng()
    data = rng.integers(low=0, high=2, size=128229198)
    LOG.info(f"Data ({data.shape}, {data.dtype}):\n{data}")

    LOG.info("Decoding")
    data_decoded = encoding.decode_bitstream(data)
    LOG.info(
        f"Data decoded ({len(data_decoded)}, {data_decoded[0].dtype}):\n{data_decoded}"
    )


@pytest.mark.star
def test_map_remap_residuals():
    """Test for residual mapping and remapping."""

    rng = np.random.default_rng()
    data = rng.integers(low=-32768, high=32767, size=32)
    LOG.info(f"Data ({data.shape}, {data.dtype}):\n{data}")

    # map
    LOG.info("Mapping residuals")
    data_mapped = encoding.map_residuals(data)
    LOG.info(f"Data mapped ({data_mapped.shape}, {data_mapped.dtype}):\n{data_mapped}")

    # remap
    LOG.info("Remapping residuals")
    data_remapped = encoding.remap_residuals(data_mapped)
    LOG.info(
        f"Data remapped ({data_remapped.shape}, {data_remapped.dtype}):\n"
        f"{data_remapped}"
    )

    assert np.array_equal(
        a1=data, a2=data_remapped
    ), "Data and remapped data are not equal"
