"""Model testing."""

# stdlib
import logging

# external
import numpy as np
import pytest

# project
from fpcnn import models

LOG = logging.getLogger(__name__)


def test_fpcnn_instantiate_model():
    """Test the FPCNN class model instantiation method."""

    hyperparams = {
        "track_spatial_length": 1,
        "track_spatial_width": 5,
        "track_spectral_length": 1,
        "track_spectral_width": 5,
        "track_fusion_length": 1,
        "track_fusion_width": 5,
        "lr": 0.01,
        "context_offsets_spatial": [
            (-1, 0, 0),
            (-1, -1, 0),
            (0, -1, 0),
            (1, -1, 0),
            (-1, 0, -1),
            (-1, -1, -1),
            (0, -1, -1),
            (1, -1, -1),
            (-1, 0, -2),
            (-1, -1, -2),
            (0, -1, -2),
            (1, -1, -2),
        ],
        "context_offsets_spectral": [(0, 0, -1), (0, 0, -2), (0, 0, -3), (0, 0, -4)],
    }

    models.FPCNN(hp=hyperparams, logname="test_init")


def test_fpcnn_get_context():
    """Test the FPCNN class context retrieval function."""
    data_shape = (8, 8, 8)  # x, y, lambda
    offsets = [
        # first band
        (-1, -1, -1),
        (0, -1, -1),
        (1, -1, -1),
        (-1, 0, -1),
        (0, 0, -1),
        (1, 0, -1),
        (-1, 1, -1),
        (0, 1, -1),
        (1, 1, -1),
        # second band
        (-1, -1, 0),
        (0, -1, 0),
        (1, -1, 0),
        (-1, 0, 0),
        (0, 0, 0),  # index
        (1, 0, 0),
        (-1, 1, 0),
        (0, 1, 0),
        (1, 1, 0),
        # third band
        (-1, -1, 1),
        (0, -1, 1),
        (1, -1, 1),
        (-1, 0, 1),
        (0, 0, 1),
        (1, 0, 1),
        (-1, 1, 1),
        (0, 1, 1),
        (1, 1, 1),
    ]

    rng = np.random.default_rng()

    index = rng.integers(low=0, high=data_shape, size=3)
    LOG.info(f"Index: {index}")

    data = rng.integers(low=0, high=16383, size=data_shape)
    LOG.info(f"Data ({data.shape}, {data.dtype}):\n{data}")

    LOG.info("Retrieving context")
    context = models.FPCNN._get_context(data=data, index=index, offsets=offsets)
    LOG.info(f"Context ({context.shape}, {context.dtype}):\n{context}")


def test_fpcnn_build_batch():
    """Test the FPCNN class batch building function."""
    data_shape = (8, 8, 8)  # x, y, lambda
    offsets_spatial = [
        (-1, 0, 0),
        (-1, -1, 0),
        (0, -1, 0),
        (1, -1, 0),
        (-1, 0, -1),
        (-1, -1, -1),
        (0, -1, -1),
        (1, -1, -1),
        (-1, 0, -2),
        (-1, -1, -2),
        (0, -1, -2),
        (1, -1, -2),
    ]

    offsets_spectral = [(0, 0, -1), (0, 0, -2), (0, 0, -3), (0, 0, -4)]

    rng = np.random.default_rng()

    band = rng.integers(low=0, high=data_shape[2], size=1).item()
    LOG.info(f"Band ({type(band)}): {band}")

    data = rng.integers(low=0, high=16383, size=data_shape)
    LOG.info(f"Data ({data.shape}, {data.dtype}):\n{data}")

    LOG.info("Building batch")
    batch_spatial, batch_spectral, batch_labels = models.FPCNN._build_batch(
        data=data,
        band=band,
        offsets_spatial=offsets_spatial,
        offsets_spectral=offsets_spectral,
    )
    LOG.info(
        f"Batch spatial features ({batch_spatial.shape}, {batch_spatial.dtype}):\n"
        f"{batch_spatial}"
    )
    LOG.info(
        f"Batch spectral features ({batch_spectral.shape}, {batch_spectral.dtype}):\n"
        f"{batch_spectral}"
    )
    LOG.info(
        f"Batch labels ({batch_labels.shape}, {batch_labels.dtype}):\n"
        f"{batch_labels}"
    )


def test_fpcnn_compress():
    """Test the FPCNN class compression function."""

    data_shape = (8, 8, 8)  # x, y, lambda
    hyperparams = {
        "track_spatial_length": 1,
        "track_spatial_width": 5,
        "track_spectral_length": 1,
        "track_spectral_width": 5,
        "track_fusion_length": 1,
        "track_fusion_width": 5,
        "lr": 0.01,
        "context_offsets_spatial": [
            (-1, 0, 0),
            (-1, -1, 0),
            (0, -1, 0),
            (1, -1, 0),
            (-1, 0, -1),
            (-1, -1, -1),
            (0, -1, -1),
            (1, -1, -1),
            (-1, 0, -2),
            (-1, -1, -2),
            (0, -1, -2),
            (1, -1, -2),
        ],
        "context_offsets_spectral": [(0, 0, -1), (0, 0, -2), (0, 0, -3), (0, 0, -4)],
    }

    rng = np.random.default_rng()

    data = rng.integers(low=0, high=16383, size=data_shape)
    LOG.info(f"Data ({data.shape}, {data.dtype}):\n{data}")

    model = models.FPCNN(hp=hyperparams, logname="test_compress_toy")

    output = model.compress(data=data)
    LOG.info(f"Output ({output.shape}, {output.dtype}):\n{output}")


@pytest.mark.skip("Not implemented")
def test_fpccn_decompress():
    """Test the FPCNN class decompression function."""

    data_shape = (8, 8, 8)  # x, y, lambda
    hyperparams = {
        "track_spatial_length": 1,
        "track_spatial_width": 5,
        "track_spectral_length": 1,
        "track_spectral_width": 5,
        "track_fusion_length": 1,
        "track_fusion_width": 5,
        "lr": 0.01,
        "context_offsets_spatial": [
            (-1, 0, 0),
            (-1, -1, 0),
            (0, -1, 0),
            (1, -1, 0),
            (-1, 0, -1),
            (-1, -1, -1),
            (0, -1, -1),
            (1, -1, -1),
            (-1, 0, -2),
            (-1, -1, -2),
            (0, -1, -2),
            (1, -1, -2),
        ],
        "context_offsets_spectral": [(0, 0, -1), (0, 0, -2), (0, 0, -3), (0, 0, -4)],
    }

    rng = np.random.default_rng()

    data = rng.integers(low=0, high=16383, size=data_shape)
    LOG.info(f"Data ({data.shape}, {data.dtype}):\n{data}")

    model = models.FPCNN(hp=hyperparams, logname="test_decompress_toy")

    output = model.decompress(data=data)
    LOG.info(f"Output ({output.shape}, {output.dtype}):\n{output}")
