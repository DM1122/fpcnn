"""Main testing."""
# stdlib
import os
import random

# external
import numpy as np
import tensorflow as tf

# project
from fpcnn import core, pipe
from fpcnn.libs import benchmarklib, datalib


def test_main():
    """Test entire FPCNN framework."""
    # region config
    seed = 69
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    # endregion

    # region params
    data_shape = (10, 10, 10)
    grc_m = 10
    context_spatial_offsets = [
        [-1, 0, 0],
        [-1, -1, 0],
        [0, -1, 0],
        [1, -1, 0],
        [-1, 0, -1],
        [-1, -1, -1],
        [0, -1, -1],
        [1, -1, -1],
        [-1, 0, -2],
        [-1, -1, -2],
        [0, -1, -2],
        [1, -1, -2],
    ]
    context_spectral_offsets = [[0, 0, -1], [0, 0, -2], [0, 0, -3], [0, 0, -4]]
    # endregion

    # region load data
    data = datalib.load_data_hdf5(path="data/indian_pines.mat", header="indian_pines")
    data = data[0 : data_shape[0], 0 : data_shape[1], 0 : data_shape[2]]
    # endregion

    # region predictive coding
    model_a = core.FPCNN(
        offsets_spatial=context_spatial_offsets,
        offsets_spectral=context_spectral_offsets,
    )
    weights = model_a.get_weights()

    data_residuals, _ = model_a.encode(data=data)

    # region mapping
    data_flattened = data_residuals.flatten()
    data_mapped = pipe.map_residuals(data=data_flattened)
    # endregion

    # region encoding
    data_encoded = pipe.grc_encode(data=data_mapped, m=grc_m)
    # endregion

    # region decoding
    data_decoded = pipe.grc_decode(code=data_encoded, m=grc_m)
    # endregion

    # region remapping
    data_remapped = pipe.remap_residuals(data=data_decoded)
    data_reshaped = data_remapped.reshape(data_shape)
    # endregion

    # region predictive decoding
    model_b = core.FPCNN(
        offsets_spatial=context_spatial_offsets,
        offsets_spectral=context_spectral_offsets,
    )
    model_b.set_weights(weights=weights)
    data_recovered, _ = model_b.decode(data=data_reshaped)
    # endregion

    # region benchmarking
    accuracy = benchmarklib.get_acc(A=data.flatten(), B=data_recovered.flatten())
    # endregion

    assert accuracy == 1
