"""Main testing module."""

# stdlib
import os
import random

# external
import numpy as np
import pipe
import tensorflow as tf
from libs import benchmarklib, datalib, plotlib

# project
import fpcnn

# region config
seed = 69
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

np.seterr(all="raise")
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


if __name__ == "__main__":
    # region load data
    data = datalib.load_data_hdf5(path="data/indian_pines.mat", header="indian_pines")
    data = data[0 : data_shape[0], 0 : data_shape[1], 0 : data_shape[2]]
    # data = np.random.randint(low=0, high=2048, size=data_shape, dtype="uint16")

    # printlib.print_ndarray_stats(data, title="Data")
    # plotlib.plot_band(data=data, band=0, title="Datacube")
    # plotlib.plot_traces(traces=[data], title="Data", dark=True)
    # plotlib.plot_distribution(data=data, title="Data Distribution", dark=True)
    # endregion

    # region predictive coding
    model_a = fpcnn.FPCNN(
        offsets_spatial=context_spatial_offsets,
        offsets_spectral=context_spectral_offsets,
    )
    weights = model_a.get_weights()
    # model_a.print_model()

    data_residuals, losses = model_a.encode(data=data)

    # printlib.print_ndarray_stats(data_residuals, title="Residuals")
    # plotlib.plot_band(data=data_residuals, band=0, title="Residuals")
    plotlib.plot_traces(
        traces=[data_residuals, losses], names=["Residuals", "Loss"], dark=True
    )
    # plotlib.plot_distribution(data=data_residuals, title="Residuals Distribution", dark=True)
    # endregion

    # region mapping
    data_flattened = data_residuals.flatten()
    data_mapped = pipe.map_residuals(data=data_flattened)

    # printlib.print_ndarray_stats(data_mapped, title="Mapped")
    # plotlib.plot_traces(traces=[data_mapped], title="Mapped", dark=True)
    # plotlib.plot_distribution(data=data_mapped, title="Mapped Distribution", dark=True)
    # endregion

    # region encoding
    data_encoded = pipe.grc_encode(data=data_mapped, m=grc_m)

    # printlib.print_ndarray_stats(data_encoded, title="Encoding")
    # plotlib.plot_traces(traces=[data_encoded], title="Encoding", dark=True)
    # plotlib.plot_distribution(data=data_encoded, title="Encoding Distribution", dark=True)
    # endregion

    # TRANSMISSION
    print()
    print("TRANSMISSION")
    print()

    # region decoding
    data_decoded = pipe.grc_decode(code=data_encoded, m=grc_m)

    # printlib.print_ndarray_stats(data_decoded, title="Decoding")
    # plotlib.plot_traces(traces=[data_decoded], title="Decoding", dark=True)
    # plotlib.plot_distribution(data=data_decoded, title="Decoding Distribution", dark=True)
    # endregion

    # region remapping
    data_remapped = pipe.remap_residuals(data=data_decoded)
    data_reshaped = data_remapped.reshape(data_shape)

    # printlib.print_ndarray_stats(data_remapped, title="Remapped")
    # plotlib.plot_traces(traces=[data_remapped], title="Remapped", dark=True)
    # plotlib.plot_distribution(data=data_remapped, title="Remapped Distribution", dark=True)
    # endregion

    # region predictive decoding
    model_b = fpcnn.FPCNN(
        offsets_spatial=context_spatial_offsets,
        offsets_spectral=context_spectral_offsets,
    )
    model_b.set_weights(weights=weights)
    data_recovered, losses = model_b.decode(data=data_reshaped)

    # printlib.print_ndarray_stats(data_recovered, title="Recovered")
    plotlib.plot_band(data=data_recovered, band=0, title="Recovered")
    # plotlib.plot_traces(traces=[data_recovered,losses], names=["Recovered","Loss"], dark=True)
    # plotlib.plot_distribution(data=data_recovered, title="Recovered Distribution", dark=True)
    # endregion

    # region benchmarking
    benchmarklib.print_error(
        A=data.flatten(), B=data_recovered.flatten(), title="Reconstruction"
    )

    benchmarklib.get_compression_factor(A=data, B=data_encoded)

    errors = benchmarklib.get_diff(A=data.flatten(), B=data_recovered.flatten())

    # plotlib.plot_traces(traces=[errors], title="Reconstruction Error", dark=True)
    # plotlib.plot_distribution(data=errors, title="Reconstruction Error Distribution", dark=True)
    # endregion

    input("Press Enter to continue...")
