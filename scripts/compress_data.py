"""Compress dataset."""

# stdlib
import logging
from pathlib import Path

# external
import numpy as np

# project
from fpcnn import encoding, models
from fpcnn.libs import benchmarklib, datalib

# region paths config
log_path = Path("logs/scripts")
data_path = Path("../data/indian_pines.mat")
output_path = Path("output")
# endregion

# region logging config
log_path.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=(log_path / Path(__file__).stem).with_suffix(".log"),
    filemode="w",
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
LOG = logging.getLogger(__name__)
# endregion

# region params
data_header = "indian_pines"
data_shape = (16, 16, 16)  # x, y, lambda
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
# endregion

# load data
LOG.info(f"Loading data from '{data_path}'")
data = datalib.load_data_hdf5(path=data_path, header=data_header)
data = data[0 : data_shape[0], 0 : data_shape[1], 0 : data_shape[2]]

# compress
model = models.FPCNN(hp=hyperparams, logname="compress_ip")

# get initial weights and biases to encode later
weights_biases = model.get_weights()


data_compressed = model.compress(data=data)


LOG.info(
    f"Data compressed ({data_compressed.shape}, {data_compressed.dtype}):\n"
    f"{data_compressed}"
)

# encode
data_mapped = encoding.map_residuals(data_compressed)

# Convert floats representing weights and biases to bit representation
toAppend = encoding.encode_weights_biases(weights_biases)

data_encoded = encoding.grc_encode(data=data_mapped, m=0)

# Append weights and biases to the encoded bitstream
data_encoded = np.append(data_encoded, toAppend)

# Reciver weights and biases from encoded bit stream
recoveredWB = encoding.decode_bitstream(data_encoded)

# Removed encoded weights & biases from bitstream to prepare bitstream for grc decoding
data_encoded = data_encoded[:-4848].copy()


bpc = benchmarklib.get_bpc_encoded(original=data, encoded=data_encoded)
print(f"BPC: {bpc}")

# sav
output_path.mkdir(exist_ok=True)
np.save(file="output/ip_encoded", arr=data_encoded)
