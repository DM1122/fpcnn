"""Compress indian pines dataset."""

# stdlib
import logging
from pathlib import Path

# external
import numpy as np

# project
from fpcnn import models
from fpcnn.libs import datalib

# region paths config
log_path = Path("logs/scripts")
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
data_shape = (145, 145, 220)  # x, y, lambda
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
data = datalib.load_data_hdf5(path="data/indian_pines.mat", header="indian_pines")
LOG.info(f"Original data ({data.shape}, {data.dtype}):\n{data}")
data = data[0 : data_shape[0], 0 : data_shape[1], 0 : data_shape[2]]
LOG.info(f"Cropped data ({data.shape}, {data.dtype}):\n{data}")

# run model
model = models.FPCNN(hp=hyperparams, logname="compress_indian_pines")
output = model.compress(data=data)
LOG.info(f"Output ({output.shape}, {output.dtype}):\n{output}")

# save output
output_path.mkdir(exist_ok=True)
np.save(file="output/indian_pines_compressed", arr=output)
