# stdlib
import logging
from pathlib import Path

# project
from fpcnn import gym

# region paths config
log_path = Path("logs/scripts")
data_path = Path("data/indian_pines.mat")
output_path = Path("output")
# endregion

# region logging config
log_path.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=(log_path / Path(__file__).stem).with_suffix(".log"),
    filemode="w",
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
LOG = logging.getLogger(__name__)
# endregion


LOG.info("Starting tuning")
tuner = gym.Tuner_FPCNN_GRC()

tuner.explore(36)
