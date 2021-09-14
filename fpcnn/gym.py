"""A gym for getting those models in shape."""

# stdlib
import logging

# import os
import winsound
from datetime import datetime

# external
import skopt

# project
from fpcnn import encoding, models
from fpcnn.libs import benchmarklib, datalib, plotlib

# from tqdm.auto import tqdm


LOG = logging.getLogger(__name__)


class Tuner:
    """A tuner for hyperparameter optimization. All model-specific tuners must subclass
    from this class."""

    def __init__(self):
        # the following must be defined in derived classes:
        self._space = None

    def tune(self, n_calls):
        """Tunes the defined black box function according to the defined hyperparamter
        space.

        Args:
            n_calls (int): Number of calls to black box function. Must be greater than
                or equal to 3.
        """

        time = datetime.now().strftime("%Y%m%d-%H%M%S")

        opt_res = skopt.gp_minimize(
            func=self._blackbox,
            dimensions=self._space,
            base_estimator=None,
            n_calls=n_calls,
            n_initial_points=3,
            initial_point_generator="random",
            acq_func="gp_hedge",
            acq_optimizer="sampling",
            x0=None,
            y0=None,
            random_state=None,
            verbose=True,
            callback=self._callback,
            n_points=10000,
            n_restarts_optimizer=None,
            xi=0.01,
            kappa=1.96,
            noise="gaussian",
            n_jobs=None,
            model_queue_size=None,
        )

        LOG.info(
            (
                "Tuning results:\n"
                f"Location of min:\t{opt_res.x}\n"
                f"Function value at min:\t{opt_res.fun}"
            )
        )

        # region plots
        LOG.info("Drawing plots")
        plotlib.plot_skopt_evaluations(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_objective(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_convergence(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_regret(opt_res, f"logs/tuner/{time}/")
        # endregion

        winsound.MessageBeep()

    def explore(self, n_calls):
        """Conducts random search by uniform sampling within the given bounds for the
        defined black box function and hyperparamter space.

        Args:
            n_calls (int): Number of calls to black box function.
        """

        time = datetime.now().strftime("%Y%m%d-%H%M%S")

        opt_res = skopt.gp_minimize(
            func=self._blackbox,
            dimensions=self._space,
            base_estimator=None,
            n_calls=n_calls,
            n_initial_points=n_calls,
            initial_point_generator="random",
            acq_func="gp_hedge",
            acq_optimizer="sampling",
            x0=None,
            y0=None,
            random_state=None,
            verbose=True,
            callback=self._callback,
            n_points=10000,
            n_restarts_optimizer=None,
            xi=0.01,
            kappa=1.96,
            noise="gaussian",
            n_jobs=None,
            model_queue_size=None,
        )

        LOG.info(
            (
                "Exploration results:\n"
                f"Location of min:\t{opt_res.x}\n"
                f"Function value at min:\t{opt_res.fun}"
            )
        )

        # region plots
        plotlib.plot_skopt_evaluations(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_objective(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_convergence(opt_res, f"logs/tuner/{time}/")
        plotlib.plot_skopt_regret(opt_res, f"logs/tuner/{time}/")
        # endregion

        winsound.MessageBeep()

    def _blackbox(self, params):
        raise NotImplementedError("Method must be implemented in derived classes.")

    @staticmethod
    def _callback(opt_res):
        LOG.debug(f"Hyperparameters that were just tested: {opt_res.x_iters[-1]}")

    def _get_data(self, params):
        raise NotImplementedError("Method must be implemented in derived classes.")

    def _build_model(self, params):
        raise NotImplementedError("Method must be implemented in derived classes.")

    def _record_hparams(self, writer, params, loss, acc):
        raise NotImplementedError("Method must be implemented in derived classes.")


class Tuner_IP(Tuner):
    """A tuner for FPCNN on the Indian Pines dataset."""

    def __init__(self):
        super().__init__()

        self.space = [
            skopt.space.space.Categorical(
                categories=[1, 4, 8, 16, 32, 64, 128, 256],
                transform="identity",
                name="track_spatial_length",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8, 16, 32, 64, 128, 256],
                transform="identity",
                name="track_spatial_width",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8, 16, 32, 64, 128, 256],
                transform="identity",
                name="track_spectral_length",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8, 16, 32, 64, 128, 256],
                transform="identity",
                name="track_spectral_width",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8, 16, 32, 64, 128, 256],
                transform="identity",
                name="track_fusion_length",
            ),
            skopt.space.space.Categorical(
                categories=[1, 4, 8, 16, 32, 64, 128, 256],
                transform="identity",
                name="track_fusion_width",
            ),
            skopt.space.space.Real(
                low=0.0001,
                high=1.0,
                name="lr",
            ),
            skopt.space.space.Integer(
                low=0,
                high=32,
                name="grc_m",
            ),
        ]

    def _get_data(self, params):
        data = datalib.load_data_hdf5(
            path="data/indian_pines.mat", header="indian_pines"
        )
        LOG.info(f"Original data ({data.shape}, {data.dtype}):\n{data}")
        data = data[0:8, 0:8, 0:8]
        LOG.info(f"Cropped data ({data.shape}, {data.dtype}):\n{data}")

        return data

    def _blackbox(self, params):
        """The blackbox function to be optimized.
        Args:
            params (list): A list of hyperparameters to be evaluated in the current
                call to the blackbox function.

        Returns:
            float: Function loss.
        """

        data = self._get_data(params)
        model = self._build_model(params)

        data_compressed = model.compress(data)
        data_mapped = encoding.map_residuals(data_compressed.flatten())
        data_encoded = encoding.grc_encode(data=data_mapped, m=params[7])

        loss = benchmarklib.get_bpc_encoded(original=data, encoded=data_encoded)

        return loss

    def _build_model(self, params):
        model = models.FPCNN(
            hp={
                "track_spatial_length": int(params[0]),
                "track_spatial_width": int(params[1]),
                "track_spectral_length": int(params[2]),
                "track_spectral_width": int(params[3]),
                "track_fusion_length": int(params[4]),
                "track_fusion_width": int(params[5]),
                "lr": int(params[6]),
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
                "context_offsets_spectral": [
                    (0, 0, -1),
                    (0, 0, -2),
                    (0, 0, -3),
                    (0, 0, -4),
                ],
            },
            logname="tuning_ip",
        )

        return model

    def _record_hparams(self, writer, params, loss):
        """Records the current hyperparameters along with metrics to the Tensorboard
            writer.

        Args:
            writer (SummaryWriter): The tensorboard writer provided by the Trainer.
            params (list): list of evaluated hyperparameters.
            loss (torch.Tensor): Final test loss output.
        """
        writer.add_hparams(
            hparam_dict={
                "track_spatial_length": int(params[0]),
                "track_spatial_width": int(params[1]),
                "track_spectral_length": int(params[2]),
                "track_spectral_width": int(params[3]),
                "track_fusion_length": int(params[4]),
                "track_fusion_width": int(params[5]),
                "lr": int(params[6]),
            },
            metric_dict={"Metric/bpc": float(loss)},
        )
