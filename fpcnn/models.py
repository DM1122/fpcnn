"""FINCH Predictive Coder Neural Network core classes."""

# stdlib
import logging
import os
from datetime import datetime

# external
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm

# project
from fpcnn.libs import printlib

LOG = logging.getLogger(__name__)


class FPCNN:
    """FINCH Predictive Coder Neural Network class.

    Compresses hyperspectral datacubes using an adaptive algorithm.

    Args:
        hp (dict): Hyperparameters used to define model architecture.
        logname (str, optional): Name for logging directory. Defaults to None.
    """

    def __init__(self, hp, logname=None):
        self._hp = hp
        self._logname = logname
        LOG.info("Validating context offsets")
        self._validate_offsets(self._hp["context_offsets_spatial"])
        self._validate_offsets(self._hp["context_offsets_spectral"])

        self._logdate = datetime.now().strftime("%Y%m%d-%H%M%S")

        # region logging setup
        LOG.info("Configuring logging directories")
        if self._logname is not None:
            self._logdir_tb = f"logs/tensorboard/{self._logname}/"
            self._logdir_plot = f"logs/plots/{self._logname}/"
        else:
            self._logdir_tb = "logs/tensorboard/"
            self._logdir_plot = "logs/plots/"

        if not os.path.exists(self._logdir_plot):
            os.makedirs(self._logdir_plot)

        self._writer = tf.summary.create_file_writer(
            logdir=self._logdir_tb + self._logdate,
            max_queue=None,
            flush_millis=None,
            filename_suffix=None,
            name=None,
        )
        LOG.info({type(self._writer)})
        # endregion

        # region model instantiation
        LOG.info("Building model")
        self._model, self._optimizer, self._losser = self._build_model()
        summary = printlib.get_tf_model_summary(self._model)
        LOG.info(summary)
        try:
            tf.keras.utils.plot_model(
                self._model,
                self._logdir_plot + self._logdate + ".png",
                show_shapes=True,
            )
        except ImportError:
            LOG.warning("Could not draw model. Probably running headless.")

        # write initial weights and biases to tensorboard
        self._tb_weights(step=-1)

        LOG.info(f"Using {self._optimizer} optimizer")
        LOG.info(f"Using {self._losser} loss function")
        # endregion

    def _build_model(self):
        """Build model according to defined hyperparameters.

        Loss reduction must be set to tf.keras.losses.Reduction.NONE

        Returns:
            model (tf.keras.Model): Compiled Tensorflow model.
            optimizer (tf.keras.optimizers.Optimizer): Optimizer.
            losser (tf.keras.losses.Loss): Loss function.
        """

        # region inputs
        inputs_spatial = tf.keras.Input(
            shape=(len(self._hp["context_offsets_spatial"]),), name="Spatial_Context"
        )
        inputs_spectral = tf.keras.Input(
            shape=(len(self._hp["context_offsets_spectral"]),), name="Spectral_Context"
        )
        # endregion

        # region spatial track
        x = inputs_spatial
        for i in range(self._hp["track_spatial_length"]):
            x = tf.keras.layers.Dense(
                units=self._hp["track_spatial_width"],
                activation="relu",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                name=f"Spatial_Extraction_{i}",
            )(x)
        track_spatial = x
        # endregion

        # region spectral track
        x = inputs_spectral
        for i in range(self._hp["track_spectral_length"]):
            x = tf.keras.layers.Dense(
                units=self._hp["track_spectral_width"],
                activation="relu",
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                name=f"Spectral_Extraction_{i}",
            )(x)
        track_spectral = x
        # endregion

        # region fusion track
        x = tf.keras.layers.Concatenate(axis=-1, name="Merge")(
            [track_spatial, track_spectral]
        )
        for i in range(self._hp["track_fusion_length"]):
            x = tf.keras.layers.Dense(
                units=self._hp["track_fusion_width"],
                activation=None,
                use_bias=True,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                name=f"Fusion_{i}",
            )(x)
        track_merge = x
        # endregion

        # region output
        output = tf.keras.layers.Dense(
            units=1,
            activation="relu",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="Output",
        )(track_merge)
        # endregion

        model = tf.keras.Model(
            inputs=[inputs_spatial, inputs_spectral], outputs=output, name="FPCNN"
        )

        # optimizer
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.Variable(self._hp["lr"]),
            beta_1=tf.Variable(0.9),
            beta_2=tf.Variable(0.999),
            epsilon=tf.Variable(1e-7),
            amsgrad=False,
            name="Adam",
        )  # https://gist.github.com/yoshihikoueno/4ff0694339f88d579bb3d9b07e609122
        # this access will invoke optimizer._iterations method and create optimizer.iter
        # attribute
        optimizer.iterations
        optimizer.decay = tf.Variable(0.0)

        # losser
        losser = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE,
            name="mean_squared_error",
        )

        # compilation
        model.compile(
            optimizer=optimizer,
            loss=losser,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
        )

        return model, optimizer, losser

    def _tb_weights(self, step):
        """Write model weights and biases to Tensorboard.

        Args:
            step (int): The current iteration step for which to write weights.
        """
        with self._writer.as_default():
            for layer in self._model.layers:
                layer_params = layer.get_weights()
                if layer_params != []:
                    tf.summary.histogram(
                        name=layer.name + "/weights",
                        data=layer.get_weights()[0],
                        step=step,
                        buckets=32,
                        description="Layer weights",
                    )
                    tf.summary.histogram(
                        name=layer.name + "/biases",
                        data=layer.get_weights()[1],
                        step=step,
                        buckets=32,
                        description="Layer biases",
                    )

    def _tb_datacube(self, cube, name, desc="A datacube"):
        """Writes a hypespectral datacube to tensorboard as a series of timestep images.

        Each band is written as a single timestep image.

        Args:
            cube (numpy.ndarray): Hyperspectral datacube of shape (x, y, lambda).
        """
        with self._writer.as_default():
            cube_scaled = (
                (cube - cube.min()) * (1 / (cube.max() - cube.min()) * 255)
            ).astype(
                np.uint8
            )  # scale to avoid clipping
            for ix_band in range(cube.shape[2]):
                cube_scaled_sliced = cube_scaled[:, :, ix_band]
                cube_scaled_sliced_shaped = np.expand_dims(
                    a=cube_scaled_sliced, axis=(0, -1)
                )  # reshape to (k, h, w, c)
                tf.summary.image(
                    name=name,
                    data=cube_scaled_sliced_shaped,
                    step=ix_band,
                    description=desc,
                )

    def _tb_metric(self, name, data, step, desc="A metric"):
        """Write scalar metric to Tensorboard.

        Args:
            data (float): The metric to record.
        """
        with self._writer.as_default():
            tf.summary.scalar(
                name=name,
                data=data,
                step=step,
                description=desc,
            )

    @staticmethod
    def _validate_offsets(offsets):
        """Check to make sure offset selection does not access unseen voxels.

        Args:
            offsets (list): List of voxel offsets.

        Raises:
            ValueError: In the case a context offset in invalid, ValueError is raised.
        """
        for offset in offsets:
            if (offset[0] >= 0 and offset[1] == 0 and offset[2] >= 0) or (
                offset[1] > 0 and offset[2] >= 0
            ):
                raise ValueError(
                    f"Offset {offset} is invalid. Attempted to access future voxel."
                )

    @staticmethod
    def _get_context(data, index, offsets):
        """Retrieves context voxels at index from a datacube.

        Instead of initializing context as a list, a numpy array is preallocated.

        Args:
            data (ndarray): The datacube.
            index (list): Datacube index at which to retrieve context.
            offsets (list): List of offsets from index at which to retrieve context.
        Returns:
            ndarray: Context vector.
        """
        context = np.empty(shape=len(offsets), dtype=data.dtype)

        for i, offset in enumerate(offsets):
            if (
                index[0] + offset[0] not in range(data.shape[0])
                or index[1] + offset[1] not in range(data.shape[1])
                or index[2] + offset[2] not in range(data.shape[2])
            ):
                voxel = 0
            else:
                voxel = data[
                    index[0] + offset[0], index[1] + offset[1], index[2] + offset[2]
                ]  # TODO: use vector addition

            context[i] = voxel

        return context

    @staticmethod
    def _build_batch(data, band, offsets_spatial, offsets_spectral):
        # alloc features
        batch_spatial = np.empty(
            shape=(data.shape[0] * data.shape[1], len(offsets_spatial)),
            dtype=data.dtype,
        )
        LOG.debug(
            f"Allocating batch spatial array: {batch_spatial.shape}, "
            f"{batch_spatial.dtype}"
        )
        batch_spectral = np.empty(
            shape=(data.shape[0] * data.shape[1], len(offsets_spectral)),
            dtype=data.dtype,
        )
        LOG.debug(
            f"Allocating batch spectral array: {batch_spectral.shape}, "
            f"{batch_spectral.dtype}"
        )

        # alloc labels
        batch_labels = np.empty(
            shape=(data.shape[0] * data.shape[1], 1), dtype=data.dtype
        )
        LOG.debug(
            f"Empty batch labels array: {batch_labels.shape}, {batch_labels.dtype}"
        )

        idx_batch = 0
        for j in range(data.shape[1]):
            for i in range(data.shape[0]):  # TODO: use more consistent ix naming
                idx_data = (i, j, band)

                # get features
                context_spatial = FPCNN._get_context(
                    data=data, index=idx_data, offsets=offsets_spatial
                )
                batch_spatial[idx_batch] = context_spatial

                context_spectral = FPCNN._get_context(
                    data=data, index=idx_data, offsets=offsets_spectral
                )
                batch_spectral[idx_batch] = context_spectral

                # get labels
                batch_labels[idx_batch] = data[idx_data]
                idx_batch += 1

        return batch_spatial, batch_spectral, batch_labels

    def compress(self, data):
        """Compress datacube using predictive encoder.

        Args:
            data (numpy.ndarray): Hyperspectral datacube of shape (x, y, lambda).

        Returns:
            output (numpy.ndarray): Prediction residuals.
        """
        LOG.info("Writing input datacube to Tensorboard")
        self._tb_datacube(cube=data, name="input cube", desc="The input datacube")

        LOG.info("Preallocating output array")
        output = np.empty(shape=data.shape, dtype=data.dtype)
        LOG.debug(f"Empty output array ({output.shape}, {output.dtype}):\n{output}")

        LOG.info("Starting compression")
        pbar = tqdm(
            desc="Compressing datacube",
            total=data.shape[2],
            leave=True,
            unit="band",
            dynamic_ncols=True,
        )
        for ix_band in range(data.shape[2]):
            LOG.debug(f"Starting band {ix_band}")
            # region forward pass
            LOG.debug("Building batch")
            (
                context_spatial_batch,
                context_spectral_batch,
                labels_batch,
            ) = self._build_batch(
                data=data,
                band=ix_band,
                offsets_spatial=self._hp["context_offsets_spatial"],
                offsets_spectral=self._hp["context_offsets_spectral"],
            )
            LOG.debug(
                f"Batch spatial features ({context_spatial_batch.shape}, "
                f"{context_spatial_batch.dtype}):\n{context_spatial_batch}"
            )
            LOG.debug(
                f"Batch spectral features ({context_spectral_batch.shape}, "
                f"{context_spectral_batch.dtype}):\n{context_spectral_batch}"
            )
            LOG.debug(
                f"Batch labels ({labels_batch.shape}, {labels_batch.dtype}):\n"
                f"{labels_batch}"
            )

            with tf.GradientTape() as tape:
                # logits
                LOG.debug("Computing logits")
                logits = self._model(
                    {
                        "Spatial_Context": context_spatial_batch,
                        "Spectral_Context": context_spectral_batch,
                    }
                )
                LOG.debug(f"Logits ({logits.shape}, {logits.dtype}):\n{logits}")

                # loss
                LOG.debug("Computing batch loss")
                loss_obj = self._losser(y_true=labels_batch, y_pred=logits)
                loss = tf.math.reduce_mean(
                    input_tensor=loss_obj, name="reduction"
                )  # average loss across batch
                LOG.debug(f"Batch loss ({type(loss)}, {loss.dtype}): {loss}")

            # residual
            LOG.debug("Computing residual")
            logits_rounded = np.round(logits)
            LOG.debug(
                f"Logits rounded ({logits_rounded.shape}, {logits_rounded.dtype}):"
                f"\n{logits_rounded}"
            )
            residual = labels_batch - logits_rounded
            residual = residual.reshape(
                data.shape[0], data.shape[1]
            )  # reshape to (x, y)
            LOG.debug(f"Residual ({residual.shape}, {residual.dtype}):\n{residual}")

            output[:, :, ix_band] = residual  # store slice in output

            LOG.debug("Writing metrics to Tensorboard")
            error_avg = np.mean(residual)
            error_percent = np.abs((error_avg / np.mean(labels_batch))) * 100
            self._tb_metric(
                name="loss",
                data=loss,
                step=ix_band,
                desc="Mean squared error averaged across band",
            )
            self._tb_metric(
                name="error",
                data=error_avg,
                step=ix_band,
                desc="Absolute error between voxel prediction and actual value, "
                "averaged acros band",
            )
            self._tb_metric(
                name="%_error",
                data=error_percent,
                step=ix_band,
                desc="Percentage error",
            )
            # endregion

            # region backward pass
            LOG.debug("Computing gradients")
            grads = tape.gradient(
                target=loss,
                sources=self._model.trainable_variables,
            )
            LOG.debug("Applying gradients")
            self._optimizer.apply_gradients(
                grads_and_vars=zip(grads, self._model.trainable_variables),
                name="Applying grads",
            )

            LOG.debug("Writing weights and biases to Tensorboard")
            self._tb_weights(step=ix_band)

            # endregion
            pbar.update()
        pbar.close()

        self._tb_datacube(
            cube=output,
            name="output cube",
            desc="The compressed output datacube of residuals",
        )

        return output

    def decompress(self, data):  # TODO: fix broken graph
        """Decompress datacube using predictive decoder.

        Args:
            data (numpy.ndarray): Array of residuals.

        Returns:
            output (numpy.ndarray): Reconstruction of original datacube.
        """
        LOG.info("Preallocating output array")
        output = np.empty(shape=data.shape, dtype=data.dtype)

        LOG.info("Starting decompression")
        pbar = tqdm(
            desc="Decompressing datacube",
            total=data.shape[2],
            leave=True,
            unit="band",
            dynamic_ncols=True,
        )
        for ix_band in range(data.shape[2]):
            LOG.debug(f"Starting band {ix_band}")

            LOG.info("Preallocating logits array for the band")
            logits_batch = tf.zeros(
                shape=(data.shape[0], data.shape[1]),
                dtype=tf.dtypes.float32,
                name="Batch loss array",
            )
            # logits_batch = np.empty(shape=(data.shape[0],data.shape[1]),
            # dtype=np.float32)

            # region forward passes
            with tf.GradientTape() as tape:
                for ix_col in range(data.shape[1]):
                    for ix_row in range(data.shape[0]):
                        index = (ix_row, ix_col, ix_band)

                        # error
                        residual = data[index]
                        LOG.debug(f"Residual at index {index}: {residual}")

                        # context
                        context_spatial = self._get_context(
                            data=output,
                            index=index,
                            offsets=self._hp["context_offsets_spatial"],
                        )
                        context_spectral = self._get_context(
                            data=output,
                            index=index,
                            offsets=self._hp["context_offsets_spectral"],
                        )

                        context_spatial = np.expand_dims(a=context_spatial, axis=0)
                        context_spectral = np.expand_dims(a=context_spectral, axis=0)

                        LOG.debug(
                            f"Spatial context ({context_spatial.shape}, "
                            f"{context_spatial.dtype}):\n{context_spatial}"
                        )
                        LOG.debug(
                            f"Spectral context ({context_spectral.shape}, "
                            f"{context_spectral.dtype}):\n{context_spectral}"
                        )

                        # logits
                        LOG.debug("Computing logits")
                        logits = self._model(
                            {
                                "Spatial_Context": context_spatial,
                                "Spectral_Context": context_spectral,
                            }
                        )
                        logits_batch[
                            ix_row, ix_col
                        ] = logits  # store logits for batch loss computation
                        # TODO: check if graph is broken
                        LOG.debug(f"Logits ({logits.shape}, {logits.dtype}):\n{logits}")
                        logits_rounded = np.round(logits)
                        LOG.debug(
                            f"Logits rounded ({logits_rounded.shape}, "
                            f"{logits_rounded.dtype}):\n{logits_rounded}"
                        )

                        # label
                        label = logits_rounded + residual
                        LOG.debug(
                            f"Recovered label ({label.shape}, {label.dtype}):\n{label}"
                        )

                        output[index] = label  # store recovered label in output

                # loss
                LOG.debug("Computing batch loss")
                loss_obj = self._losser(
                    y_true=output[:, :, ix_band], y_pred=logits_batch
                )  # TODO: check if logits need to be rounded here
                loss = tf.math.reduce_mean(
                    input_tensor=loss_obj, name="reduction"
                )  # average loss across batch
                LOG.debug(f"Batch loss ({type(loss)}, {loss.dtype}): {loss}")

            LOG.debug("Writing metrics to Tensorboard")
            self._tb_metric(
                name="loss",
                data=loss,
                step=ix_band,
                desc="Mean squared error averaged across band",
            )
            # endregion

            # region backward pass
            LOG.debug("Computing gradients")
            grads = tape.gradient(
                target=loss,
                sources=self._model.trainable_variables,
            )
            LOG.debug("Applying gradients")
            self._optimizer.apply_gradients(
                grads_and_vars=zip(grads, self._model.trainable_variables),
                name="Applying grads",
            )

            LOG.debug("Writing weights and biases to Tensorboard")
            self._tb_weights(step=ix_band)

            # endregion
            pbar.update()
        pbar.close()

        self._tb_datacube(
            cube=output, name="decompressed cube", desc="The decompressed cube"
        )

        return output

    def get_weights(self):
        """Get model weights.

        Returns:
            weights (list{ndarray}): List of ndarrays defining weights. See Tensorflow
                documentation for structure.
        """
        weights = self._model.get_weights()
        return weights

    def set_weights(self, weights):
        """Set model weights.

        Args:
            weights (list{ndarray}): List of ndarrays defining weights
        """
        self._model.set_weights(weights)
