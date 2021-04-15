"""FINCH Predictive Coder Neural Network core class."""

# external
import numpy as np
import tensorflow as tf

# project
from fpcnn.libs import printlib


class FPCNN:
    """FINCH Predictive Coder Neural Network class."""

    def __init__(self, offsets_spatial, offsets_spectral):
        """Initialize model class.

        Args:
            offsets_spatial (list): list of spatial offsets
            offsets_spectral (list): list of spectral offsets
        """
        self.__validate_offsets(offsets=offsets_spatial)
        self.__validate_offsets(offsets=offsets_spectral)

        self.offsets_spatial = offsets_spatial
        self.offsets_spectral = offsets_spectral

        self.model, self.optimizer, self.losser = self.__instantiate_model()

    def __instantiate_model(self):
        """Compile model according to architecture.

        Returns:
            model (tf.keras.Model): compiled tensorflow model
        """
        inputs_spatial = tf.keras.Input(
            shape=(len(self.offsets_spatial),), name="Spatial_Context"
        )
        inputs_spectral = tf.keras.Input(
            shape=(len(self.offsets_spectral),), name="Spectral_Context"
        )

        dense_spatial = tf.keras.layers.Dense(
            units=5,
            activation="relu",
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="Spatial_Extraction",
        )
        x1 = dense_spatial(inputs_spatial)

        dense_spectral = tf.keras.layers.Dense(
            units=5,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name="Spectral_Extraction",
        )
        x2 = dense_spectral(inputs_spectral)

        x3 = tf.keras.layers.Concatenate(axis=-1)([x1, x2])

        dense_concat = tf.keras.layers.Dense(
            units=5,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
        x4 = dense_concat(x3)

        dense_output = tf.keras.layers.Dense(
            units=1,
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
        output = dense_output(x4)

        model = tf.keras.Model(
            inputs=[inputs_spatial, inputs_spectral], outputs=output, name="FPCNN"
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.Variable(0.001),
            beta_1=tf.Variable(0.9),
            beta_2=tf.Variable(0.999),
            epsilon=tf.Variable(1e-7),
            amsgrad=False,
            name="Adam",
        )  # https://gist.github.com/yoshihikoueno/4ff0694339f88d579bb3d9b07e609122
        optimizer.iterations  # this access will invoke optimizer._iterations method and create optimizer.iter attribute
        optimizer.decay = tf.Variable(0.0)

        losser = tf.keras.losses.MeanSquaredError(name="mean_squared_error")

        metrics = ["accuracy"]

        model.compile(
            optimizer=optimizer,
            loss=losser,
            metrics=metrics,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
        )

        return model, optimizer, losser

    @staticmethod
    def __validate_offsets(offsets):
        """Check to make sure offset selection does not access unseen voxels.

        Args:
            offsets (list): list of voxel offsets

        Raises:
            ValueError: future voxel accessed
        """
        for offset in offsets:
            if (offset[0] >= 0 and offset[1] == 0 and offset[2] >= 0) or (
                offset[1] > 0 and offset[2] >= 0
            ):
                raise ValueError(
                    f"Offset {offset} is invalid. Attempted to access future voxel."
                )

    @staticmethod
    def __get_context(data, index, offsets):
        """Get current voxel context.

        Args:
            data (ndarray): hyperspectral datacube
            index (tuple): current index in datacube
            offsets (list): list of offset indicies

        Returns:
            context (ndarray): array of context values
        """
        context = []

        for offset in offsets:
            if (
                index[0] + offset[0] not in range(data.shape[0])
                or index[1] + offset[1] not in range(data.shape[1])
                or index[2] + offset[2] not in range(data.shape[2])
            ):
                element = 0
            else:
                element = data[
                    index[0] + offset[0], index[1] + offset[1], index[2] + offset[2]
                ]

            context.append(element)

        context = np.reshape(
            np.array(context), (1, len(context))
        )  # update to use squeeze (?)

        return context

    def print_model(self):
        """Print model architecture visualization."""
        self.model.summary()
        tf.keras.utils.plot_model(self.model, "fpcnn.png", show_shapes=True)

    def encode(self, data):
        """Encode datacube using predictive encoder.

        Args:
            data (ndarray): hyperspectral datacube

        Returns:
            output (ndarray): prediction residuals
            losses (ndarray): array of losses captured during model adaptation
        """
        output = np.zeros(shape=data.shape, dtype="int16", order="C")
        losses = []

        n = data.shape[0] * data.shape[1] * data.shape[2]
        bar = printlib.ProgressBar(message="Predictive encoding", max=n)
        for k in range(data.shape[2]):
            for j in range(data.shape[1]):
                for i in range(data.shape[0]):
                    index = (i, j, k)

                    # get label
                    label = data[index].item()

                    # get context
                    context_spatial = self.__get_context(
                        data=data, index=index, offsets=self.offsets_spatial
                    )
                    context_spectral = self.__get_context(
                        data=data, index=index, offsets=self.offsets_spectral
                    )

                    with tf.GradientTape() as tape:
                        # get prediction
                        pred = self.model(
                            {
                                "Spatial_Context": context_spatial,
                                "Spectral_Context": context_spectral,
                            }
                        )

                        # get loss
                        loss = self.losser(y_true=label, y_pred=pred)

                    losses.append(loss.numpy().item())

                    # get error
                    pred_rounded = round(
                        pred.numpy().item()
                    )  # might want to round ceil or floor? Rounding is non-differentiable, so rounding does not occur inside the tape
                    error = label - pred_rounded
                    output[index] = error

                    # compute gradients and update weights
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(
                        grads_and_vars=zip(grads, self.model.trainable_variables),
                        name=None,
                        experimental_aggregate_gradients=True,
                    )

                    bar.next()
        bar.finish()

        losses = np.array(losses)

        return output, losses

    def decode(self, data):
        """Decode residuals.

        Args:
            data (ndarray): array of residuals

        Returns:
            output (ndarray): reconstruction of original datacube
            losses (ndarray): array of losses captured during model adaptation
        """
        output = np.zeros(shape=data.shape, dtype="uint16", order="C")
        losses = []

        n = data.shape[0] * data.shape[1] * data.shape[2]
        bar = printlib.ProgressBar(message="Predictive decoding", max=n)
        for k in range(data.shape[2]):
            for j in range(data.shape[1]):
                for i in range(data.shape[0]):
                    index = (i, j, k)

                    # get error
                    error = data[index].item()

                    # get context
                    context_spatial = self.__get_context(
                        data=output, index=index, offsets=self.offsets_spatial
                    )
                    context_spectral = self.__get_context(
                        data=output, index=index, offsets=self.offsets_spectral
                    )

                    with tf.GradientTape() as tape:
                        # get prediction
                        pred = self.model(
                            {
                                "Spatial_Context": context_spatial,
                                "Spectral_Context": context_spectral,
                            }
                        )
                        pred_rounded = round(pred.numpy().item())

                        # get label
                        label = pred_rounded + error

                        # get loss
                        loss = self.losser(y_true=label, y_pred=pred)

                    output[index] = label
                    losses.append(loss.numpy().item())

                    # compute gradients and update weights
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(
                        grads_and_vars=zip(grads, self.model.trainable_variables),
                        name=None,
                        experimental_aggregate_gradients=True,
                    )

                    bar.next()
        bar.finish()

        losses = np.array(losses)

        return output, losses

    def get_weights(self):
        """Get model weights.

        Returns:
            weights (list{ndarray}): List of ndarrays defining weights
        """
        weights = self.model.get_weights()
        return weights

    def set_weights(self, weights):
        """Set model weights.

        Args:
            weights (list{ndarray}): List of ndarrays defining weights
        """
        self.model.set_weights(weights)
