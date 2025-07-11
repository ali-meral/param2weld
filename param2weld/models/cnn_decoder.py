import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from param2weld.models.losses import hybrid_loss, ssim_loss_term


def build_decoder_model(
    input_dim: int = 3,
    resolution: int = 32,
    seed_size: int = 8,
    dropout_rate: float = 0.2,
    dense_units: int = 512,
    filters_block1: int = 32,
    filters_block2: int = 16,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-4,
    w_mae: float = 0.8,
    w_ssim: float = 0.2,
) -> tf.keras.Model:
    """
    Build and compile the convolutional decoder model.

    Parameters
    ----------
    input_dim : int
        Number of input features (typically 3: velocity, power, spotsize)
    resolution : int
        Output image size (e.g., 32x32)
    seed_size : int
        Intermediate spatial size before upsampling
    dropout_rate : float
        Dropout rate applied after the dense layer
    dense_units : int
        Number of units in the dense layer (must be divisible by seed_size^2)
    filters_block1 : int
        Number of filters in the first convolutional block
    filters_block2 : int
        Number of filters in the second convolutional block
    l2_reg : float
        L2 regularization factor
    learning_rate : float
        Initial learning rate for Adam optimizer
    w_mae : float
        Weight for MAE component in hybrid loss
    w_ssim : float
        Weight for SSIM component in hybrid loss

    Returns
    -------
    tf.keras.Model
        Compiled decoder model with hybrid loss and metrics.
    """
    if dense_units < seed_size * seed_size or dense_units % (seed_size * seed_size) != 0:
        raise ValueError(
            f"dense_units must be a multiple of {seed_size * seed_size}, got {dense_units}"
        )

    channels = dense_units // (seed_size * seed_size)

    inputs = layers.Input(shape=(input_dim,), name="weld_params")
    x = layers.Dense(dense_units)(inputs)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Reshape((seed_size, seed_size, channels))(x)

    # Upsampling and convolution blocks
    x = layers.UpSampling2D(size=2)(x)
    x = conv_block(x, filters_block1, l2_reg)

    x = layers.UpSampling2D(size=2)(x)
    x = conv_block(x, filters_block2, l2_reg)
    residual = conv_block(x, filters_block2, l2_reg)
    x = layers.Add()([x, residual])

    outputs = layers.Conv2D(1, 3, padding="same", activation="sigmoid", name="morphology")(x)

    model = models.Model(inputs, outputs, name="cnn_decoder")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=hybrid_loss(w_mae, w_ssim),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            ssim_loss_term,
        ],
    )

    return model


def conv_block(x, filters, l2_reg):
    """
    2D convolution block: Conv → BN → ReLU.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor
    filters : int
        Number of filters in the Conv2D layer
    l2_reg : float
        L2 regularization factor

    Returns
    -------
    tf.Tensor
        Output tensor after convolutional block
    """
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x
