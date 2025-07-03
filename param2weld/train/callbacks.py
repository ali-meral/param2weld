import os
import tensorflow as tf
from pathlib import Path


def get_default_callbacks(
    log_dir: Path,
    patience_es: int = 50,
    patience_lr: int = 10,
    factor_lr: float = 0.5,
    min_lr: float = 1e-6,
    extra: list = None,
):
    """
    Return default training callbacks: EarlyStopping, ReduceLROnPlateau, TensorBoard.

    Parameters
    ----------
    log_dir : Path
        Directory to store TensorBoard logs.
    patience_es : int
        Patience for EarlyStopping.
    patience_lr : int
        Patience for learning rate reduction.
    factor_lr : float
        Factor to reduce LR by.
    min_lr : float
        Minimum learning rate.
    extra : list
        Any additional callbacks to append.

    Returns
    -------
    list of tf.keras.callbacks.Callback
    """
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=factor_lr,
            patience=patience_lr,
            min_lr=min_lr,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]

    if extra:
        callbacks.extend(extra)

    return callbacks
