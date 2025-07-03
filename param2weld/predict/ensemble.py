import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List
from tensorflow.keras.models import load_model

from param2weld.models.losses import hybrid_loss


def load_ensemble_models(
    model_dir: Path,
    fold_count: int = 10,
    w_mae: float = 0.8,
    w_ssim: float = 0.2,
) -> List[tf.keras.Model]:
    """
    Load trained ensemble models from directory and compile with custom loss.

    Parameters
    ----------
    model_dir : Path
        Directory containing model files named cnn_decoder_fold_{i}.keras.
    fold_count : int
        Number of folds to load.
    w_mae : float
        Weight for MAE loss.
    w_ssim : float
        Weight for SSIM loss.

    Returns
    -------
    list of tf.keras.Model
        Compiled models.
    """
    models = []
    loss_fn = hybrid_loss(w_mae, w_ssim)

    for i in range(fold_count):
        model_path = model_dir / f"cnn_decoder_fold_{i}.keras"
        if model_path.exists():
            model = load_model(model_path, compile=False)
            model.compile(loss=loss_fn)
            models.append(model)

    return models


def predict_ensemble(models: List[tf.keras.Model], X: np.ndarray) -> np.ndarray:
    """
    Predict with ensemble and average the results.

    Parameters
    ----------
    models : list of tf.keras.Model
        Ensemble models.
    X : np.ndarray, shape (N, 3)
        Scaled input parameters.

    Returns
    -------
    np.ndarray, shape (N, H, W, 1)
        Averaged output predictions.
    """
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    preds = [model(X_tensor, training=False) for model in models]
    return tf.reduce_mean(preds, axis=0).numpy()
