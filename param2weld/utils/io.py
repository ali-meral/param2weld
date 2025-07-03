import os
import json
import pickle
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model, save_model


def save_model_artifacts(
    model,
    model_path: Path,
    history: dict,
    history_path: Path,
    scaler=None,
    scaler_path: Path = None,
):
    """
    Save model, training history, and optional scaler.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    model_path : Path
        Path to save model (.keras).
    history : dict
        Keras training history.
    history_path : Path
        Path to save history (.npy).
    scaler : sklearn scaler, optional
        Fitted input scaler.
    scaler_path : Path, optional
        Path to save scaler (.pkl).
    """
    os.makedirs(model_path.parent, exist_ok=True)
    save_model(model, model_path, include_optimizer=False)
    np.save(history_path, history)

    if scaler is not None and scaler_path is not None:
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)


def load_model_and_scaler(model_path: Path, scaler_path: Path = None):
    """
    Load Keras model and optional scaler.

    Parameters
    ----------
    model_path : Path
        Path to saved .keras model.
    scaler_path : Path, optional
        Path to saved .pkl scaler.

    Returns
    -------
    tuple (model, scaler or None)
    """
    model = load_model(model_path, compile=False)
    scaler = None
    if scaler_path and scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    return model, scaler


def save_json(data: dict, path: Path):
    """
    Save a dictionary as a JSON file.

    Parameters
    ----------
    data : dict
        Data to save.
    path : Path
        Output .json path.
    """
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
