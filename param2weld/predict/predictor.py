import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List

from param2weld.predict.ensemble import predict_ensemble


def generate_prediction_image(
    velocity: float,
    power: float,
    spotsize: float,
    scaler: StandardScaler,
    ensemble_models: List,
) -> np.ndarray:
    """
    Generate a morphology prediction image for given laser parameters.

    Parameters
    ----------
    velocity : float
        Laser scan speed.
    power : float
        Laser power.
    spotsize : float
        Laser spot size.
    scaler : StandardScaler
        Fitted input parameter scaler.
    ensemble_models : list of tf.keras.Model
        Trained ensemble models.

    Returns
    -------
    np.ndarray, shape (H, W)
        Predicted image (values in [0, 1]).
    """
    input_vector = np.array([[velocity, power, spotsize]], dtype=np.float32)
    input_scaled = scaler.transform(input_vector)
    prediction = predict_ensemble(ensemble_models, input_scaled)
    return prediction[0, :, :, 0]
