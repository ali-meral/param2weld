import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    """
    Save a fitted scaler to disk using joblib.

    Parameters
    ----------
    scaler : StandardScaler
        The fitted scaler to save.
    path : Path
        Destination file path (e.g. 'scaler.pkl').
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path) -> StandardScaler:
    """
    Load a previously saved scaler from disk.

    Parameters
    ----------
    path : Path
        Path to the saved joblib scaler file.

    Returns
    -------
    StandardScaler
        Loaded scaler.
    """
    return joblib.load(path)
