import os
import glob
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Union


def load_data(
    folders: Union[List[str], List[Path], None] = None,
    resolution: int = 32,
    mid_mode: str = "mean",
    start_frac: float = 0.5,
    end_frac: float = 0.8,
    data_dir: Union[str, Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and process simulation images from folders.

    Parameters
    ----------
    folders : list of folder paths named like base_velocity_power_spotsize
        If None, uses subdirectories under data_dir.
    resolution : int
        Subfolder name where images are located (e.g., '32')
    mid_mode : 'mean' or 'median'
        Averaging mode for selecting keyhole images.
    start_frac : float
        Fraction to skip at the beginning of sequence.
    end_frac : float
        Fraction to skip at the end of sequence.
    data_dir : Path or str
        Optional base folder to scan if folders is None.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 3)
        Laser parameters [velocity, power, spotsize].
    y : np.ndarray, shape (n_samples, resolution * resolution)
        Flattened morphology images (normalized to [0, 1]).
    filenames : list of str
        Processed folder paths.
    """
    if folders is None:
        if data_dir is None:
            raise ValueError("If `folders` is None, `data_dir` must be provided.")
        base = Path(data_dir)
        folders = [f for f in base.iterdir() if f.is_dir()]

    X, y, filenames = [], [], []

    for folder in folders:
        name = os.path.basename(str(folder))
        parts = name.split("_")
        if len(parts) < 4:
            continue
        try:
            velocity, power, spotsize = map(float, parts[1:4])
        except ValueError:
            continue

        image_dir = os.path.join(folder, str(resolution))
        if not os.path.isdir(image_dir):
            continue

        files = glob.glob(os.path.join(image_dir, "*.png"))
        if not files:
            continue

        try:
            nums = np.array([float(os.path.splitext(os.path.basename(f))[0]) for f in files])
        except ValueError:
            continue

        order = np.argsort(nums)
        sorted_files = [files[i] for i in order]
        n_images = len(sorted_files)

        first_index = int(n_images * start_frac)
        last_index = int(n_images * end_frac)
        selected_files = sorted_files[first_index:last_index]

        imgs = []
        for p in selected_files:
            if os.path.exists(p):
                arr = np.array(Image.open(p).convert("L"), dtype=np.float32)
                imgs.append(arr)

        if not imgs:
            continue

        stacked = np.stack(imgs, axis=0)
        avg_img = np.median(stacked, axis=0) if mid_mode == "median" else np.mean(stacked, axis=0)

        X.append([velocity, power, spotsize])
        y.append(avg_img.flatten() / 255.0)
        filenames.append(str(folder))

    if X:
        return np.array(X), np.array(y), filenames

    return np.empty((0, 3)), np.empty((0, resolution * resolution)), []
