
import os
import pickle
import numpy as np
import datetime
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from param2weld.data.loader import load_data
from param2weld.data.scaler import save_scaler
from param2weld.models.cnn_decoder import build_decoder_model
from param2weld.config.config import Config
from param2weld.train.callbacks import get_default_callbacks


def run_cv_training(
    config: Config,
    data_folders: list,
    output_dir: Path = None,
    **kwargs
):
    """
    Train CNN decoder using k-fold cross-validation.

    Parameters
    ----------
    config : Config
        Global configuration object.
    data_folders : list of str or Path
        List of paths to simulation folders.
    output_dir : Path, optional
        Optional output directory to save models/logs.
    **kwargs : dict
        Optional overrides: mid_mode, w_mae, w_ssim, batch_size, dropout_rate, etc.
    """
    resolution = config.resolution
    k = config.kfold_splits

    # Load optional overrides from kwargs
    mid_mode = kwargs.get("mid_mode", "mean")
    w_mae = kwargs.get("w_mae", config.w_mae)
    w_ssim = kwargs.get("w_ssim", config.w_ssim)
    batch_size = kwargs.get("batch_size", 8)
    epochs = kwargs.get("epochs", 500)
    patience_es = kwargs.get("patience_es", 50)
    patience_lr = kwargs.get("patience_lr", 10)
    factor_lr = kwargs.get("factor_lr", 0.5)
    min_lr = kwargs.get("min_lr", 1e-6)
    dropout_rate = kwargs.get("dropout", 0.0)
    dense_units = kwargs.get("dense_units", 512)
    filters_block1 = kwargs.get("f_block1", 32)
    filters_block2 = kwargs.get("f_block2", 16)
    l2_reg = kwargs.get("l2_reg", 1e-6)
    learning_rate = kwargs.get("learning_rate", 3e-4)

    # Print parameter summary
    print("Configuration for training:")
    print(f"  mid_mode:       {mid_mode}")
    print(f"  w_mae:          {w_mae}")
    print(f"  w_ssim:         {w_ssim}")
    print(f"  batch_size:     {batch_size}")
    print(f"  dropout:        {dropout_rate}")
    print(f"  dense_units:    {dense_units}")
    print(f"  filters_block1: {filters_block1}")
    print(f"  filters_block2: {filters_block2}")
    print(f"  l2_reg:         {l2_reg}")
    print(f"  learning_rate:  {learning_rate}")
    print(f"  epochs:         {epochs}")
    print(f"  patience_es:    {patience_es}")
    print(f"  patience_lr:    {patience_lr}")

    # Collect all parameters into a dict for saving
    params_summary = {
        "mid_mode": mid_mode,
        "w_mae": w_mae,
        "w_ssim": w_ssim,
        "dropout": dropout_rate,
        "dense_units": dense_units,
        "f_block1": filters_block1,
        "f_block2": filters_block2,
        "l2_reg": l2_reg,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }

    mae_ssim = f"mae{int(w_mae * 100)}_ssim{int(w_ssim * 100)}"

    # Default output directory
    if output_dir is None:
        output_dir = config.model_dir / mid_mode / mae_ssim

    os.makedirs(output_dir, exist_ok=True)
    kf = KFold(n_splits=k, shuffle=True, random_state=config.seed)
    metrics_list = []

    # Create a timestamped directory for this run
    run_timestamp_dir = config.log_dir / mid_mode / mae_ssim / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(run_timestamp_dir, exist_ok=True)

    fold_assignments = {} # Store which folders go to which fold
    for fid, (train_idx, val_idx) in enumerate(kf.split(data_folders)):
        print(f"\nStarting fold {fid + 1}/{k}")

        model_path = output_dir / f"cnn_decoder_fold_{fid}.keras"
        history_path = output_dir / f"history_fold_{fid}.npy"
        scaler_path = output_dir / f"scaler_fold_{fid}.pkl"

        if model_path.exists() and history_path.exists():
            print(f"Fold {fid} already trained — skipping.")
            continue

        train_folders = [data_folders[i] for i in train_idx]
        val_folders = [data_folders[i] for i in val_idx]

        # Save fold assignments for reproducibility
        fold_assignments[fid] = {
            "val": [str(p) for p in val_folders],
            "train": [str(p) for p in train_folders]
        }

        # Load training and validation data
        X_train, y_train, _ = load_data(train_folders, resolution=resolution, mid_mode=mid_mode)
        X_val, y_val, _ = load_data(val_folders, resolution=resolution, mid_mode=mid_mode)

        if X_train.size == 0 or X_val.size == 0:
            print(f"Fold {fid}: Skipping due to empty data.")
            continue

        # Standardize input features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Reshape targets into images
        y_train_img = y_train.reshape(-1, resolution, resolution, 1)
        y_val_img = y_val.reshape(-1, resolution, resolution, 1)

        # Build CNN model
        model = build_decoder_model(
            resolution=resolution,
            dropout_rate=dropout_rate,
            dense_units=dense_units,
            filters_block1=filters_block1,
            filters_block2=filters_block2,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
            w_mae=w_mae,
            w_ssim=w_ssim,
        )

        # TensorBoard logging directory
        logdir = run_timestamp_dir / f"fold_{fid}"
        os.makedirs(logdir, exist_ok=True)

        # Save training params for this fold
        with open(logdir / "params.json", "w") as f:
            import json
            json.dump(params_summary, f, indent=2)

        # Get callbacks (includes TensorBoard)
        callbacks = get_default_callbacks(
            log_dir=logdir,
            patience_es=patience_es,
            patience_lr=patience_lr,
            factor_lr=factor_lr,
            min_lr=min_lr,
        )

        # Train the model
        history = model.fit(
            X_train_scaled,
            y_train_img,
            validation_data=(X_val_scaled, y_val_img),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate on validation set
        eval_res = model.evaluate(X_val_scaled, y_val_img, verbose=0)
        metrics_list.append(eval_res)
        print(f"Fold {fid} done: val_loss={eval_res[0]:.4f} | MAE={eval_res[1]:.4f} | RMSE={eval_res[2]:.4f}")

        # Save model, history, and scaler
        model.save(model_path, include_optimizer=False)
        np.save(history_path, history.history)
        save_scaler(scaler, scaler_path)

    # Save fold assignments for reproducibility
    folds_path = output_dir / "kfold_val_folders.json"
    with open(folds_path, "w") as f:
        import json
        json.dump(fold_assignments, f, indent=2)

    # Print final averaged metrics
    if metrics_list:
        arr = np.array(metrics_list)
        print(f"\nAverage across {len(arr)} folds — Loss: {arr[:,0].mean():.4f} | MAE: {arr[:,1].mean():.4f} | RMSE: {arr[:,2].mean():.4f}")
    else:
        print("No folds were trained.")