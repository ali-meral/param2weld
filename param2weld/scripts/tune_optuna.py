"""
Run Optuna hyperparameter tuning with fixed loss config and K-fold split.
"""

import os
import json
import optuna
import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import argparse

from param2weld.config.config import Config
from param2weld.data.loader import load_data
from param2weld.models.cnn_decoder import build_decoder_model

cfg = Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument(
        "--fixed_params", type=Path, required=True,
        help="Path to JSON file with fixed parameters like mid_mode, w_mae, w_ssim"
    )
    return parser.parse_args()


class OptunaPruningCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        self.trial.report(val_loss, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def run_single_training(fold_train, fold_val, trial, resolution, fixed):
    mid_mode = fixed.get("mid_mode", "mean")
    w_mae = fixed.get("w_mae", 0.8)
    w_ssim = fixed.get("w_ssim", 0.2)

    X_train, y_train, _ = load_data(fold_train, resolution=resolution, mid_mode=mid_mode)
    X_val, y_val, _ = load_data(fold_val, resolution=resolution, mid_mode=mid_mode)

    if X_train.size == 0 or X_val.size == 0:
        raise ValueError("Empty training or validation data")

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    y_train_img = y_train.reshape(-1, resolution, resolution, 1)
    y_val_img = y_val.reshape(-1, resolution, resolution, 1)

    model = build_decoder_model(
        resolution=resolution,
        dropout_rate=trial.suggest_float("dropout", 0.0, 0.4),
        dense_units=trial.suggest_categorical("dense_units", [128, 256, 512, 1024, 2048, 4096]),
        filters_block1=trial.suggest_categorical("f_block1", [32, 64]),
        filters_block2=trial.suggest_categorical("f_block2", [8, 16, 32]),
        l2_reg=trial.suggest_float("l2_reg", 1e-6, 1e-4, log=True),
        learning_rate=1e-4,
        w_mae=w_mae,
        w_ssim=w_ssim,
    )

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6),
        OptunaPruningCallback(trial),
    ]

    model.fit(
        X_train_scaled,
        y_train_img,
        validation_data=(X_val_scaled, y_val_img),
        epochs=500,
        batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
        callbacks=cb,
        verbose=0,
    )

    val_loss = model.evaluate(X_val_scaled, y_val_img, verbose=0)[0]
    return val_loss


def objective(trial, resolution, fixed, folders):
    kf = KFold(n_splits=cfg.kfold_splits, shuffle=True, random_state=cfg.seed)
    train_idx, val_idx = next(iter(kf.split(folders)))
    fold_train = [folders[i] for i in train_idx]
    fold_val = [folders[i] for i in val_idx]
    return run_single_training(fold_train, fold_val, trial, resolution, fixed)


if __name__ == "__main__":
    args = parse_args()

    # Load fixed parameter file (w_mae, w_ssim, mid_mode)
    with open(args.fixed_params, "r") as f:
        fixed = json.load(f)

    resolution = cfg.resolution
    folders = [str(p) for p in cfg.data_dir.iterdir() if p.is_dir()]

    # Construct experiment folder from fixed params
    mid = fixed.get("mid_mode", "mean")
    mae_pct = int(fixed.get("w_mae", 0.8) * 100)
    ssim_pct = int(fixed.get("w_ssim", 0.2) * 100)
    exp_name = f"mae{mae_pct}_ssim{ssim_pct}"
    results_dir = Path("experiments/optuna/results") / mid / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run Optuna study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50, interval_steps=1),
    )
    study.optimize(lambda trial: objective(trial, resolution, fixed, folders), n_trials=50)

    # Combine Optuna results with fixed config
    best_all = dict(study.best_params)
    best_all.update(fixed)

    # Save
    print("\nBest hyperparameters:")
    for k, v in best_all.items():
        print(f"{k}: {v}")

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = results_dir / f"best_hyperparams_{now}.json"
    with open(out_path, "w") as f:
        json.dump(best_all, f, indent=2)

    print(f"\nSaved best parameters to: {out_path.resolve()}")
