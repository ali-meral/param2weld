# Param2Weld

**Param2Weld** is a machine learning pipeline that predicts keyhole morphology images from laser welding parameters (velocity, power, spotsize). It uses a convolutional decoder architecture trained via k-fold cross-validation and supports ensemble prediction, TensorBoard logging, and hyperparameter tuning with Optuna.

---

## Features

* Convolutional decoder model with MAE + SSIM hybrid loss
* K-fold cross-validation training
* Scalable ensemble-based prediction
* TensorBoard logging per fold
* Hyperparameter optimization using Optuna
* Modular codebase for dataset loading, model building, training, and evaluation

---

## Project Structure

```
param2weld/
├── config/           # Configuration (e.g. resolution, loss weights)
├── data/             # Data loading and preprocessing
├── models/           # CNN decoder model and custom losses
├── predict/          # Prediction utilities and ensemble loader
├── train/            # Training loop and callbacks
main.py               # CLI entry point
```

---

## Quick Start

### 1. Set up virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model using CLI

```bash
python main.py train --data_dir /path/to/sim_folders
```

Optional arguments:

* `--model_dir`: override the default model save path
* `--params_json`: provide a JSON file to override hyperparameters (see below)

---

### 3. Hyperparameter tuning with Optuna

Run random search or fixed configuration tuning over model/training parameters using Optuna:

```bash
python -m param2weld.scripts.tune_optuna --data_dir /path/to/sim_folders --n_trials 50
```

* `--data_dir`: Path to folders named like `sim_100_300_95` containing image sequences
* `--n_trials`: Number of trials for Optuna's random search

Example using fixed parameters defined in a JSON file:

```bash
python -m param2weld.scripts.tune_optuna \
    --fixed_params experiments/optuna/configs/fixed_median_mae80_ssim20.json
```

* `--fixed_params`: Path to a JSON file containing model and training hyperparameters (overrides Optuna search)

This runs k-fold cross-validation using the specified settings and logs the results for comparison and visualization.

---

## Important Parameters (from `config.py` and Optuna space)

| Parameter        | Description                                | Default   |
| ---------------- | ------------------------------------------ | --------- |
| `resolution`     | Output image size (e.g., 32×32)            | 32        |
| `w_mae`          | Weight of MAE in hybrid loss               | 0.8       |
| `w_ssim`         | Weight of SSIM in hybrid loss              | 0.2       |
| `dense_units`    | Size of dense layer (must match reshaping) | 512       |
| `dropout`        | Dropout rate after dense layer             | 0.0–0.3   |
| `filters_block1` | Filters in first conv block                | 16–64     |
| `filters_block2` | Filters in second conv block               | 8–32      |
| `learning_rate`  | Learning rate for Adam optimizer           | 1e-4–3e-4 |
| `batch_size`     | Batch size for training                    | 8–32      |
| `kfold_splits`   | Number of folds for cross-validation       | 10        |

---

## TensorBoard

Training logs are saved per fold in a timestamped directory. To view (example shown below):

```bash
tensorboard --logdir logs/mean/20250703-194233
```

This command assumes a run under `logs/mean/<timestamp>` — update the path based on your specific run directory.

TensorBoard displays validation loss, MAE, RMSE, and SSIM metrics across all folds.

---

## Design Notes

* **Model**: The decoder transforms a 3D parameter vector into a 32×32 image using upsampling and convolutional layers.
* **Loss Function**: A hybrid of MAE and SSIM is used to balance pixel-level accuracy and structural fidelity.
* **Data Handling**: Simulation folders are named like `sim_100_300_95`, from which average keyhole images are computed across a selected time window.
* **Scalability**: Ensemble prediction is built-in, averaging over all trained fold models.
* **Reproducibility**: Each fold logs its config and results. The fold assignments are saved for validation consistency.
