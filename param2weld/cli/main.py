import argparse
import json
from pathlib import Path

from param2weld.config.config import Config
from param2weld.train.trainer import run_cv_training
from param2weld.utils.logger import get_logger

logger = get_logger("param2weld.cli")


def main():
    """
    Entry point for the command-line interface.

    Supports:
    - train: Run k-fold cross-validation using config defaults
             and optional overrides via a JSON hyperparameter file.
    """
    parser = argparse.ArgumentParser(description="Param2Weld CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Run k-fold training")
    train_parser.add_argument(
        "--data_dir", type=Path, required=True,
        help="Path to folder containing sim_* directories"
    )
    train_parser.add_argument(
        "--model_dir", type=Path, default=None,
        help="Optional directory to save models (default: config.model_dir)"
    )
    train_parser.add_argument(
        "--params_json", type=Path, default=None,
        help="Optional path to JSON file with hyperparameter overrides"
    )

    args = parser.parse_args()
    cfg = Config()

    # Collect training folders, skipping any holdout_* directories
    folders = [
        str(p) for p in args.data_dir.iterdir()
        if p.is_dir() and not p.name.startswith("holdout_")
    ]

    # Load optional parameter overrides from JSON
    params = {}
    if args.params_json:
        if not args.params_json.exists():
            raise FileNotFoundError(f"Config file not found: {args.params_json}")
        logger.info(f"Loading hyperparameters from: {args.params_json}")
        with open(args.params_json, "r") as f:
            params = json.load(f)

    # Start training
    run_cv_training(
        config=cfg,
        data_folders=folders,
        output_dir=args.model_dir,
        **params
    )


if __name__ == "__main__":
    main()
