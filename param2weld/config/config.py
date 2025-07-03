from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    # Image resolution
    resolution: int = 32

    # Hybrid loss weights
    w_mae: float = 0.8
    w_ssim: float = 0.2

    # Cross-validation
    kfold_splits: int = 10
    seed: int = 42

    # Default laser parameter
    default_spotsize: float = 95.0

    # Directory paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.model_dir = self.project_root / "models"
        self.log_dir = self.project_root / "logs"
