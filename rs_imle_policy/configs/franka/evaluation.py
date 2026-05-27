from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from rs_imle_policy.configs.train_config import VisionConfig


@dataclass
class EvaluationConfig:
    task_name: str = "example"
    output_dir: Path = Path("experiments/franka_evaluation")
    dataset_path: Optional[Path] = None
    robot_ip: str = "172.16.0.2"
    n_experiments: int = 10
    vision: VisionConfig = field(default_factory=VisionConfig)
