import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from InquirerPy import inquirer

from rs_imle_policy.configs.panda_configs import SinglePandaConfig 

@dataclass
class DefaultEvaluationConfig:
    """Configuration for evaluation config"""
    path: Path
    epoch: Optional[int|str] = None
    timeout: int = 60  # Timeout for experiment in seconds
    episodes: int = 10  # exclusive max episode id to run
    initial_id: int = 0  # first episode/experiment id to run
    evaluation_path: Optional[Path] = None
    repeat_experiment_id: Optional[int] = None  # experiment id to repeat
    n_samples: int = 10  # total repeated-evaluation samples to collect
    silent_rerun: bool = True  # Whether to open or not the current rerun recording
    exp_name: Optional[str] = None
    dry_run: bool = False
    traj_consistency: bool = False # Only valid for RS-IMLE
    weight_path: Path = field(init=False)

    def __post_init__(self):
        if self.exp_name is None:
            self.exp_name = ask_run_name(self.path.name)
        if isinstance(self.epoch, int):
            self.epoch = str(self.epoch).zfill(4)
        if self.epoch is None:
            self.epoch = "last"
        self.run_name:str = sanitize_run_name(self.exp_name, self.epoch)
        self.weight_path = self.path / f"ema_net_epoch_{self.epoch}.pth"

        self.check()

    def check(self):
        if not self.weight_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.weight_path}")

@dataclass
class SinglePandaEvaluationConfig(DefaultEvaluationConfig):
    robot_config:SinglePandaConfig = field(default_factory = SinglePandaConfig)

def ask_run_name(default:str) -> str:
    exp_name = inquirer.text(
        "Enter the experiment name: ",
        default=default,
    ).execute()
    return exp_name

def sanitize_run_name(name: str, epoch: int | None | str) -> str:
    run_name = re.sub(r"\s+", "_", name.strip())
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)
    run_name = run_name.strip("._-")
    if not run_name:
        raise ValueError("Experiment name cannot be empty.")
    if epoch:
        run_name += f"_epoch_{epoch}"
    return run_name

if __name__ == "__main__":
    import tyro
    args = tyro.cli(SinglePandaEvaluationConfig)
    print(tyro.extras.to_yaml(args))
    print(args.run_name)
