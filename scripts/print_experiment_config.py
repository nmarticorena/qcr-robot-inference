from dataclasses import dataclass
from pathlib import Path

import tyro

from rs_imle_policy.configs.experiment_configs import *  # noqa: F401
from rs_imle_policy.configs.train_config import ExperimentConfig


@dataclass
class Args:
    path: Path


def resolve_config_path(path: Path) -> Path:
    if path.is_dir():
        return path / "config.yaml"
    return path


def main(args: Args) -> None:
    config_path = resolve_config_path(args.path)
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config file at {config_path}")

    with config_path.open("r") as f:
        config = tyro.extras.from_yaml(ExperimentConfig, f)

    print(config)



if __name__ == "__main__":
    main(tyro.cli(Args))
