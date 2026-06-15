
import tyro

from rs_imle_policy.configs.experiment_configs import FrankaExperimentConfigChoice
from rs_imle_policy.train import build_franka_dataset, run_training


def main(config: FrankaExperimentConfigChoice) -> None:
    dataset = build_franka_dataset(config)
    run_training(config, dataset)


if __name__ == "__main__":
    main(tyro.cli(FrankaExperimentConfigChoice))
