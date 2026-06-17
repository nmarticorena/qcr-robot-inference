
import tyro

from rs_imle_policy.configs.experiment_configs import FrankaExperimentConfigChoice
from rs_imle_policy.datasets.single_franka import PandaPolicyDataset
from rs_imle_policy.train import run_training


def main(config: FrankaExperimentConfigChoice) -> None:
    train_dataset, val_dataset = PandaPolicyDataset.train_val(config, val_episode_count=10)
    run_training(config, train_dataset, val_dataset=val_dataset)


if __name__ == "__main__":
    main(tyro.cli(FrankaExperimentConfigChoice))
