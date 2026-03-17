import re
import os
import subprocess

import tyro
import rerun as rr
from InquirerPy import inquirer


from rs_imle_policy.configs.default_configs import ExperimentConfigChoice  # noqa: F401
from rs_imle_policy.configs.train_config import LoaderConfig, ExperimentConfig, RSIMLE
from rs_imle_policy.inference import RobotInferenceController

args = tyro.cli(LoaderConfig)


if args.exp_name is not None:
    exp_name = args.exp_name
else:
    exp_name = inquirer.text("Enter the experiment name: ").execute()
    exp_name = re.sub(r"\s+", "_", exp_name.strip())

config = tyro.extras.from_yaml(ExperimentConfig, open(args.path / "config.yaml"))
config.epoch = args.epoch
if isinstance(config.model, RSIMLE):
    config.model.traj_consistency = True

rr.init("Robot Inference ", recording_id=exp_name)
os.makedirs(f"saved_evaluation_media/{exp_name}", exist_ok=True)
rr.save(f"saved_evaluation_media/{exp_name}/rerun_recording.rrd")
subprocess.Popen(
    ["rerun", f"saved_evaluation_media/{exp_name}/rerun_recording.rrd"],
    shell=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

controller = RobotInferenceController(config, eval_name=exp_name, timeout=args.timeout, dry_run=args.dry_run)

controller.run_experiments(10)
controller.perception_system.stop()
