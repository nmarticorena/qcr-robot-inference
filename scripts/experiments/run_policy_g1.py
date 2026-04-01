import re
import numpy as np
import os
import subprocess
import json

import tyro
import rerun as rr
from InquirerPy import inquirer


from rs_imle_policy.configs.default_configs import ExperimentConfigChoice  # noqa: F401
from rs_imle_policy.configs.train_config import LoaderConfig, ExperimentConfig, RSIMLE
from rs_imle_policy.inference_g1 import G1ArmsInferenceController

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

# rr.init("Robot Inference ", recording_id=exp_name)
rec = rr.RecordingStream("Robot Inference", recording_id = exp_name)
os.makedirs(f"saved_evaluation_media/{exp_name}", exist_ok=True)
rec.save(f"saved_evaluation_media/{exp_name}/rerun_recording.rrd")
rec.spawn()
subprocess.Popen(
    ["rerun", f"saved_evaluation_media/{exp_name}/rerun_recording.rrd"],
    shell=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

with open("home.json", "r") as f:
    home = np.array(json.load(f)["home"])
controller = G1ArmsInferenceController(
    config, rec, eval_name=exp_name, timeout=args.timeout, dry_run=args.dry_run, simulation=False, home = home
)

controller.run_experiments(10)
controller.perception_system.stop()
