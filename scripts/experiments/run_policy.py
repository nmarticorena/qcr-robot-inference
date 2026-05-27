import json
import re
import os
import subprocess
from pathlib import Path

import tyro
import rerun as rr
from InquirerPy import inquirer


from rs_imle_policy.configs.experiment_configs import FrankaExperimentConfigChoice  # noqa: F401
from rs_imle_policy.configs.train_config import LoaderConfig, ExperimentConfig, RSIMLE
from rs_imle_policy.inference import RobotInferenceController


def resolve_evaluation_manifest(path: Path) -> Path:
    if path.is_dir():
        return path / "experiments.json"
    return path


def resolve_manifest_image_path(image_path: str, manifest_dir: Path) -> str:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return str(candidate)

    manifest_relative = (manifest_dir / candidate).resolve()
    if manifest_relative.exists():
        return str(manifest_relative)

    cwd_relative = candidate.resolve()
    if cwd_relative.exists():
        return str(cwd_relative)

    return str(manifest_relative)


def load_evaluation_setup(path: Path) -> dict:
    manifest_path = resolve_evaluation_manifest(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")

    with manifest_path.open("r") as f:
        manifest = json.load(f)

    experiments = manifest.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in evaluation manifest: {manifest_path}")

    manifest_dir = manifest_path.parent
    for experiment in experiments:
        experiment["images"] = {
            camera_name: resolve_manifest_image_path(image_path, manifest_dir)
            for camera_name, image_path in experiment.get("images", {}).items()
        }

    return manifest


def validate_evaluation_setup(manifest: dict, config: ExperimentConfig) -> None:
    config_cameras = tuple(config.data.vision.cameras)
    manifest_serials = manifest.get("camera_serials", {})
    config_serials = {
        params.name: params.serial_number for params in config.data.vision.cameras_params
    }
    available_reference_cameras = set()
    for camera_name in config_cameras:
        manifest_serial = manifest_serials.get(camera_name)
        config_serial = config_serials[camera_name]
        has_reference_image = any(
            camera_name in experiment.get("images", {})
            for experiment in manifest.get("experiments", [])
        )
        if has_reference_image:
            available_reference_cameras.add(camera_name)
        if manifest_serial is not None and manifest_serial != config_serial:
            raise ValueError(
                f"Camera serial mismatch for {camera_name}: "
                f"manifest={manifest_serial}, config={config_serial}"
            )
    if not available_reference_cameras:
        raise ValueError(
            "Evaluation manifest does not contain reference images for any of the "
            f"policy cameras {config_cameras}"
        )


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

evaluation_manifest = None
evaluation_experiments = None
evaluation_home_q = None
if args.evaluation_path is not None:
    evaluation_manifest = load_evaluation_setup(args.evaluation_path)
    validate_evaluation_setup(evaluation_manifest, config)
    evaluation_experiments = evaluation_manifest["experiments"]
    evaluation_home_q = evaluation_manifest.get("home_q")

rr.init("Robot Inference ", recording_id=exp_name)
os.makedirs(f"saved_evaluation_media/{exp_name}", exist_ok=True)
rr.save(f"saved_evaluation_media/{exp_name}/rerun_recording.rrd")
subprocess.Popen(
    ["rerun", f"saved_evaluation_media/{exp_name}/rerun_recording.rrd"],
    shell=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

controller = RobotInferenceController(
    config,
    eval_name=exp_name,
    timeout=args.timeout,
    dry_run=args.dry_run,
    home_q=evaluation_home_q,
)

try:
    if evaluation_experiments is not None:
        controller.run_evaluation_experiments(evaluation_experiments[: args.episodes])
    else:
        controller.run_experiments(args.episodes)
finally:
    controller.perception_system.stop()
