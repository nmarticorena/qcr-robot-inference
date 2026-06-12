import json
import subprocess
from pathlib import Path
from typing import Optional

import rerun as rr
import tyro

from rs_imle_policy.configs.eval_config import SinglePandaEvaluationConfig
from rs_imle_policy.configs.train_config import ExperimentConfig
from rs_imle_policy.inference import RobotInferenceController
from rs_imle_policy.configs.experiment_configs import FrankaExperimentConfigChoice # noqa: F401


class IndividualPolicyEvaluation:
    """Minimal policy evaluation runner"""

    def __init__(self, args: SinglePandaEvaluationConfig):
        self.args = args
        self.media_dir = Path("saved_evaluation_media")/ args.run_name
        self.media_dir.mkdir(parents=True, exist_ok=True)

        self.policy_config = self.load_policy_config()
        self.evaluation_manifest = self._load_evaluation_manifest()
        self.home_q = None if self.evaluation_manifest is None else self.evaluation_manifest.get("home_q")

    def run(self) -> None:
        recording_path = self._start_rerun_recording()
        print(f"Saving evaluation media to: {self.media_dir}")
        print(f"Saving Rerun recording to: {recording_path}")
        print(f"Loading checkpoint: {self.args.weight_path}")

        controller = RobotInferenceController(
            self.policy_config,
            eval_details=self.args,
            media_dir=self.media_dir,
            home_q=self.home_q,
            folder=self.args.path,
        )

        try:
            if self.evaluation_manifest is None:
                controller.run_experiments(self.args.episodes, initial_id=self.args.initial_id)
            elif self.args.repeat_experiment_id is not None:
                controller.run_evaluation_until_n_samples(
                    self._find_experiment(self.args.repeat_experiment_id),
                    self.args.n_samples,
                )
            else:
                controller.run_evaluation_experiments(self._select_experiments())
        finally:
            controller.perception_system.stop()

    def load_policy_config(self) -> ExperimentConfig:
        with (self.args.path / "config.yaml").open("r") as f:
            config = tyro.extras.from_yaml(ExperimentConfig, f)
        config.epoch = self._policy_epoch()
        return config

    def _policy_epoch(self) -> Optional[int]:
        if self.args.epoch == "last":
            return None
        return int(self.args.epoch)

    def _manifest_path(self) -> Optional[Path]:
        if self.args.evaluation_path is None:
            return None
        if self.args.evaluation_path.is_dir():
            return self.args.evaluation_path / "experiments.json"
        return self.args.evaluation_path

    def _load_evaluation_manifest(self) -> Optional[dict]:
        manifest_path = self._manifest_path()
        if manifest_path is None:
            return None

        with manifest_path.open("r") as f:
            manifest = json.load(f)

        manifest_dir = manifest_path.parent
        for experiment in manifest["experiments"]:
            experiment["images"] = {
                camera_name: self._resolve_image_path(image_path, manifest_dir)
                for camera_name, image_path in experiment.get("images", {}).items()
            }
        return manifest

    @staticmethod
    def _resolve_image_path(image_path: str, manifest_dir: Path) -> str:
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

    def _select_experiments(self) -> list[dict]:
        assert self.evaluation_manifest is not None
        selected = []
        for fallback_index, experiment in enumerate(self.evaluation_manifest["experiments"]):
            episode_id = int(experiment.get("index", fallback_index))
            if self.args.initial_id <= episode_id < self.args.episodes:
                experiment = dict(experiment)
                experiment.setdefault("index", episode_id)
                selected.append(experiment)
        return selected

    def _find_experiment(self, episode_id: int) -> dict:
        assert self.evaluation_manifest is not None
        for fallback_index, experiment in enumerate(self.evaluation_manifest["experiments"]):
            candidate_id = int(experiment.get("index", fallback_index))
            if candidate_id == episode_id:
                experiment = dict(experiment)
                experiment.setdefault("index", candidate_id)
                return experiment
        raise ValueError(f"No evaluation experiment with id {episode_id}.")

    def _start_rerun_recording(self) -> Path:
        recording_path = self.media_dir / "rerun_recording.rrd"
        rr.init("Robot Inference", recording_id=self.args.run_name)
        rr.save(str(recording_path))
        if not self.args.silent_rerun:
            subprocess.Popen(
                ["rerun", str(recording_path)],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return recording_path


def main() -> None:
    IndividualPolicyEvaluation(tyro.cli(SinglePandaEvaluationConfig)).run()


if __name__ == "__main__":
    main()
