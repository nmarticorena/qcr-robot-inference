#!/usr/bin/env python3

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
from InquirerPy import inquirer


@dataclass
class Config:
    remote: str = "hpc"
    remote_root: str = "repos/qcr-robot-inference/saved_weights"
    local_root: Path = Path("saved_weights")
    epochs: Optional[tuple[int, ...]] = None
    include_last: bool = False


@dataclass(frozen=True)
class RemoteExperiment:
    group: str
    name: str
    modified_timestamp: float
    modified_date: str

    @property
    def label(self) -> str:
        return f"{self.group}/{self.name}"

    @property
    def dated_label(self) -> str:
        return f"{self.label} ({self.modified_date})"

    def remote_path(self, config: Config) -> str:
        return f"{config.remote_root}/{self.group}/{self.name}"

    def local_path(self, config: Config) -> Path:
        return config.local_root / self.group / self.name


def run(cmd: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def weight_include_patterns(config: Config) -> list[str]:
    patterns: list[str] = []

    if config.include_last:
        patterns.append("ema_net_epoch_last.pth")

    for epoch in config.epochs:
        if epoch < 0:
            raise ValueError(f"Epochs must be non-negative, got {epoch}.")

        patterns.append(f"ema_net_epoch_{epoch:04d}.pth")

    return patterns


def list_remote_experiments(config: Config) -> list[RemoteExperiment]:
    cmd = [
        "ssh",
        config.remote,
        (
            f"cd {config.remote_root} || exit 1; "
            "find . -mindepth 2 -maxdepth 2 -type d "
            "-printf '%T@\t%Td-%Tm-%TY\t%p\n'"
        ),
    ]

    output = run(cmd, capture=True)

    experiments: list[RemoteExperiment] = []

    for line in output.splitlines():
        line = line.strip()

        if not line:
            continue

        fields = line.split("\t", maxsplit=2)

        if len(fields) != 3:
            continue

        timestamp, modified_date, path = fields
        parts = path.removeprefix("./").split("/")

        if len(parts) != 2:
            continue

        group, name = parts
        experiments.append(
            RemoteExperiment(
                group=group,
                name=name,
                modified_timestamp=float(timestamp),
                modified_date=modified_date,
            )
        )

    return sorted(experiments, key=lambda exp: exp.modified_timestamp, reverse=True)

def get_available_epochs(config: Config, choices: list[RemoteExperiment]) -> None:
    def check_cmd(exp: RemoteExperiment) -> list[str]:
        cmd = [
            "ssh",
            config.remote,
            (
                f"ls {config.remote_root}/{exp.group}/{exp.name}/ema_net_epoch_*.pth"
            ),
        ]
        return cmd
    
    def get_epochs(remote_experiment: RemoteExperiment) -> list[int]:
        output = run(check_cmd(remote_experiment), capture=True)
        files = output.splitlines()
        epochs = []
        for file in files:
            filename = file.strip().split("/")[-1]
            if filename.startswith("ema_net_epoch_") and filename.endswith(".pth"):
                epoch_str = filename[len("ema_net_epoch_"):-len(".pth")]
                if epoch_str.isdigit():
                    epochs.append(int(epoch_str))
        return epochs

    ephocs = [set(get_epochs(exp)) for exp in choices]
    common_epochs = set.intersection(*ephocs) if ephocs else set()
    common_epochs = sorted(common_epochs, reverse=True)

    selected_epochs = inquirer.checkbox(
        message="Which epochs do you want to transfer?",
        choices=[{"name": str(epoch), "value": epoch} for epoch in sorted(common_epochs)],
        instruction="Use <space> to select, <enter> to confirm",
        validate=lambda result: len(result) > 0,
        invalid_message="Select at least one epoch.",
    ).execute()

    return selected_epochs
    





def transfer_experiment(exp: RemoteExperiment, config: Config) -> None:
    local_path = exp.local_path(config)
    local_path.mkdir(parents=True, exist_ok=True)

    weight_includes = [
        f"--include={pattern}" for pattern in weight_include_patterns(config)
    ]

    cmd = [
        "rsync",
        "-avzP",
        "--include=*/",
        *weight_includes,
        "--include=*.yaml",
        "--include=*.yml",
        "--include=*.pkl",
        "--exclude=*",
        f"{config.remote}:{exp.remote_path(config)}/",
        f"{local_path}/",
    ]

    print()
    print(f"Transferring: {exp.label}")
    print(" ".join(cmd))
    print()

    run(cmd)


def main(config: Config) -> None:
    experiments = list_remote_experiments(config)

    if not experiments:
        raise RuntimeError(f"No experiments found under {config.remote}:{config.remote_root}")

    choices = [
        {
            "name": f"{exp.dated_label} -> {exp.remote_path(config)}",
            "value": exp,
        }
        for exp in experiments
    ]

    selected: list[RemoteExperiment] = inquirer.checkbox(
        message="Which experiments do you want to transfer?",
        choices=choices,
        instruction="Use <space> to select, <enter> to confirm",
        validate=lambda result: len(result) > 0,
        invalid_message="Select at least one experiment.",
    ).execute()

    if config.epochs is None:
        config.epochs = get_available_epochs(config, selected)
        

    for exp in selected:
        transfer_experiment(exp, config)

    print()
    print("Transferred experiments:")
    for exp in selected:
        print(f"  - {exp.dated_label}: {exp.remote_path(config)} -> {exp.local_path(config)}")


if __name__ == "__main__":
    main(tyro.cli(Config))
