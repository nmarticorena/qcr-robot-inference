#!/usr/bin/env python3

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from InquirerPy import inquirer


REMOTE = "hpc"
REMOTE_ROOT = "repos/qcr-robot-inference/saved_weights"
LOCAL_ROOT = Path("saved_weights")

@dataclass(frozen=True)
class RemoteExperiment:
    group: str
    name: str

    @property
    def remote_path(self) -> str:
        return f"{REMOTE_ROOT}/{self.group}/{self.name}"

    @property
    def local_path(self) -> Path:
        return LOCAL_ROOT / self.group / self.name

    @property
    def label(self) -> str:
        return f"{self.group}/{self.name}"


def run(cmd: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def list_remote_experiments() -> list[RemoteExperiment]:
    cmd = [
        "ssh",
        REMOTE,
        (
            f"cd {REMOTE_ROOT} || exit 1; "
            "find . -mindepth 2 -maxdepth 2 -type d | sort"
        ),
    ]

    output = run(cmd, capture=True)

    experiments: list[RemoteExperiment] = []

    for line in output.splitlines():
        line = line.strip().removeprefix("./")

        if not line:
            continue

        parts = line.split("/")

        if len(parts) != 2:
            continue

        group, name = parts
        experiments.append(RemoteExperiment(group=group, name=name))

    return experiments


def transfer_experiment(exp: RemoteExperiment) -> None:
    exp.local_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rsync",
        "-avzP",
        "--include=*/",
        "--include=*_last.pth",
        "--include=*last*.pth",
        "--include=*.yaml",
        "--include=*.yml",
        "--include=*.pkl",
        "--exclude=*",
        f"{REMOTE}:{exp.remote_path}/",
        f"{exp.local_path}/",
    ]

    print()
    print(f"Transferring: {exp.label}")
    print(" ".join(cmd))
    print()

    run(cmd)


def main() -> None:
    experiments = list_remote_experiments()

    if not experiments:
        raise RuntimeError(f"No experiments found under {REMOTE}:{REMOTE_ROOT}")

    choices = [
        {
            "name": exp.label,
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

    for exp in selected:
        transfer_experiment(exp)

    print()
    print("Transferred experiments:")
    for exp in selected:
        print(f"  - {exp.label} -> {exp.local_path}")


if __name__ == "__main__":
    main()
