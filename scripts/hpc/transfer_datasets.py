#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro
from InquirerPy import inquirer


@dataclass
class Config:
    remote: str = "hpc"
    local_root: Path = Path("~/panda_dc/data")
    remote_root: str = "/work/cyphy/robot_learning/single_panda"
    ignore_existing: bool = True


@dataclass(frozen=True)
class LocalDataset:
    path: Path
    modified_timestamp: float
    modified_date: str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def dated_label(self) -> str:
        return f"{self.name} ({self.modified_date})"

    def remote_path(self, config: Config) -> str:
        return f"{config.remote_root}/{self.name}"


def run(cmd: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def list_local_datasets(config: Config) -> list[LocalDataset]:
    local_root = config.local_root.expanduser()

    if not local_root.exists():
        raise RuntimeError(f"Local dataset root does not exist: {local_root}")

    datasets: list[LocalDataset] = []

    for path in local_root.iterdir():
        if not path.is_dir():
            continue

        modified_timestamp = path.stat().st_mtime
        modified_date = datetime.fromtimestamp(modified_timestamp).strftime("%d-%m-%Y")
        datasets.append(
            LocalDataset(
                path=path,
                modified_timestamp=modified_timestamp,
                modified_date=modified_date,
            )
        )

    return sorted(datasets, key=lambda dataset: dataset.modified_timestamp, reverse=True)


def transfer_dataset(dataset: LocalDataset, config: Config) -> None:
    remote_path = dataset.remote_path(config)

    run(["ssh", config.remote, f"mkdir -p {shlex.quote(remote_path)}"])

    cmd = [
        "rsync",
        "-avzP",
    ]

    if config.ignore_existing:
        cmd.append("--ignore-existing")

    cmd.extend(
        [
            f"{dataset.path}/",
            f"{config.remote}:{remote_path}/",
        ]
    )

    print()
    print(f"Transferring: {dataset.dated_label}")
    print(" ".join(cmd))
    print()

    run(cmd)


def main(config: Config) -> None:
    datasets = list_local_datasets(config)

    if not datasets:
        raise RuntimeError(f"No datasets found under {config.local_root.expanduser()}")

    choices = [
        {
            "name": f"{dataset.dated_label} -> {dataset.path}",
            "value": dataset,
        }
        for dataset in datasets
    ]

    selected: list[LocalDataset] = inquirer.checkbox(
        message="Which datasets do you want to transfer to HPC?",
        choices=choices,
        instruction="Use <space> to select, <enter> to confirm",
        validate=lambda result: len(result) > 0,
        invalid_message="Select at least one dataset.",
    ).execute()

    for dataset in selected:
        transfer_dataset(dataset, config)

    print()
    print("Transferred datasets:")
    for dataset in selected:
        print(f"  - {dataset.dated_label}: {dataset.path} -> {config.remote}:{dataset.remote_path(config)}")


if __name__ == "__main__":
    main(tyro.cli(Config))
