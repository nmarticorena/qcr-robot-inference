#!/usr/bin/env python3

import subprocess
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro
from InquirerPy import inquirer


@dataclass
class Config:
    local_root: Path = Path("./data/")
    remote: str = "hpc"
    remote_root: str = "/work/cyphy/robot_learning/single_panda"
    ignore_existing: bool = True


@dataclass(frozen=True)
class RemoteDataset:
    path: str
    modified_timestamp: float
    modified_date: str
    
    @property
    def dated_label(self) -> str:
        return f"{self.path} ({self.modified_date})"

    def remote_path(self, config: Config) -> str:
        return f"{config.remote_root}/{self.path}"


def run(cmd: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def list_remote_datasets(config: Config) -> list[RemoteDataset]:
    cmd = [
        "ssh", 
        config.remote, 
        (
            f"cd {config.remote_root} || exit 1; "
            "find . -mindepth 1 -maxdepth 1 -type d "
            "-printf '%T@\t%Td-%Tm-%TY\t%p\n'"
        ),
    ]

    output = run(cmd, capture=True)

    datasets: list[RemoteDataset] = []

    for line in output.splitlines():
        line = line.strip()

        if not line:
            continue

        fields = line.split("\t", maxsplit=2)

        if len(fields) != 3:
            continue

        timestamp, modified_date, path = fields

        datasets.append(
            RemoteDataset(
                path=path,
                modified_timestamp=float(timestamp),
                modified_date=modified_date,
            )
        )

    return sorted(datasets, key=lambda dataset: dataset.modified_timestamp, reverse=True)


def transfer_dataset(dataset: RemoteDataset, config: Config) -> None:
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
            f"{config.remote}:{remote_path}/",
            f"{config.local_root}/{dataset.path}/",
        ]
    )

    print()
    print(f"Transferring: {dataset.dated_label}")
    print(" ".join(cmd))
    print()

    run(cmd)


def main(config: Config) -> None:
    datasets = list_remote_datasets(config)

    if not datasets:
        raise RuntimeError(f"No datasets found under {config.local_root.expanduser()}")

    choices = [
        {
            "name": f"{dataset.dated_label} -> {config.local_root}/{dataset.path}",
            "value": dataset,
        }
        for dataset in datasets
    ]

    selected: list[RemoteDataset] = inquirer.checkbox(
        message="Which datasets do you want to transfer from HPC?",
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
