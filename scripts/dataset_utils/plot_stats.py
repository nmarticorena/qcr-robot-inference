from dataclasses import dataclass
from pathlib import Path
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclass
class Args:
    path: Path
    save_path: Path | None = None


def resolve_stats_path(path: Path) -> Path:
    if path.is_dir():
        return path / "stats.pkl"
    return path


def load_stats(path: Path) -> dict:
    stats_path = resolve_stats_path(path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Could not find stats file at {stats_path}")
    with stats_path.open("rb") as f:
        return pickle.load(f)


def main(args: Args) -> None:
    stats = load_stats(args.path)
    keys = list(stats.keys())
    if not keys:
        raise ValueError("stats.pkl is empty")

    ncols = 2
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, key in zip(axes, keys):
        mins = np.asarray(stats[key]["min"]).reshape(-1)
        maxs = np.asarray(stats[key]["max"]).reshape(-1)
        idx = np.arange(len(mins))

        ax.plot(idx, mins, marker="o", label="min")
        ax.plot(idx, maxs, marker="o", label="max")
        ax.fill_between(idx, mins, maxs, alpha=0.2)
        ax.set_title(key)
        ax.set_xlabel("dimension")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[len(keys) :]:
        ax.axis("off")

    fig.suptitle(str(resolve_stats_path(args.path)), fontsize=12)
    fig.tight_layout()

    if args.save_path is not None:
        fig.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"saved plot to {args.save_path}")

    plt.show()


if __name__ == "__main__":
    main(tyro.cli(Args))
