import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import tyro
from InquirerPy import inquirer


RESULTS_FILENAME = "evaluation_results.json"
METADATA_FILENAME = "run_metadata.json"


@dataclass
class Args:
    media_root: Path = Path("saved_evaluation_media")
    evaluation: Path | None = None
    output: Path | None = None
    show: bool = True
    match_experiment_ids: bool = False


@dataclass
class RunSummary:
    run_dir: Path
    label: str
    epoch: int
    results_by_id: dict[int, bool]
    raw_result_count: int
    duplicate_ids: list[int]
    evaluation_manifest: str | None


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def direct_run_dirs(evaluation_dir: Path) -> list[Path]:
    return sorted(
        child
        for child in evaluation_dir.iterdir()
        if child.is_dir() and (child / RESULTS_FILENAME).exists()
    )


def run_dirs(evaluation_dir: Path) -> list[Path]:
    direct_runs = direct_run_dirs(evaluation_dir)
    if direct_runs:
        return direct_runs

    if (evaluation_dir / RESULTS_FILENAME).exists():
        return [evaluation_dir]

    return sorted({path.parent for path in evaluation_dir.rglob(RESULTS_FILENAME)})


def evaluation_dirs(media_root: Path) -> list[Path]:
    if not media_root.exists():
        raise FileNotFoundError(f"Saved evaluation media folder not found: {media_root}")
    if not media_root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {media_root}")

    candidates = []
    if direct_run_dirs(media_root) or (media_root / RESULTS_FILENAME).exists():
        candidates.append(media_root)

    for child in sorted(media_root.iterdir()):
        if child.is_dir() and run_dirs(child):
            candidates.append(child)

    return candidates


def resolve_evaluation_dir(args: Args) -> Path:
    if args.evaluation is not None:
        if args.evaluation.exists():
            return args.evaluation

        candidate = args.media_root / args.evaluation
        if candidate.exists():
            return candidate

        raise FileNotFoundError(
            f"Evaluation folder not found: {args.evaluation} or {candidate}"
        )

    candidates = evaluation_dirs(args.media_root)
    if not candidates:
        raise FileNotFoundError(
            f"No folders containing {RESULTS_FILENAME} found under {args.media_root}"
        )
    if len(candidates) == 1:
        return candidates[0]

    choices = [
        {
            "name": f"{candidate.relative_to(args.media_root)} ({len(run_dirs(candidate))} runs)",
            "value": candidate,
        }
        for candidate in candidates
    ]
    return inquirer.select(
        message="Select evaluation:",
        choices=choices,
    ).execute()


def parse_epoch(run_dir: Path, metadata: dict) -> int | None:
    checkpoint = metadata.get("checkpoint", {})
    epoch = checkpoint.get("epoch")
    if epoch is not None:
        return int(epoch)

    label = checkpoint.get("label")
    if isinstance(label, str):
        match = re.search(r"epoch_(\d+)", label)
        if match is not None:
            return int(match.group(1))

    match = re.search(r"epoch_(\d+)", run_dir.name)
    if match is not None:
        return int(match.group(1))

    return None


def run_label(run_dir: Path, metadata: dict) -> str:
    base_run_name = metadata.get("base_run_name")
    if base_run_name:
        return str(base_run_name)

    return re.sub(r"_epoch_\d+$", "", run_dir.name)


def load_run_summary(run_dir: Path) -> RunSummary | None:
    results_path = run_dir / RESULTS_FILENAME
    metadata_path = run_dir / METADATA_FILENAME
    if not results_path.exists():
        return None

    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    epoch = parse_epoch(run_dir, metadata)
    if epoch is None:
        print(f"Skipping {run_dir}: could not determine checkpoint epoch.")
        return None

    results = load_json(results_path)
    if not isinstance(results, list):
        raise ValueError(f"Expected {results_path} to contain a JSON list.")

    results_by_id: dict[int, bool] = {}
    duplicate_ids = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        if "experiment_index" not in result or "success" not in result:
            continue

        experiment_id = int(result["experiment_index"])
        if experiment_id in results_by_id:
            duplicate_ids.add(experiment_id)
        results_by_id[experiment_id] = bool(result["success"])

    if not results_by_id:
        print(f"Skipping {run_dir}: no result entries with experiment_index and success.")
        return None

    return RunSummary(
        run_dir=run_dir,
        label=run_label(run_dir, metadata),
        epoch=epoch,
        results_by_id=results_by_id,
        raw_result_count=len(results),
        duplicate_ids=sorted(duplicate_ids),
        evaluation_manifest=metadata.get("evaluation_manifest"),
    )


def load_run_summaries(evaluation_dir: Path) -> list[RunSummary]:
    summaries = [
        summary
        for summary in (load_run_summary(run_dir) for run_dir in run_dirs(evaluation_dir))
        if summary is not None
    ]
    if not summaries:
        raise ValueError(f"No plottable evaluation runs found in {evaluation_dir}")
    return sorted(summaries, key=lambda summary: (summary.label, summary.epoch))


def matched_ids(summaries: list[RunSummary], enabled: bool) -> dict[Path, set[int]]:
    if not enabled:
        return {
            summary.run_dir: set(summary.results_by_id)
            for summary in summaries
        }

    common_ids = set(summaries[0].results_by_id)
    for summary in summaries[1:]:
        common_ids &= set(summary.results_by_id)

    if not common_ids:
        raise ValueError("No common experiment ids found across selected runs.")

    return {summary.run_dir: common_ids for summary in summaries}


def success_rate(summary: RunSummary, ids: set[int]) -> float:
    successes = sum(1 for experiment_id in ids if summary.results_by_id[experiment_id])
    return successes / len(ids)


def print_summary_table(
    summaries: list[RunSummary],
    ids_by_run: dict[Path, set[int]],
) -> None:
    print("Evaluation summary")
    print("label\tepoch\tsuccess_rate\tmatched_ids\traw_results\tfolder")
    for summary in summaries:
        ids = ids_by_run[summary.run_dir]
        rate = success_rate(summary, ids)
        duplicate_note = (
            f" duplicate_ids={summary.duplicate_ids}" if summary.duplicate_ids else ""
        )
        print(
            f"{summary.label}\t"
            f"{summary.epoch}\t"
            f"{rate:.3f}\t"
            f"{len(ids)}\t"
            f"{summary.raw_result_count}\t"
            f"{summary.run_dir}{duplicate_note}"
        )


def plot_success_rates(
    evaluation_dir: Path,
    summaries: list[RunSummary],
    ids_by_run: dict[Path, set[int]],
):
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = sorted({summary.label for summary in summaries})

    for label in labels:
        label_summaries = [
            summary for summary in summaries if summary.label == label
        ]
        label_summaries.sort(key=lambda summary: summary.epoch)

        epochs = [summary.epoch for summary in label_summaries]
        rates = [
            success_rate(summary, ids_by_run[summary.run_dir]) * 100.0
            for summary in label_summaries
        ]
        ax.plot(epochs, rates, marker="o", linewidth=2, label=label)

    ax.set_title(f"{evaluation_dir.name} success rate by checkpoint epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def main(args: Args) -> None:
    evaluation_dir = resolve_evaluation_dir(args)
    summaries = load_run_summaries(evaluation_dir)
    ids_by_run = matched_ids(summaries, args.match_experiment_ids)

    print(f"Loaded evaluation: {evaluation_dir}")
    if args.match_experiment_ids:
        common_ids = sorted(next(iter(ids_by_run.values())))
        print(f"Matched experiment ids: {common_ids}")
    print_summary_table(summaries, ids_by_run)

    fig = plot_success_rates(evaluation_dir, summaries, ids_by_run)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main(tyro.cli(Args))
