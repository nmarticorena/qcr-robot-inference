import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import tyro
from InquirerPy import inquirer


LEGACY_RESULTS_FILENAME = "evaluation_results.json"
ATTEMPT_RESULT_FILENAME = "result.json"
METADATA_FILENAME = "run_metadata.json"


@dataclass
class Args:
    media_root: Path = Path("saved_evaluation_media")
    evaluation: Path | None = None
    output: Path | None = None
    show: bool = True
    match_experiment_ids: bool = False


@dataclass(frozen=True)
class AttemptResult:
    experiment_id: int
    attempt_index: int | None
    success: bool
    path: Path


@dataclass
class RunSummary:
    run_dir: Path
    label: str
    epoch: int
    attempts: list[AttemptResult]
    raw_result_count: int
    duplicate_ids: list[int]
    evaluation_manifest: str | None
    source_format: str


def load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def parse_int(value: Any, fallback: int | None = None) -> int | None:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1", "success", "succeeded"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", "fail", "failed"}:
            return False
    return bool(value)


def episode_id_from_dir(path: Path) -> int | None:
    match = re.fullmatch(r"episode_(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def attempt_index_from_dir(path: Path) -> int | None:
    if not path.name.isdigit():
        return None
    return int(path.name)


def attempt_result_sort_key(path: Path) -> tuple[int, int, str]:
    episode_id = episode_id_from_dir(path.parent.parent)
    attempt_index = attempt_index_from_dir(path.parent)
    if episode_id is None:
        episode_id = episode_id_from_dir(path.parent)
    return (
        episode_id if episode_id is not None else 10**9,
        attempt_index if attempt_index is not None else 10**9,
        path.as_posix(),
    )


def attempt_result_paths(run_dir: Path) -> list[Path]:
    result_paths = []
    for episode_dir in sorted(run_dir.glob("episode_*")):
        if not episode_dir.is_dir() or episode_id_from_dir(episode_dir) is None:
            continue

        direct_result = episode_dir / ATTEMPT_RESULT_FILENAME
        if direct_result.exists():
            result_paths.append(direct_result)

        for attempt_dir in sorted(episode_dir.iterdir()):
            if not attempt_dir.is_dir() or attempt_index_from_dir(attempt_dir) is None:
                continue
            result_path = attempt_dir / ATTEMPT_RESULT_FILENAME
            if result_path.exists():
                result_paths.append(result_path)

    return sorted(result_paths, key=attempt_result_sort_key)


def is_run_dir(path: Path) -> bool:
    return (path / LEGACY_RESULTS_FILENAME).exists() or bool(attempt_result_paths(path))


def direct_run_dirs(evaluation_dir: Path) -> list[Path]:
    return sorted(child for child in evaluation_dir.iterdir() if child.is_dir() and is_run_dir(child))


def infer_run_dir_from_attempt_path(result_path: Path) -> Path | None:
    if result_path.name != ATTEMPT_RESULT_FILENAME:
        return None

    if (
        attempt_index_from_dir(result_path.parent) is not None
        and episode_id_from_dir(result_path.parent.parent) is not None
    ):
        return result_path.parent.parent.parent

    if episode_id_from_dir(result_path.parent) is not None:
        return result_path.parent.parent

    return None


def run_dirs(evaluation_dir: Path) -> list[Path]:
    direct_runs = direct_run_dirs(evaluation_dir)
    if direct_runs:
        return direct_runs

    if is_run_dir(evaluation_dir):
        return [evaluation_dir]

    nested_runs = {path.parent for path in evaluation_dir.rglob(LEGACY_RESULTS_FILENAME)}
    for result_path in evaluation_dir.rglob(ATTEMPT_RESULT_FILENAME):
        run_dir = infer_run_dir_from_attempt_path(result_path)
        if run_dir is not None:
            nested_runs.add(run_dir)

    return sorted(run_dir for run_dir in nested_runs if is_run_dir(run_dir))


def evaluation_dirs(media_root: Path) -> list[Path]:
    if not media_root.exists():
        raise FileNotFoundError(f"Saved evaluation media folder not found: {media_root}")
    if not media_root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {media_root}")

    candidates = []
    if is_run_dir(media_root) or direct_run_dirs(media_root):
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

        raise FileNotFoundError(f"Evaluation folder not found: {args.evaluation} or {candidate}")

    candidates = evaluation_dirs(args.media_root)
    if not candidates:
        raise FileNotFoundError(f"No folders containing evaluation results found under {args.media_root}")
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


def result_experiment_id(
    result: dict,
    fallback_id: int | None,
    fallback_index: int,
) -> int:
    experiment_id = parse_int(
        result.get("experiment_index", result.get("episode_id", result.get("episode"))),
        fallback_id,
    )
    if experiment_id is None:
        experiment_id = fallback_index
    return experiment_id


def result_attempt_index(
    result: dict,
    fallback_attempt_index: int | None,
    fallback_index: int,
) -> int | None:
    return parse_int(
        result.get("attempt_index", result.get("episode")),
        fallback_attempt_index if fallback_attempt_index is not None else fallback_index,
    )


def attempt_from_result(
    result: dict,
    path: Path,
    fallback_experiment_id: int | None,
    fallback_attempt_index: int | None,
    fallback_index: int,
) -> AttemptResult | None:
    if "success" not in result:
        return None

    return AttemptResult(
        experiment_id=result_experiment_id(result, fallback_experiment_id, fallback_index),
        attempt_index=result_attempt_index(result, fallback_attempt_index, fallback_index),
        success=parse_bool(result["success"]),
        path=path,
    )


def load_legacy_attempts(run_dir: Path) -> tuple[list[AttemptResult], int]:
    results_path = run_dir / LEGACY_RESULTS_FILENAME
    results = load_json(results_path)
    if not isinstance(results, list):
        raise ValueError(f"Expected {results_path} to contain a JSON list.")

    attempts = []
    for fallback_index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        attempt = attempt_from_result(
            result=result,
            path=results_path,
            fallback_experiment_id=None,
            fallback_attempt_index=None,
            fallback_index=fallback_index,
        )
        if attempt is not None:
            attempts.append(attempt)
    return attempts, len(results)


def load_per_attempt_results(run_dir: Path) -> tuple[list[AttemptResult], int]:
    attempts = []
    result_paths = attempt_result_paths(run_dir)
    for fallback_index, result_path in enumerate(result_paths):
        result = load_json(result_path)
        if not isinstance(result, dict):
            continue

        fallback_experiment_id = episode_id_from_dir(result_path.parent.parent)
        fallback_attempt_index = attempt_index_from_dir(result_path.parent)
        if fallback_experiment_id is None:
            fallback_experiment_id = episode_id_from_dir(result_path.parent)

        attempt = attempt_from_result(
            result=result,
            path=result_path,
            fallback_experiment_id=fallback_experiment_id,
            fallback_attempt_index=fallback_attempt_index,
            fallback_index=fallback_index,
        )
        if attempt is not None:
            attempts.append(attempt)

    return attempts, len(result_paths)


def load_run_attempts(run_dir: Path) -> tuple[list[AttemptResult], int, str]:
    if attempt_result_paths(run_dir):
        attempts, raw_count = load_per_attempt_results(run_dir)
        return attempts, raw_count, "per-attempt"

    attempts, raw_count = load_legacy_attempts(run_dir)
    return attempts, raw_count, "legacy-list"


def load_run_summary(run_dir: Path) -> RunSummary | None:
    metadata_path = run_dir / METADATA_FILENAME
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    epoch = parse_epoch(run_dir, metadata)
    if epoch is None:
        print(f"Skipping {run_dir}: could not determine checkpoint epoch.")
        return None

    attempts, raw_result_count, source_format = load_run_attempts(run_dir)
    if not attempts:
        print(f"Skipping {run_dir}: no result entries with success.")
        return None

    experiment_id_counts = Counter(attempt.experiment_id for attempt in attempts)
    duplicate_ids = sorted(experiment_id for experiment_id, count in experiment_id_counts.items() if count > 1)

    return RunSummary(
        run_dir=run_dir,
        label=run_label(run_dir, metadata),
        epoch=epoch,
        attempts=attempts,
        raw_result_count=raw_result_count,
        duplicate_ids=duplicate_ids,
        evaluation_manifest=metadata.get("evaluation_manifest"),
        source_format=source_format,
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


def experiment_ids(summary: RunSummary) -> set[int]:
    return {attempt.experiment_id for attempt in summary.attempts}


def matched_ids(summaries: list[RunSummary], enabled: bool) -> dict[Path, set[int]]:
    if not enabled:
        return {summary.run_dir: experiment_ids(summary) for summary in summaries}

    common_ids = experiment_ids(summaries[0])
    for summary in summaries[1:]:
        common_ids &= experiment_ids(summary)

    if not common_ids:
        raise ValueError("No common experiment ids found across selected runs.")

    return {summary.run_dir: common_ids for summary in summaries}


def selected_attempts(summary: RunSummary, ids: set[int]) -> list[AttemptResult]:
    return [attempt for attempt in summary.attempts if attempt.experiment_id in ids]


def success_rate(summary: RunSummary, ids: set[int]) -> float:
    attempts = selected_attempts(summary, ids)
    successes = sum(1 for attempt in attempts if attempt.success)
    return successes / len(attempts)


def print_summary_table(
    summaries: list[RunSummary],
    ids_by_run: dict[Path, set[int]],
) -> None:
    print("Evaluation summary")
    print("label\tepoch\tsuccess_rate\tattempts\tunique_ids\tmatched_ids\traw_results\tformat\tfolder")
    for summary in summaries:
        ids = ids_by_run[summary.run_dir]
        attempts = selected_attempts(summary, ids)
        rate = success_rate(summary, ids)
        duplicate_note = f" duplicate_ids={summary.duplicate_ids}" if summary.duplicate_ids else ""
        print(
            f"{summary.label}\t"
            f"{summary.epoch}\t"
            f"{rate:.3f}\t"
            f"{len(attempts)}\t"
            f"{len(experiment_ids(summary))}\t"
            f"{len(ids)}\t"
            f"{summary.raw_result_count}\t"
            f"{summary.source_format}\t"
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
        label_summaries = [summary for summary in summaries if summary.label == label]
        label_summaries.sort(key=lambda summary: summary.epoch)

        epochs = [summary.epoch for summary in label_summaries]
        rates = [success_rate(summary, ids_by_run[summary.run_dir]) * 100.0 for summary in label_summaries]
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
