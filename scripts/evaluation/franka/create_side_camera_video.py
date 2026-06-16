import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
RESULTS_FILENAME = "evaluation_results.json"
DEFAULT_MEDIA_ROOTS = (Path("saved_media"), Path("saved_evaluation_media"))
DEFAULT_FPS = 10.0
TILE_WIDTH = 640
TILE_HEIGHT = 480
RESULT_OVERLAY_ALPHA = 0.45


@dataclass(frozen=True)
class VideoEntry:
    episode: int
    camera: str
    path: Path
    success: bool


def resolve_run_dir(path: Path) -> Path:
    if path.exists():
        return path

    for media_root in DEFAULT_MEDIA_ROOTS:
        candidate = media_root / path
        if candidate.exists():
            return candidate

    searched = ", ".join(str(root / path) for root in DEFAULT_MEDIA_ROOTS)
    raise FileNotFoundError(f"Run folder not found: {path} (also tried {searched})")


def split_episode_video(path: Path) -> tuple[int, str] | None:
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return None

    match = re.fullmatch(r"(\d+)_(.+)", path.stem)
    if match is None:
        return None

    return int(match.group(1)), match.group(2)


def load_episode_videos(run_dir: Path, camera_filter: str) -> dict[int, dict[str, Path]]:
    episodes: dict[int, dict[str, Path]] = {}
    for path in sorted(run_dir.iterdir()):
        parsed = split_episode_video(path)
        if parsed is None:
            continue

        episode, camera = parsed
        if camera_filter and camera_filter not in camera:
            continue

        episodes.setdefault(episode, {})[camera] = path

    if not episodes:
        filter_text = f" matching camera filter {camera_filter!r}" if camera_filter else ""
        raise ValueError(f"No episode camera videos found in {run_dir}{filter_text}.")

    return episodes


def load_results(run_dir: Path) -> dict[int, bool]:
    results_path = run_dir / RESULTS_FILENAME
    if not results_path.exists():
        return {}

    with results_path.open("r") as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError(f"Expected {results_path} to contain a JSON list.")

    success_by_id = {}
    for result in results:
        if not isinstance(result, dict) or "success" not in result:
            continue

        episode_id = result.get("experiment_index", result.get("episode"))
        if episode_id is None:
            continue

        success_by_id[int(episode_id)] = bool(result["success"])

    return success_by_id


def video_fps(captures: list[cv2.VideoCapture], fallback: float = DEFAULT_FPS) -> float:
    for capture in captures:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            return float(fps)
    return fallback


def video_lengths(captures: list[cv2.VideoCapture]) -> list[int]:
    lengths = []
    for capture in captures:
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if length <= 0:
            raise ValueError("Could not determine video length.")
        lengths.append(length)
    return lengths


def make_tile(frame: np.ndarray, label: str) -> np.ndarray:
    tile = cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT))
    cv2.rectangle(tile, (0, 0), (TILE_WIDTH, 34), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return tile


def make_grid(tiles: list[np.ndarray]) -> np.ndarray:
    cols = math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    blank = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)

    padded_tiles = tiles + [blank] * (rows * cols - len(tiles))
    row_images = []
    for row in range(rows):
        start = row * cols
        row_images.append(np.hstack(padded_tiles[start : start + cols]))
    return np.vstack(row_images)


def result_overlay_tile(entry: VideoEntry, frame: np.ndarray) -> np.ndarray:
    color = (40, 170, 40) if entry.success else (35, 35, 210)
    text = "SUCCESS" if entry.success else "FAIL"
    tile = cv2.resize(frame, (TILE_WIDTH, TILE_HEIGHT))
    color_overlay = np.full_like(tile, color, dtype=np.uint8)
    tile = cv2.addWeighted(color_overlay, RESULT_OVERLAY_ALPHA, tile, 1.0 - RESULT_OVERLAY_ALPHA, 0)

    label = f"Episode {entry.episode} {entry.camera}: {text}"
    cv2.rectangle(tile, (0, TILE_HEIGHT - 48), (TILE_WIDTH, TILE_HEIGHT), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, scale, thickness)
    x = max(0, (TILE_WIDTH - text_size[0]) // 2)
    y = TILE_HEIGHT - 16
    cv2.putText(
        tile,
        label,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return tile


def create_mosaic_video(
    entries: list[VideoEntry],
    output_path: Path,
    result_seconds: float,
) -> None:
    captures = [cv2.VideoCapture(str(entry.path)) for entry in entries]
    try:
        for entry, capture in zip(entries, captures):
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {entry.path}")

        fps = video_fps(captures)
        lengths = video_lengths(captures)
        total_frames = max(lengths)

        cols = math.ceil(math.sqrt(len(entries)))
        rows = math.ceil(len(entries) / cols)
        output_size = (cols * TILE_WIDTH, rows * TILE_HEIGHT)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
            fps,
            output_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")

        last_frames: list[np.ndarray | None] = [None] * len(captures)
        try:
            result_frame_count = max(1, round(result_seconds * fps))
            progress = tqdm(total=total_frames + result_frame_count, unit="frame", desc="Rendering mosaic")
            with progress:
                for frame_idx in range(total_frames):
                    tiles = []
                    for ix, (entry, capture, length) in enumerate(zip(entries, captures, lengths)):
                        if frame_idx < length:
                            ok, frame = capture.read()
                            if ok:
                                last_frames[ix] = frame

                        if last_frames[ix] is None:
                            raise RuntimeError(f"No readable frames for {entry.path}.")

                        if frame_idx < length:
                            tiles.append(
                                make_tile(last_frames[ix], f"Episode {entry.episode} {entry.camera}")
                            )
                        else:
                            tiles.append(result_overlay_tile(entry, last_frames[ix]))

                    writer.write(make_grid(tiles))
                    progress.update()

                result_tiles = []
                for entry, frame in zip(entries, last_frames):
                    if frame is None:
                        raise RuntimeError(f"No final frame available for {entry.path}.")
                    result_tiles.append(result_overlay_tile(entry, frame))

                end_frame = make_grid(result_tiles)
                for _ in range(result_frame_count):
                    writer.write(end_frame)
                    progress.update()
        finally:
            writer.release()
    finally:
        for capture in captures:
            capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-episode mosaic videos from saved evaluation camera videos. "
            "Finished cameras hold their final frame with a green or red result overlay."
        )
    )
    parser.add_argument("run_dir", type=Path, help="Run folder, or path relative to saved_media.")
    parser.add_argument("--episode", type=int, help="Only render one episode id.")
    parser.add_argument(
        "--camera-filter",
        default="side",
        help="Only include cameras whose name contains this text. Use '' for all cameras.",
    )
    parser.add_argument("--output", type=Path, help="Output video path.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the run folder.")
    parser.add_argument("--result-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.result_seconds < 0:
        raise ValueError("--result-seconds must be non-negative.")

    run_dir = resolve_run_dir(args.run_dir)
    output_dir = args.output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episode_videos(run_dir, args.camera_filter)
    results = load_results(run_dir)
    episode_ids = [args.episode] if args.episode is not None else sorted(episodes)
    camera_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.camera_filter).strip("._-")
    output_prefix = f"{camera_label}_cameras" if camera_label else "all_cameras"

    entries = []
    for episode in episode_ids:
        if episode not in episodes:
            raise ValueError(f"Episode {episode} has no matching camera videos in {run_dir}.")
        if episode not in results:
            raise ValueError(
                f"Episode {episode} has no success result in {run_dir / RESULTS_FILENAME}."
            )

        for camera, path in sorted(episodes[episode].items()):
            entries.append(
                VideoEntry(
                    episode=episode,
                    camera=camera,
                    path=path,
                    success=results[episode],
                )
            )

    if args.output is not None:
        output_path = args.output
    else:
        episode_suffix = f"_{args.episode}" if args.episode is not None else ""
        output_path = output_dir / f"{output_prefix}_mosaic{episode_suffix}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_mosaic_video(entries, output_path, args.result_seconds)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
