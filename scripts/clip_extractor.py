#!/usr/bin/env python3
"""
Clip extractor for sessions.zip input. Need ffmpeg.

Usage examples:
  python scripts/clip_extractor.py --zip dataset/example/sessions.zip --output out/
  python scripts/clip_extractor.py --zip sessions.zip --output out/ --export-clips --export-ratio 0.02
  python scripts/clip_extractor.py --zip sessions.zip --output out/ --allow-partial

Outputs:
  - clip_index.jsonl
  - optional per-sample folders under output_dir/clips
  - optional frames cache under output_dir/frames/<session_id>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable


def _read_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            lines.append(line)
    return lines


def _ensure_frames(video_path: Path, frames_dir: Path, fps: int) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in frames_dir.glob("*.jpg") if p.is_file())
    if existing:
        return existing
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; provide pre-extracted frames.")
    output_pattern = frames_dir / "%06d.jpg"
    cmd = [
        ffmpeg,
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        "-start_number",
        "0",
        str(output_pattern),
    ]
    subprocess.run(cmd, check=True)
    return sorted(p for p in frames_dir.glob("*.jpg") if p.is_file())


def _relative_paths(paths: list[Path], root: Path) -> list[str]:
    return [str(p.relative_to(root)) for p in paths]


def _indices_recent(t: int) -> list[int]:
    return list(range(t - 7, t + 1))


def _indices_lookahead(t: int) -> list[int]:
    return list(range(t, t + 8))


def _indices_summary(t: int) -> list[int]:
    # 60s window at 2FPS -> 120 frames; sample every 2s -> 30 frames
    return list(range(t - 120, t, 4))


def _indices_lookahead_summary(t: int) -> list[int]:
    return list(range(t, t + 120, 4))


def _all_in_range(indices: Iterable[int], start: int, end: int) -> bool:
    return all(start <= idx <= end for idx in indices)


def _extract_zip(zip_path: Path, output_dir: Path) -> Path:
    unpack_dir = output_dir / "_sessions" / zip_path.stem
    if unpack_dir.exists():
        return unpack_dir
    unpack_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(unpack_dir)
    return unpack_dir


def _find_sessions_root(unpack_dir: Path) -> Path:
    candidate = unpack_dir / "sessions"
    if candidate.exists():
        return candidate
    return unpack_dir


def _link_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown link mode: {mode}")


def _clip_indices(indices: Iterable[int], count: int) -> list[int]:
    return [idx for idx in indices if 0 <= idx < count]


def extract_clips(
    zip_path: Path,
    output_dir: Path,
    fps: int = 2,
    step: int = 2,
    allow_partial: bool = False,
    export_clips: bool = False,
    export_ratio: float = 0.01,
    link_mode: str = "hardlink",
    seed: int = 0,
) -> tuple[Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    unpack_dir = _extract_zip(zip_path, output_dir)
    sessions_root = _find_sessions_root(unpack_dir)
    if not sessions_root.exists():
        raise ValueError("sessions root not found after unzip.")

    index_path = output_dir / "clip_index.jsonl"
    samples_written = 0

    with index_path.open("w", encoding="utf-8") as index_handle:
        for session_dir in sorted(p for p in sessions_root.iterdir() if p.is_dir()):
            session_id = session_dir.name
            video_path = session_dir / "video.mp4"
            compiled_actions = session_dir / "compiled_actions.jsonl"
            goal_path = session_dir / "goal.jsonl"
            instruct_path = session_dir / "labeling_instruct.jsonl"

            if not video_path.exists() or not compiled_actions.exists():
                print(f"skip {session_id}: missing video or compiled_actions", file=sys.stderr)
                continue

            frames_dir = output_dir / "frames" / session_id
            frames = _ensure_frames(video_path, frames_dir, fps)
            actions = _read_lines(compiled_actions)
            if not frames or not actions:
                print(f"skip {session_id}: no frames or actions", file=sys.stderr)
                continue

            count = min(len(frames), len(actions))
            if len(frames) != len(actions):
                print(
                    f"warning: {session_id} frames({len(frames)}) != actions({len(actions)}); using {count}",
                    file=sys.stderr,
                )
            frames = frames[:count]
            actions = actions[:count]

            goals = _read_lines(goal_path) if goal_path.exists() else []
            instructs = _read_lines(instruct_path) if instruct_path.exists() else []

            if allow_partial:
                min_t = 0
                max_t = count - 1
            else:
                min_t = 120
                max_t = count - 117
            if min_t >= max_t:
                print(f"warning: {session_id} not enough frames for windows", file=sys.stderr)
                continue

            for t in range(min_t, max_t + 1, step):
                recent_idx = _indices_recent(t)
                summary_idx = _indices_summary(t)
                lookahead_idx = _indices_lookahead(t)
                lookahead_summary_idx = _indices_lookahead_summary(t)

                if allow_partial:
                    recent_idx = _clip_indices(recent_idx, count)
                    summary_idx = _clip_indices(summary_idx, count)
                    lookahead_idx = _clip_indices(lookahead_idx, count)
                    lookahead_summary_idx = _clip_indices(lookahead_summary_idx, count)
                else:
                    if recent_idx[0] < 0 or lookahead_summary_idx[-1] >= count:
                        continue

                sample_id = f"{session_id}_t{t:06d}"
                record = {
                    "sample_id": sample_id,
                    "session_id": session_id,
                    "anchor_t": t,
                    "recent_clip": _relative_paths([frames[i] for i in recent_idx], output_dir),
                    "summary_clip": _relative_paths([frames[i] for i in summary_idx], output_dir),
                    "lookahead_clip": _relative_paths([frames[i] for i in lookahead_idx], output_dir),
                    "lookahead_summary_clip": _relative_paths(
                        [frames[i] for i in lookahead_summary_idx], output_dir
                    ),
                    "action_t": actions[t],
                    "goal_t": goals[t] if t < len(goals) else "",
                    "instruct_t": instructs[t] if t < len(instructs) else "",
                }

                index_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                samples_written += 1

                if export_clips and rng.random() <= export_ratio:
                    sample_dir = output_dir / "clips" / sample_id
                    recent_dir = sample_dir / "recent"
                    summary_dir = sample_dir / "summary"
                    lookahead_dir = sample_dir / "lookahead"
                    lookahead_summary_dir = sample_dir / "lookahead_summary"
                    for src in [frames[i] for i in recent_idx]:
                        _link_file(src, recent_dir / src.name, link_mode)
                    for src in [frames[i] for i in summary_idx]:
                        _link_file(src, summary_dir / src.name, link_mode)
                    for src in [frames[i] for i in lookahead_idx]:
                        _link_file(src, lookahead_dir / src.name, link_mode)
                    for src in [frames[i] for i in lookahead_summary_idx]:
                        _link_file(src, lookahead_summary_dir / src.name, link_mode)
                    (sample_dir / "action.txt").write_text(actions[t] + "\n", encoding="utf-8")
                    if record["goal_t"]:
                        (sample_dir / "goal.txt").write_text(record["goal_t"] + "\n", encoding="utf-8")
                    if record["instruct_t"]:
                        (sample_dir / "labeling_instruct.txt").write_text(
                            record["instruct_t"] + "\n", encoding="utf-8"
                        )
                    meta = {"sample_id": sample_id, "anchor_t": t, "session_id": session_id}
                    (sample_dir / "meta.json").write_text(
                        json.dumps(meta, ensure_ascii=True, indent=2) + "\n"
                    )

    return index_path, samples_written


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract clips from sessions.zip.")
    parser.add_argument("--zip", type=Path, required=True, help="Path to sessions.zip.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--fps", type=int, default=2, help="FPS for frame extraction.")
    parser.add_argument("--step", type=int, default=2, help="Anchor step in frames.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Keep samples even if any clip window is partial at boundaries.",
    )
    parser.add_argument("--export-clips", action="store_true", help="Export per-sample folders.")
    parser.add_argument("--export-ratio", type=float, default=0.01, help="Fraction of samples to export.")
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="hardlink",
        help="How to materialize per-sample folders.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for export sampling.")
    args = parser.parse_args()

    index_path, samples_written = extract_clips(
        zip_path=args.zip,
        output_dir=args.output,
        fps=args.fps,
        step=args.step,
        allow_partial=args.allow_partial,
        export_clips=args.export_clips,
        export_ratio=args.export_ratio,
        link_mode=args.link_mode,
        seed=args.seed,
    )
    print(f"wrote {samples_written} samples to {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
