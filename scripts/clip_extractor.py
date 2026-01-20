#!/usr/bin/env python3
"""
Clip extractor for 2FPS video + action string.

Outputs:
  - JSONL index for training
  - Optional per-sample folders for QA
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class MidStepSpan:
    start_frame: int
    end_frame: int
    mid_step_id: str
    mid_step_text: str


def _read_actions(path: Path) -> list[str]:
    actions: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            actions.append(line)
    return actions


def _list_frames(frames_dir: Path) -> list[Path]:
    frames = sorted(frames_dir.glob("*"))
    frames = [p for p in frames if p.is_file()]
    return frames


def _ensure_frames(video_path: Optional[Path], frames_dir: Path, fps: int) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = _list_frames(frames_dir)
    if existing:
        return existing
    if video_path is None:
        raise ValueError("frames_dir is empty and no video path was provided.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; provide --frames-dir with pre-extracted frames.")
    output_pattern = frames_dir / "%06d.jpg"
    cmd = [
        ffmpeg,
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        str(output_pattern),
    ]
    subprocess.run(cmd, check=True)
    return _list_frames(frames_dir)


def _read_mid_steps(path: Path) -> list[MidStepSpan]:
    spans: list[MidStepSpan] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                spans.append(
                    MidStepSpan(
                        start_frame=int(rec["start_frame"]),
                        end_frame=int(rec["end_frame"]),
                        mid_step_id=str(rec["mid_step_id"]),
                        mid_step_text=str(rec["mid_step_text"]),
                    )
                )
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for rec in reader:
                spans.append(
                    MidStepSpan(
                        start_frame=int(rec["start_frame"]),
                        end_frame=int(rec["end_frame"]),
                        mid_step_id=str(rec["mid_step_id"]),
                        mid_step_text=str(rec["mid_step_text"]),
                    )
                )
    else:
        raise ValueError("mid_steps file must be .jsonl or .csv")
    return spans


def _build_mid_step_lookup(
    spans: list[MidStepSpan], frame_count: int
) -> list[Optional[MidStepSpan]]:
    lookup: list[Optional[MidStepSpan]] = [None] * frame_count
    for span in spans:
        if span.start_frame < 0 or span.end_frame < span.start_frame:
            raise ValueError(f"invalid mid_step span: {span}")
        for idx in range(span.start_frame, min(span.end_frame + 1, frame_count)):
            if lookup[idx] is not None:
                raise ValueError(f"overlapping mid_step at frame {idx}")
            lookup[idx] = span
    return lookup


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


def _indices_recent(t: int) -> list[int]:
    return list(range(t - 7, t + 1))


def _indices_lookahead(t: int) -> list[int]:
    return list(range(t, t + 8))


def _indices_summary(t: int) -> list[int]:
    # 60s window at 2FPS -> 120 frames; sample every 2s -> step of 4 frames
    return list(range(t - 120, t + 1, 4))


def _indices_lookahead_summary(t: int) -> list[int]:
    return list(range(t, t + 121, 4))


def _all_in_range(indices: Iterable[int], start: int, end: int) -> bool:
    return all(start <= idx <= end for idx in indices)


def _relative_paths(paths: list[Path], root: Path) -> list[str]:
    return [str(p.relative_to(root)) for p in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract clips for labeling/training.")
    parser.add_argument("--video", type=Path, help="Path to video file (optional if frames_dir provided).")
    parser.add_argument("--frames-dir", type=Path, help="Directory containing pre-extracted frames.")
    parser.add_argument("--actions", type=Path, required=True, help="Action string file (1 line per frame).")
    parser.add_argument("--mid-steps", type=Path, help="Optional mid_step spans (.jsonl or .csv).")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--episode-id", type=str, help="Episode id for sample ids.")
    parser.add_argument("--fps", type=int, default=2, help="FPS for frame extraction.")
    parser.add_argument("--step", type=int, default=2, help="Anchor step in frames.")
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

    if args.frames_dir is None and args.video is None:
        raise SystemExit("Either --video or --frames-dir must be provided.")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.frames_dir or (output_dir / "frames")

    frames = _ensure_frames(args.video, frames_dir, args.fps)
    actions = _read_actions(args.actions)
    if not frames or not actions:
        raise SystemExit("No frames or actions found.")

    count = min(len(frames), len(actions))
    if len(frames) != len(actions):
        print(
            f"warning: frames({len(frames)}) != actions({len(actions)}); using first {count}",
            file=sys.stderr,
        )
    frames = frames[:count]
    actions = actions[:count]

    mid_step_lookup: Optional[list[Optional[MidStepSpan]]] = None
    if args.mid_steps:
        spans = _read_mid_steps(args.mid_steps)
        mid_step_lookup = _build_mid_step_lookup(spans, count)

    episode_id = args.episode_id or (args.video.stem if args.video else "episode")
    index_path = output_dir / "clip_index.jsonl"
    rng = random.Random(args.seed)

    samples_written = 0
    with index_path.open("w", encoding="utf-8") as index_handle:
        min_t = 120
        max_t = count - 1 - 120
        if min_t >= max_t:
            print("warning: not enough frames for full windows; no samples generated.", file=sys.stderr)
            return 0

        for t in range(min_t, max_t + 1, args.step):
            recent_idx = _indices_recent(t)
            summary_idx = _indices_summary(t)
            lookahead_idx = _indices_lookahead(t)
            lookahead_summary_idx = _indices_lookahead_summary(t)

            if recent_idx[0] < 0 or lookahead_summary_idx[-1] >= count:
                continue

            if mid_step_lookup is not None:
                span = mid_step_lookup[t]
                if span is None:
                    continue
                if not (
                    _all_in_range(recent_idx, span.start_frame, span.end_frame)
                    and _all_in_range(summary_idx, span.start_frame, span.end_frame)
                    and _all_in_range(lookahead_idx, span.start_frame, span.end_frame)
                    and _all_in_range(lookahead_summary_idx, span.start_frame, span.end_frame)
                ):
                    continue
            else:
                span = None

            sample_id = f"{episode_id}_t{t:06d}"
            record = {
                "sample_id": sample_id,
                "episode_id": episode_id,
                "anchor_t": t,
                "frames_root": str(frames_dir),
                "recent_clip": _relative_paths([frames[i] for i in recent_idx], frames_dir),
                "summary_clip": _relative_paths([frames[i] for i in summary_idx], frames_dir),
                "lookahead_clip": _relative_paths([frames[i] for i in lookahead_idx], frames_dir),
                "lookahead_summary_clip": _relative_paths(
                    [frames[i] for i in lookahead_summary_idx], frames_dir
                ),
                "action_t": actions[t],
            }
            if span is not None:
                record["mid_step_id"] = span.mid_step_id
                record["mid_step_text"] = span.mid_step_text

            index_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            samples_written += 1

            if args.export_clips and rng.random() <= args.export_ratio:
                sample_dir = output_dir / "clips" / sample_id
                recent_dir = sample_dir / "recent"
                summary_dir = sample_dir / "summary"
                lookahead_dir = sample_dir / "lookahead"
                lookahead_summary_dir = sample_dir / "lookahead_summary"
                for src in [frames[i] for i in recent_idx]:
                    _link_file(src, recent_dir / src.name, args.link_mode)
                for src in [frames[i] for i in summary_idx]:
                    _link_file(src, summary_dir / src.name, args.link_mode)
                for src in [frames[i] for i in lookahead_idx]:
                    _link_file(src, lookahead_dir / src.name, args.link_mode)
                for src in [frames[i] for i in lookahead_summary_idx]:
                    _link_file(src, lookahead_summary_dir / src.name, args.link_mode)
                (sample_dir / "action.txt").write_text(actions[t] + "\n", encoding="utf-8")
                meta = {
                    "sample_id": sample_id,
                    "anchor_t": t,
                    "mid_step_id": record.get("mid_step_id"),
                    "mid_step_text": record.get("mid_step_text"),
                }
                (sample_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n")

    print(f"wrote {samples_written} samples to {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
