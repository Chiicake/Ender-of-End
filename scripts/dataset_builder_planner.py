#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset builder (planner).")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-empty-retrieval", action="store_true")
    args = parser.parse_args()

    index_path = args.input_dir / "clip_index.jsonl"
    if not index_path.exists():
        raise SystemExit(f"clip_index.jsonl not found: {index_path}")

    records = _read_jsonl(index_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "planner.jsonl"
    report = {
        "input": str(index_path),
        "output": str(out_path),
        "total": len(records),
        "written": 0,
        "skipped": 0,
        "missing_retrieval": 0,
    }

    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            required = [
                "recent_clip",
                "summary_clip",
                "goal_t",
                "short_goal_dsl",
                "next_mid_step",
                "attempt",
            ]
            if not all(key in record for key in required):
                report["skipped"] += 1
                continue

            retrieval = record.get("retrieved_memory")
            if retrieval is None:
                report["missing_retrieval"] += 1
                if args.allow_empty_retrieval:
                    retrieval = {}
                else:
                    report["skipped"] += 1
                    continue

            sample = {
                "input": {
                    "recent_clip": record.get("recent_clip"),
                    "summary_clip": record.get("summary_clip"),
                    "goal": record.get("goal_t"),
                    "labeling_instruct": record.get("instruct_t", ""),
                    "retrieved_memory": retrieval,
                },
                "target": {
                    "goal": record.get("goal"),
                    "next_mid_step": record.get("next_mid_step"),
                    "short_goal_dsl": record.get("short_goal_dsl"),
                    "horizon_steps": record.get("horizon_steps"),
                    "done_evidence": record.get("done_evidence"),
                    "fallback_if_failed": record.get("fallback_if_failed"),
                    "uncertainty": record.get("uncertainty"),
                    "attempt": record.get("attempt"),
                },
            }
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
            report["written"] += 1

    report_path = output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
