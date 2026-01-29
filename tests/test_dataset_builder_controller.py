import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


class TestDatasetBuilderController(unittest.TestCase):
    def test_requires_action(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            output_dir = tmp / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            records = [
                {
                    "recent_clip": ["frames/000001.jpg"],
                    "action_t": "",
                    "short_goal_dsl": [],
                },
                {
                    "recent_clip": ["frames/000002.jpg"],
                    "action_t": "<|action_start|>0 0 0 ; g1 ; g2 ; g3 ; g4 ; g5 ; g6<|action_end|>",
                    "short_goal_dsl": [],
                    "plan_id": "plan_1",
                },
            ]
            _write_jsonl(input_dir / "clip_index.jsonl", records)

            script = Path("scripts/dataset_builder_controller.py")
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads((output_dir / "build_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["written"], 1)
            self.assertEqual(report["missing_action"], 1)
            output = json.loads((output_dir / "controller.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(output["target"]["action_t"], records[1]["action_t"])


if __name__ == "__main__":
    unittest.main()
