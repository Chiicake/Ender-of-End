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


class TestDatasetBuilderPlanner(unittest.TestCase):
    def test_allow_empty_retrieval(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            output_dir = tmp / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            record = {
                "recent_clip": ["frames/000001.jpg"],
                "summary_clip": ["frames/000000.jpg"],
                "goal_t": "<|goal_start|>long/mid<|goal_end|>",
                "short_goal_dsl": [],
                "next_mid_step": "step",
                "attempt": "历史总结/当前思考/下一步规划",
            }
            _write_jsonl(input_dir / "clip_index.jsonl", [record])

            script = Path("scripts/dataset_builder_planner.py")
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--allow-empty-retrieval",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn('"written": 1', result.stdout)
            output_path = output_dir / "planner.jsonl"
            self.assertTrue(output_path.exists())
            output = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertEqual(output["input"]["retrieved_memory"], {})

    def test_require_retrieval(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_dir = tmp / "input"
            output_dir = tmp / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            record = {
                "recent_clip": ["frames/000001.jpg"],
                "summary_clip": ["frames/000000.jpg"],
                "goal_t": "<|goal_start|>long/mid<|goal_end|>",
                "short_goal_dsl": [],
                "next_mid_step": "step",
                "attempt": "历史总结/当前思考/下一步规划",
            }
            _write_jsonl(input_dir / "clip_index.jsonl", [record])

            script = Path("scripts/dataset_builder_planner.py")
            result = subprocess.run(
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
            self.assertEqual(report["written"], 0)
            self.assertEqual(report["missing_retrieval"], 1)


if __name__ == "__main__":
    unittest.main()
