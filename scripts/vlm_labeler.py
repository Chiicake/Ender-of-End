#!/usr/bin/env python3
"""CLI entrypoint for VLM labeler.

Usage examples:
  python3 scripts/vlm_labeler.py --input-dir out/ --dry-run --trim-payload
  python3 scripts/vlm_labeler.py --input-dir out/ --base-url http://127.0.0.1:8000/v1 --model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from labeler import LabelerConfig, run_labeler  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VLM labeler over clip_index.jsonl.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing clip_index.jsonl.")
    parser.add_argument("--index-file", type=Path, help="Override clip_index.jsonl path.")
    parser.add_argument("--output-index", type=Path, help="Write updated index to this path.")
    parser.add_argument("--base-url", type=str, default=os.getenv("VLM_LABELER_BASE_URL"))
    parser.add_argument("--model", type=str, default=os.getenv("VLM_LABELER_MODEL"))
    parser.add_argument(
        "--endpoint",
        type=str,
        default=os.getenv("VLM_LABELER_ENDPOINT"),
        help="HTTP endpoint for the labeler backend.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=os.getenv("VLM_LABELER_BACKEND", "openai"),
        choices=["openai", "ollama"],
        help="Backend type: openai or ollama.",
    )
    parser.add_argument(
        "--ollama-format",
        type=str,
        default=os.getenv("VLM_LABELER_OLLAMA_FORMAT", "json"),
        help="Ollama response format (e.g., json).",
    )
    parser.add_argument(
        "--ollama-num-predict",
        type=int,
        default=None,
        help="Ollama num_predict override.",
    )
    parser.add_argument(
        "--ollama-fallback-no-format",
        action="store_true",
        default=True,
        help="Retry Ollama once without format when output is empty/invalid.",
    )
    parser.add_argument(
        "--no-ollama-fallback-no-format",
        action="store_false",
        dest="ollama_fallback_no_format",
    )
    parser.add_argument(
        "--flush-every-batch",
        action="store_true",
        default=True,
        help="Write clip_index.jsonl after each batch (default: on).",
    )
    parser.add_argument(
        "--no-flush-every-batch",
        action="store_false",
        dest="flush_every_batch",
        help="Only write clip_index.jsonl at the end.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("VLM_LABELER_API_KEY"),
        help="Bearer token for the labeler backend.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-concurrency", type=int, default=4, help="Max concurrent requests.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per request.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count for failed requests.")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="Retry backoff base in seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labels.")
    parser.add_argument("--dry-run", action="store_true", help="Log payloads without sending requests.")
    parser.add_argument("--skip-validation", action="store_true", help="Skip schema/enum validation.")
    parser.add_argument("--no-enums", action="store_true", help="Do not include enums in payload.")
    parser.add_argument("--log-requests", action="store_true", default=True, help="Log requests.")
    parser.add_argument("--no-log-requests", action="store_false", dest="log_requests")
    parser.add_argument("--log-responses", action="store_true", default=True, help="Log responses.")
    parser.add_argument("--no-log-responses", action="store_false", dest="log_responses")
    parser.add_argument("--log-full-payload", action="store_true", default=True, help="Log full payloads.")
    parser.add_argument("--trim-payload", action="store_false", dest="log_full_payload")
    parser.add_argument("--limit", type=int, help="Max samples to process.")
    parser.add_argument("--dsl-ops", type=Path, default=Path("src/common/enums/dsl_ops.json"))
    parser.add_argument("--done-evidence", type=Path, default=Path("src/common/enums/done_evidence.json"))
    parser.add_argument("--fallback-actions", type=Path, default=Path("src/common/enums/fall_back.json"))
    parser.add_argument("--system-prompt", type=str, help="Override system prompt text.")
    parser.add_argument("--system-prompt-file", type=Path, help="Load system prompt from file.")
    parser.add_argument("--user-prompt-file", type=Path, help="Load user prompt template from file.")
    parser.add_argument("--prompts-dir", type=Path, help="Directory containing system_prompt.txt and user_prompt.txt.")
    args = parser.parse_args()

    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
    user_prompt_template = None
    if args.user_prompt_file:
        user_prompt_template = args.user_prompt_file.read_text(encoding="utf-8")

    ollama_format = args.ollama_format
    if ollama_format is not None and ollama_format.strip().lower() in {"", "none", "null"}:
        ollama_format = None

    config = LabelerConfig(
        input_dir=args.input_dir,
        index_path=args.index_file,
        output_index_path=args.output_index,
        endpoint=args.endpoint,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout_sec=args.timeout,
        temperature=args.temperature,
        max_concurrency=args.max_concurrency,
        backend=args.backend,
        ollama_format=ollama_format,
        ollama_num_predict=args.ollama_num_predict,
        ollama_fallback_no_format=args.ollama_fallback_no_format,
        flush_every_batch=args.flush_every_batch,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        retry_backoff_sec=args.retry_backoff,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        validate=not args.skip_validation,
        include_enums=not args.no_enums,
        log_requests=args.log_requests,
        log_responses=args.log_responses,
        log_full_payload=args.log_full_payload,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        prompts_dir=args.prompts_dir,
        dsl_ops_path=args.dsl_ops,
        done_evidence_path=args.done_evidence,
        fallback_actions_path=args.fallback_actions,
        limit=args.limit,
    )

    run_labeler(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
