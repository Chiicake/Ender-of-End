"""Core VLM labeler logic.

`run_labeler` reads clip_index.jsonl, encodes frames to base64, calls a remote
VLM service via LangChain, validates the response, and writes labels back
to the index.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


@dataclass
class LabelerConfig:
    input_dir: Path
    index_path: Path | None = None
    output_index_path: Path | None = None
    endpoint: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout_sec: float = 120.0
    temperature: float = 0.0
    max_concurrency: int = 4
    batch_size: int = 8
    max_retries: int = 3
    retry_backoff_sec: float = 1.0
    overwrite: bool = False
    dry_run: bool = False
    validate: bool = True
    include_enums: bool = True
    log_requests: bool = True
    log_responses: bool = True
    log_full_payload: bool = True
    system_prompt: str | None = None
    user_prompt_template: str | None = None
    prompts_dir: Path | None = None
    system_prompt_path: Path | None = None
    user_prompt_path: Path | None = None
    mime_type: str = "image/jpeg"
    dsl_ops_path: Path = Path("src/common/enums/dsl_ops.json")
    done_evidence_path: Path = Path("src/common/enums/done_evidence.json")
    fallback_actions_path: Path = Path("src/common/enums/fall_back.json")
    limit: int | None = None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_prompt_text(path: Path | None) -> str | None:
    if not path:
        return None
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _load_dsl_ops(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    data = _load_json(path)
    ops = data.get("ops", [])
    op_names: set[str] = set()
    for entry in ops:
        if isinstance(entry, dict) and "op" in entry:
            op_names.add(str(entry["op"]))
    return ops, op_names


def _load_enum_list(path: Path, keys: Iterable[str]) -> list[str]:
    data = _load_json(path)
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    tmp_path.replace(path)


def _encode_frames(
    rel_paths: list[str],
    input_dir: Path,
    mime_type: str,
) -> list[dict[str, str]] | None:
    encoded: list[dict[str, str]] = []
    for rel_path in rel_paths:
        path = input_dir / rel_path
        if not path.exists():
            print(f"[error] missing frame: {path}")
            return None
        data = path.read_bytes()
        encoded.append(
            {
                "mime": mime_type,
                "data": base64.b64encode(data).decode("ascii"),
            }
        )
    return encoded


def _build_item(record: dict[str, Any], config: LabelerConfig) -> dict[str, Any] | None:
    try:
        recent = record["recent_clip"]
        summary = record["summary_clip"]
        lookahead = record["lookahead_clip"]
        lookahead_summary = record["lookahead_summary_clip"]
    except KeyError as exc:
        print(f"[error] missing clip field {exc} in sample {record.get('sample_id', '')}")
        return None
    if not all(isinstance(value, list) for value in [recent, summary, lookahead, lookahead_summary]):
        print(f"[error] clip fields must be lists in sample {record.get('sample_id', '')}")
        return None

    recent_payload = _encode_frames(recent, config.input_dir, config.mime_type)
    summary_payload = _encode_frames(summary, config.input_dir, config.mime_type)
    lookahead_payload = _encode_frames(lookahead, config.input_dir, config.mime_type)
    lookahead_summary_payload = _encode_frames(lookahead_summary, config.input_dir, config.mime_type)
    if any(payload is None for payload in [recent_payload, summary_payload, lookahead_payload, lookahead_summary_payload]):
        return None

    goal = record.get("goal_t") or record.get("goal") or ""
    instruct = record.get("instruct_t") or record.get("labeling_instruct") or ""

    return {
        "sample_id": record.get("sample_id", ""),
        "recent_clip": recent_payload,
        "summary_clip": summary_payload,
        "lookahead_clip": lookahead_payload,
        "lookahead_summary_clip": lookahead_summary_payload,
        "goal": goal,
        "labeling_instruct": instruct,
    }


def _normalize_output(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None

    normalized = dict(raw)
    if isinstance(normalized.get("short_goal_dsl"), dict):
        normalized["short_goal_dsl"] = [normalized["short_goal_dsl"]]
    if isinstance(normalized.get("fallback_if_failed"), str):
        normalized["fallback_if_failed"] = [normalized["fallback_if_failed"]]
    if isinstance(normalized.get("done_evidence"), str):
        normalized["done_evidence"] = [normalized["done_evidence"]]
    if "horizon_steps" in normalized:
        try:
            normalized["horizon_steps"] = int(normalized["horizon_steps"])
        except (TypeError, ValueError):
            return None
    if "uncertainty" in normalized and isinstance(normalized["uncertainty"], str):
        normalized["uncertainty"] = normalized["uncertainty"].lower()
    return normalized


def _validate_output(
    output: dict[str, Any],
    dsl_ops: set[str],
    done_evidence_enum: list[str],
    fallback_enum: list[str],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    required_fields = [
        "goal",
        "next_mid_step",
        "short_goal_dsl",
        "horizon_steps",
        "done_evidence",
        "fallback_if_failed",
        "uncertainty",
        "attempt",
    ]
    for field in required_fields:
        if field not in output:
            errors.append(f"missing field: {field}")

    goal = output.get("goal")
    if not isinstance(goal, str):
        errors.append("goal must be string")
    elif "<|goal_start|>" not in goal or "<|goal_end|>" not in goal:
        errors.append("goal format invalid")

    short_goal_dsl = output.get("short_goal_dsl")
    if not isinstance(short_goal_dsl, list):
        errors.append("short_goal_dsl must be list")
    else:
        for entry in short_goal_dsl:
            if not isinstance(entry, dict) or "op" not in entry:
                errors.append("short_goal_dsl entry missing op")
                continue
            op = str(entry["op"])
            if dsl_ops and op not in dsl_ops:
                errors.append(f"unknown dsl op: {op}")

    done_evidence = output.get("done_evidence")
    if not isinstance(done_evidence, list):
        errors.append("done_evidence must be list")
    else:
        if any(item for item in done_evidence_enum):
            for item in done_evidence:
                if item not in done_evidence_enum:
                    errors.append(f"unknown done_evidence: {item}")

    fallback = output.get("fallback_if_failed")
    if not isinstance(fallback, list):
        errors.append("fallback_if_failed must be list")
    else:
        if any(item for item in fallback_enum):
            for item in fallback:
                if item not in fallback_enum:
                    errors.append(f"unknown fallback_if_failed: {item}")

    uncertainty = output.get("uncertainty")
    if uncertainty not in {"low", "mid", "high"}:
        errors.append("uncertainty must be low/mid/high")

    horizon_steps = output.get("horizon_steps")
    if not isinstance(horizon_steps, int):
        errors.append("horizon_steps must be int")

    return (len(errors) == 0, errors)


def _default_system_prompt() -> str:
    return "\n".join(
        [
            "You are a game automation data labeling assistant.",
            "Return JSON only; do not include explanations.",
            "short_goal_dsl must be executable within 1-10 seconds.",
            "goal may be wrong or empty; correct it using visual evidence.",
            "lookahead clips are only for judging next_mid_step, horizon_steps, and done_evidence.",
            "Do not use lookahead clips to invent short_goal_dsl.",
            "Output fields: goal, next_mid_step, short_goal_dsl, horizon_steps,",
            "done_evidence, fallback_if_failed, uncertainty, attempt.",
            "uncertainty must be one of: low, mid, high.",
        ]
    )


def _default_prompts_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


def _normalize_base_url(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        trimmed = trimmed[: -len("/chat/completions")]
    if "/v1/" in trimmed:
        prefix = trimmed.split("/v1/")[0]
        return f"{prefix}/v1"
    return trimmed


def _build_user_text(
    item: dict[str, Any],
    dsl_ops_enum: list[dict[str, Any]],
    done_evidence_enum: list[str],
    fallback_enum: list[str],
    include_enums: bool,
) -> str:
    lines = [
        "Input fields:",
        f"sample_id: {item.get('sample_id', '')}",
        f"goal: {item.get('goal', '')}",
        f"labeling_instruct: {item.get('labeling_instruct', '')}",
        f"recent_clip: {len(item.get('recent_clip', []))} frames attached",
        f"summary_clip: {len(item.get('summary_clip', []))} frames attached",
        f"lookahead_clip: {len(item.get('lookahead_clip', []))} frames attached",
        f"lookahead_summary_clip: {len(item.get('lookahead_summary_clip', []))} frames attached",
        "Field definitions:",
        "goal: <|goal_start|>long_goal/mid_goal<|goal_end|> format.",
        "next_mid_step: if current step is complete, output the next step; otherwise keep current.",
        "short_goal_dsl: list of DSL ops with args; ops must come from dsl_ops_enum.",
        "horizon_steps: number of frames at 2FPS (1 step = 0.5s).",
        "done_evidence: list from done_evidence_enum.",
        "fallback_if_failed: list from fallback_actions_enum.",
        "uncertainty: low/mid/high.",
        "attempt: summary of past, current reasoning, and next plan.",
    ]
    if include_enums:
        lines.extend(
            [
                "Enums:",
                f"dsl_ops_enum: {json.dumps(dsl_ops_enum, ensure_ascii=False)}",
                f"done_evidence_enum: {json.dumps(done_evidence_enum, ensure_ascii=False)}",
                f"fallback_actions_enum: {json.dumps(fallback_enum, ensure_ascii=False)}",
            ]
        )
    else:
        lines.append("Enums: omitted")
    return "\n".join(lines)


def _render_user_prompt(
    template: str,
    item: dict[str, Any],
    dsl_ops_enum: list[dict[str, Any]],
    done_evidence_enum: list[str],
    fallback_enum: list[str],
    include_enums: bool,
) -> str:
    return template.format_map(
        {
            "sample_id": item.get("sample_id", ""),
            "goal": item.get("goal", ""),
            "labeling_instruct": item.get("labeling_instruct", ""),
            "recent_count": len(item.get("recent_clip", [])),
            "summary_count": len(item.get("summary_clip", [])),
            "lookahead_count": len(item.get("lookahead_clip", [])),
            "lookahead_summary_count": len(item.get("lookahead_summary_clip", [])),
            "dsl_ops_enum": json.dumps(dsl_ops_enum, ensure_ascii=False)
            if include_enums
            else "omitted",
            "done_evidence_enum": json.dumps(done_evidence_enum, ensure_ascii=False)
            if include_enums
            else "omitted",
            "fallback_actions_enum": json.dumps(fallback_enum, ensure_ascii=False)
            if include_enums
            else "omitted",
        }
    )


def _append_clip(content: list[dict[str, Any]], label: str, frames: list[dict[str, str]]) -> None:
    content.append({"type": "text", "text": f"{label} ({len(frames)} frames):"})
    for idx, frame in enumerate(frames, start=1):
        content.append({"type": "text", "text": f"{label} frame {idx}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{frame.get('mime', 'image/jpeg')};base64,{frame.get('data', '')}"
                },
            }
        )


def _build_messages(
    item: dict[str, Any],
    config: LabelerConfig,
    dsl_ops_enum: list[dict[str, Any]],
    done_evidence_enum: list[str],
    fallback_enum: list[str],
) -> list[Any]:
    system_prompt = config.system_prompt or _default_system_prompt()
    if config.user_prompt_template:
        user_text = _render_user_prompt(
            config.user_prompt_template,
            item,
            dsl_ops_enum,
            done_evidence_enum,
            fallback_enum,
            config.include_enums,
        )
    else:
        user_text = _build_user_text(
            item, dsl_ops_enum, done_evidence_enum, fallback_enum, config.include_enums
        )
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    _append_clip(content, "recent_clip", item.get("recent_clip", []))
    _append_clip(content, "summary_clip", item.get("summary_clip", []))
    _append_clip(content, "lookahead_clip", item.get("lookahead_clip", []))
    _append_clip(content, "lookahead_summary_clip", item.get("lookahead_summary_clip", []))
    return [SystemMessage(content=system_prompt), HumanMessage(content=content)]


def _request_with_retries(
    llm: ChatOpenAI,
    messages_batch: list[list[Any]],
    config: LabelerConfig,
) -> tuple[list[Any], float]:
    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        start = time.monotonic()
        try:
            result = llm.batch(messages_batch, {"max_concurrency": config.max_concurrency})
            duration = time.monotonic() - start
            return result, duration
        except Exception as exc:
            duration = time.monotonic() - start
            last_error = exc
            print(f"[retry] attempt {attempt}/{config.max_retries} failed after {duration:.2f}s: {exc}")
            if attempt < config.max_retries:
                time.sleep(config.retry_backoff_sec * (2 ** (attempt - 1)))
    raise RuntimeError(f"request failed after {config.max_retries} attempts: {last_error}")


def _extract_items(response: list[Any]) -> list[Any]:
    return response


def _should_label(record: dict[str, Any], overwrite: bool) -> bool:
    if overwrite:
        return True
    return "short_goal_dsl" not in record and "next_mid_step" not in record


def _format_payload(items: list[dict[str, Any]], full: bool) -> str:
    if full:
        return json.dumps(items, ensure_ascii=False)
    trimmed_items: list[dict[str, Any]] = []
    for item in items:
        entry = dict(item)
        for key in [
            "recent_clip",
            "summary_clip",
            "lookahead_clip",
            "lookahead_summary_clip",
        ]:
            if key in entry and isinstance(entry[key], list):
                entry[key] = [f"<base64:{len(frame.get('data', ''))}>" for frame in entry[key]]
        trimmed_items.append(entry)
    return json.dumps(trimmed_items, ensure_ascii=False)


def run_labeler(config: LabelerConfig) -> int:
    index_path = config.index_path or config.input_dir / "clip_index.jsonl"
    output_path = config.output_index_path or index_path
    records = _read_jsonl(index_path)

    dsl_ops, dsl_op_names = _load_dsl_ops(config.dsl_ops_path)
    done_evidence_enum = _load_enum_list(config.done_evidence_path, ("done_evidence", "evidence"))
    fallback_enum = _load_enum_list(
        config.fallback_actions_path, ("fallback_actions", "fallbacks", "done_evidence")
    )

    indices = [idx for idx, record in enumerate(records) if _should_label(record, config.overwrite)]
    if config.limit is not None:
        indices = indices[: config.limit]

    total = len(indices)
    print(f"[start] samples={total} index={index_path}")
    if config.dry_run:
        print("[info] dry_run enabled; no updates will be written")
    base_url = config.base_url or _normalize_base_url(config.endpoint)
    if not config.dry_run:
        if not base_url:
            raise ValueError("base_url or endpoint is required unless dry_run is set")
        if not config.model:
            raise ValueError("model is required unless dry_run is set")
    prompts_dir = config.prompts_dir or _default_prompts_dir()
    system_prompt_path = config.system_prompt_path or (prompts_dir / "system_prompt.txt")
    user_prompt_path = config.user_prompt_path or (prompts_dir / "user_prompt.txt")
    if config.system_prompt is None:
        config.system_prompt = _load_prompt_text(system_prompt_path)
    if config.user_prompt_template is None:
        config.user_prompt_template = _load_prompt_text(user_prompt_path)

    system_prompt = config.system_prompt or _default_system_prompt()
    if config.log_requests:
        print(f"[prompt] system={system_prompt}")
    llm = None
    if not config.dry_run:
        llm = ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=base_url,
            timeout=config.timeout_sec,
            temperature=config.temperature,
            max_retries=0,
        )

    updated = 0
    for batch_start in range(0, total, config.batch_size):
        batch_indices = indices[batch_start : batch_start + config.batch_size]
        batch_items: list[dict[str, Any]] = []
        batch_record_map: list[int] = []
        messages_batch: list[list[Any]] = []

        for idx in batch_indices:
            item = _build_item(records[idx], config)
            if item is None:
                continue
            batch_items.append(item)
            batch_record_map.append(idx)
            messages_batch.append(
                _build_messages(item, config, dsl_ops, done_evidence_enum, fallback_enum)
            )

        if not batch_items:
            continue

        # if config.log_requests:
        #     print(f"[request] batch={batch_start // config.batch_size} {_format_payload(batch_items, config.log_full_payload)}")

        if config.dry_run:
            continue

        response, duration = _request_with_retries(llm, messages_batch, config)

        if config.log_responses:
            print(
                f"[response] batch={batch_start // config.batch_size} "
                f"{json.dumps([getattr(item, 'content', item) for item in response], ensure_ascii=False)}"
            )

        items = _extract_items(response)
        if len(items) != len(batch_record_map):
            print(
                f"[warn] response items {len(items)} != request items {len(batch_record_map)}"
            )

        for record_idx, raw_item in zip(batch_record_map, items):
            sample_id = records[record_idx].get("sample_id", "")
            content = getattr(raw_item, "content", raw_item)
            normalized = _normalize_output(content)
            if normalized is None:
                print(f"[error] invalid response for {sample_id}; not JSON")
                continue
            if config.validate:
                ok, errors = _validate_output(
                    normalized, dsl_op_names, done_evidence_enum, fallback_enum
                )
                if not ok:
                    print(f"[error] validation failed for {sample_id}: {errors}")
                    continue

            records[record_idx].update(normalized)
            updated += 1
            summary = {
                "sample_id": sample_id,
                "duration_sec": round(duration, 2),
                "uncertainty": normalized.get("uncertainty"),
                "horizon_steps": normalized.get("horizon_steps"),
            }
            print(f"[sample] {json.dumps(summary, ensure_ascii=False)}")

    if not config.dry_run:
        _write_jsonl(output_path, records)
        print(f"[done] updated={updated} output={output_path}")
    return updated
