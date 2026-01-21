# Ender-of-End

## Overview
This repository currently contains design and technical documentation for a Python-first training framework and runtime loop. The target system covers data collection, labeling, training (Planner/Controller LoRA), evaluation, and runtime orchestration.

## Repository Layout
- `docs/`：设计文档与技术文档（中文）
- `scripts/`：可运行工具入口（clip_extractor / vlm_labeler）
- `src/`：核心库代码（labeler 逻辑与枚举）
- `dataset/`：示例数据（如 sessions.zip）

## Install Dependencies
```bash
conda env create -f environment.yml
```

## Tooling and Example Commands
### Clip Extractor
从 `sessions.zip` 抽取帧、切片并写入 `clip_index.jsonl`。
```bash
python3 scripts/clip_extractor.py --zip dataset/example/sessions.zip --output out/
python3 scripts/clip_extractor.py --zip dataset/example/sessions.zip --output out/ --allow-partial
```

### VLM Labeler
读取 `clip_index.jsonl`，编码视频帧并调用远端 VLM 标注服务，结果写回原索引。
```bash
python3 scripts/vlm_labeler.py --input-dir out/ --dry-run --trim-payload
python3 scripts/vlm_labeler.py --input-dir out/ --base-url http://127.0.0.1:8000/v1 --model gpt-4o-mini
```
可选环境变量：`VLM_LABELER_BASE_URL`, `VLM_LABELER_MODEL`, `VLM_LABELER_API_KEY`, `VLM_LABELER_ENDPOINT`

## Build, Test, and Run
目前未定义构建或测试命令；新增工具后请在此补充。
