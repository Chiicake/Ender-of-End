# Ender-of-End

施工中。。。

## Overview
This repository currently contains design and technical documentation for a Python-first training framework and runtime loop. The target system covers data collection, labeling, training (Planner/Controller LoRA), evaluation, and runtime orchestration.

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
python3 scripts/vlm_labeler.py --input-dir out/ --backend ollama --base-url http://127.0.0.1:11434/api/chat --model qwen3-vl:4b
python3 scripts/vlm_labeler.py --input-dir out/ --backend ollama --base-url http://127.0.0.1:11434/api/chat --model qwen3-vl:4b --ollama-format json
```
可选环境变量：`VLM_LABELER_BASE_URL`, `VLM_LABELER_MODEL`, `VLM_LABELER_API_KEY`, `VLM_LABELER_ENDPOINT`, `VLM_LABELER_BACKEND`

### Training GUI
桌面 GUI（数据裁切 / VLM 标注 / 训练入口）。
```bash
python3 scripts/training_gui.py
```

## Build, Test, and Run
目前未定义构建或测试命令；新增工具后请在此补充。
