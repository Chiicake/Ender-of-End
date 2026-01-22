# 训练 GUI 技术文档
src/tools/training_gui

## 目标与范围
训练 GUI 以桌面应用形式提供统一入口，支持 **数据裁切、VLM 标注、Planner 训练、Controller 训练** 四个模块的参数配置、启动/停止与日志查看。当前仅实现“数据裁切”和“VLM 标注”；训练模块先预留界面与配置项（逻辑为空）。

## 设计原则
- **单任务运行**：同一时刻只允许运行一个任务。
- **日志持久化**：所有 stdout/stderr 写入 `runs/<run_id>/logs/*.log`，GUI 实时展示。
- **可复现**：每次运行生成 `run_id` 与完整参数快照。

## 模块与职责
### 1) 数据裁切（Clip Extractor）
- 调用：`scripts/clip_extractor.py`
- 输入：`sessions.zip`
- 输出：`clip_index.jsonl` + `frames/`

### 2) VLM 标注（VLM Labeler）
- 调用：`scripts/vlm_labeler.py`
- 输入：`clip_index.jsonl` + `frames/`
- 输出：写回 `clip_index.jsonl`

### 3) Planner 训练（预留）
- 只展示配置项与按钮；不执行训练逻辑。

### 4) Controller 训练（预留）
- 只展示配置项与按钮；不执行训练逻辑。

## GUI 交互
- **配置区**：按模块分区展示参数（路径、模型、batch、重试等）。
- **执行区**：`开始 / 停止` 按钮；显示运行状态与进度摘要。
- **日志区**：实时显示 stdout/stderr，支持清空与保存。
- **历史区**：显示历史 `run_id` 与配置快照位置。

## 运行流程
1) 用户在 GUI 中填写模块参数。
2) 点击 “开始”，生成 `run_id` 与配置快照。
3) GUI 启动子进程并持续读取 stdout/stderr。
4) 日志实时显示并写入文件。
5) 点击 “停止” 发送终止信号，等待进程退出。

## 配置与目录
```
runs/
  <run_id>/
    config.json
    logs/
      stdout.log
      stderr.log
```
- `config.json` 保存当次运行的所有参数。
- 所有任务均从 GUI 调起，不直接修改业务代码。

## 日志策略
- stdout/stderr **双通道**写入并实时展示。
- GUI 只读日志文件（支持 tail），避免阻塞主线程。

## 依赖建议
- 桌面框架：`PySide6`（或 `PyQt6`）
- 进程管理：`subprocess.Popen` + 非阻塞读取
- 日志读取：后台线程或定时器轮询

## 待实现
- Planner 训练：空实现（仅 GUI 配置与按钮）。
- Controller 训练：空实现（仅 GUI 配置与按钮）。
