# Dataset Builder 技术文档
src/game_agent_train/dataset_builder

## 目标与范围
Dataset Builder 负责把采集数据与标注结果转换为可训练数据集，并保证 Planner/Controller 的输入输出对齐、plan span 正确、检索字段完整。**Planner 与 Controller 使用不同入口**，因为 Planner 训练数据可能不包含 action string。

## 入口（分别构建）
- **Planner Builder 入口**（仅 Planner 数据）：`scripts/dataset_builder_planner.py`
- **Controller Builder 入口**（Span-aligned BC）：`scripts/dataset_builder_controller.py`
> 入口脚本尚未实现，仅定义规范。

## 输入
### 通用
- `clip_index.jsonl`（包含 recent/summary/lookahead/goal/instruct）
- VLM Labeler 输出字段（写回到 clip_index）：`goal/next_mid_step/short_goal_dsl/horizon_steps/done_evidence/fallback_if_failed/uncertainty/attempt`
- 枚举文件：`src/common/enums/*.json`

### Planner Builder 额外输入
- `retrieved_memory` 所需的运行日志（attempt_log / event_log / state_summary / transitions）

### Controller Builder 额外输入
- 对齐动作：`action_t`（必须存在）
- plan span 对齐所需的事件/证据（done_evidence、need_plan、强干扰事件）

## 输出
### Planner 训练集（Retrieval-aware）
- input：`recent_clip + summary_clip + long_goal + mid_goal + retrieved_memory`
- target：Labeler 输出 JSON（含 `short_goal_dsl/next_mid_step/attempt/...`）
- 需写入：`retrieval_policy_version` 与 `retrieval_snapshot`

### Controller 训练集（Span-aligned）
- input：`image_t + history + short_goal_dsl + plan_id`
- target：`action_t`
- 每条样本必须绑定 `plan_id` 与 span 范围

## Build 前完整度检查
### Planner 必要字段
- `recent_clip` / `summary_clip`
- `goal_t`（含 long_goal/mid_goal）
- Labeler 输出 JSON 完整
- `retrieved_memory`（可缺失，缺失时标记为空 `{}`）

### Controller 必要字段
- `image_t` 与 `action_t` 对齐长度一致
- `short_goal_dsl` 与 `plan_id` 已注入
- `done_evidence` / `horizon_steps` 可用于 span 截断

缺失则 **跳过样本并记录原因**，可导出 `build_report.json`.

## 数据构建流程
1) **清洗与对齐**：丢弃缺帧/过场/切后台/加载过久片段；统计分布
2) **读取 Labeler 输出**：过滤 `uncertainty=high` 或进入复核队列
3) **Planner Builder**：
   - 构造 `retrieved_memory`
   - 输出 Planner 训练样本（JSONL）
4) **Controller Builder**：
   - 根据 `done_evidence/need_plan/干扰事件/horizon_steps` 构造 plan span
   - Span 内逐帧生成 conditional BC 样本

## Span 对齐规则（Controller）
- 起点：`plan_id` 生成时间点 `t0`
- 截断优先级：`done_evidence` > `need_plan` > 强干扰事件 > `horizon_steps` 到期
- 证据需满足稳定阈值（连续 N 帧一致）才结束 span

## 输出目录建议
```
dataset/build/
  planner/train.jsonl
  controller/train.jsonl
  build_report.json
```

## 日志与统计
- 输出样本数量、丢弃原因、uncertainty 分布、span 长度分布
- 记录 `retrieval_policy_version` 与检索快照
