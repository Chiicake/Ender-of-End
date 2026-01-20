# Dataset Builder 技术文档

## 目标与范围
Dataset Builder 负责把采集的 2FPS 视频与 action string 转换为可复现的 Planner/Controller 训练数据，并保证检索字段与 span 对齐规则一致。

## 输入与对齐规则
- **视频**：2FPS，按帧序号读取。
- **动作**：每 500ms 一行 action string，格式：
  `<|action_start|>dx dy dz ; group1 ; ... ; group15<|action_end|>`
- **对齐**：**不做时间戳对齐**，使用序号一致性：
  - 第 i 帧对应第 i 行 action。
  - 缺帧/缺行动作直接丢弃该段或整段 episode。
- **任务图（手工枚举）**：先维护小规模 `mid_step` 枚举表（10–30 个），Labeler 只能从枚举中选择。
  - Dataset Builder 读取 `mid_step` 标注文件（推荐 CSV/JSONL），按帧区间标注 `mid_step_id/text`。

## 片段抽取（clip sampling）
- **步长**：滑窗步长 2 帧（= 1 秒）。
- **recent_clip**：8 帧（4 秒），取 `[t-7..t]`。
- **summary_clip（Planner 专用）**：60 秒历史，每 2 秒采 1 帧（约 30 帧）。
- **lookahead_clip（标注用）**：8 帧（4 秒），取 `[t..t+7]`。
- **lookahead_summary_clip（标注用）**：`[t..t+60s]` 每 2 秒采 1 帧（约 30 帧）。
- **边界**：只在同一 `mid_step` 内采样，不跨界。

建议配置：
- `batch_size=8`（视吞吐调整）
- `max_retries=3`，指数退避（1s/2s/4s）
- 缓存键：`clip_hash + mid_step_id + schema_version`

## 检索字段构造（Planner）
- **规则检索优先**（不做 embedding）：
  - `topK_related`：同 `mid_step_id` 的最近失败/尝试优先，`K=5`。
  - 不足再补 `state_summary` 语义相似项。
- 记录 `retrieval_policy_version` 与 `retrieval_snapshot`（query/filters/topK 项列表）。

## Plan Span 对齐（Controller）
- span 起点：Dataset Builder 生成 `plan_id` 的时刻 `t0`。
- span 截断：`done_evidence` 稳定触发 / need_plan / 强干扰 / horizon 到期。
- **稳定阈值**：连续 3 帧 + L1/L2 置信度 ≥ 0.8。

## 输出格式（JSONL）
建议目录：
```
datasets/
  planner/
    train.jsonl
    val.jsonl
  controller/
    train.jsonl
    val.jsonl
```

Planner 样本（示例字段）：
```json
{
  "recent_clip": ["frame_000123.jpg", "..."],
  "summary_clip": ["frame_000003.jpg", "..."],
  "mid_step_id": "MID_XX",
  "mid_step_text": "...",
  "retrieved_memory": {"recent_window_events": [...], "topK_related": [...]},
  "retrieval_policy_version": "retrieval_v1.0",
  "retrieval_snapshot": {...},
  "label": {"plan_id": "...", "short_goal_dsl": [...], "terminate_on": "...", "schema_version": "plan_v1.0"}
}
```

Controller 样本（示例字段）：
```json
{
  "image_t": "frame_000123.jpg",
  "history": ["frame_000119.jpg", "action_000119.txt", "..."],
  "short_goal_dsl": [...],
  "plan_id": "...",
  "action_t": "<|action_start|>...<|action_end|>"
}
```

## 质量门槛与统计
- 解析合法率（action）与 schema 校验失败率需记录并告警。
- 记录 span 截断原因分布、检索命中率、Labeler 不确定性占比。
