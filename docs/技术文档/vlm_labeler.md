# VLM Labeler 技术文档

## 目标与范围
VLM Labeler 负责把短视频片段标注为结构化 `plan_json`（含 DSL short_goal），作为 Planner/Controller 的监督信号。当前仅有视频帧与 action 数据，因此 Labeler 的上下文主要来自 **手工维护的 mid_step 枚举** 与视频内容。

## Labeler 划分
为避免上下文混杂，拆成两个独立 Labeler：
- **Planner Labeler**：输入 `recent_clip + summary_clip + mid_step`，生成适合规划的 `plan_json`。
- **Controller Labeler**：输入 `recent_clip + mid_step`，生成稳定、可执行的 `plan_json`（侧重短期执行）。

## 输入
### 通用字段
- `mid_step_id` / `mid_step_text`：来自手工枚举表（10–30 个），不允许自由发挥。
- `constraints`（可选）：执行限制与禁用动作说明。
- `events/state_summary`（可选）：如果离线预跑得到 L1/L2 结果，可附加。

### 视频片段
- **recent_clip**：8 帧（4 秒），`[t-7..t]`，2FPS。
- **summary_clip（Planner 专用）**：60 秒历史，每 2 秒采 1 帧（约 30 帧）。

## 输出（JSON-only）
### JSON Schema（关键字段）
```json
{
  "mid_step_id": "mid_XX",
  "short_goal_dsl": [{"op": "MOVE_NAV", "args": {"...": "..."}}],
  "horizon_steps": 10,
  "terminate_on": "done_evidence_or_replan|strict_horizon",
  "done_evidence": ["dialog_open"],
  "fallback_if_failed": ["recenter_camera"],
  "uncertainty": "low|mid|high"
}
```

### 字段约束
- `mid_step_id`：必须来自手工枚举表。
- `short_goal_dsl`：仅使用 DSL op 枚举表中的操作。
- `done_evidence`：仅使用证据枚举表中的条目。
- `uncertainty`：用于过滤和抽检，`high` 默认剔除。

### 自动生成字段
- `schema_version`：由 Dataset Builder 注入固定值（如 `plan_v1.0`）。
- `plan_id`：由 Dataset Builder 生成（如 `plan_{episode_id}_{t}`）。

## Prompt 约束（建议）
**System**
- 你是游戏自动化数据标注助手。
- 只输出 JSON，严格符合 schema，不得输出解释文字。
- short_goal 必须在 1–10 秒内可执行。

**User**
- mid_step_id / mid_step_text
- constraints
- recent_clip
- summary_clip（Planner 专用）
- dsl_ops_enum / done_evidence_enum

## HTTP 接口与批处理
建议使用批量请求：
```json
{
  "items": [
    {
      "mid_step_id": "talk_gate_npc",
      "mid_step_text": "...",
      "recent_clip": ["frame_000123.jpg", "..."],
      "summary_clip": ["frame_000003.jpg", "..."],
      "constraints": []
    }
  ]
}
```

### 推荐策略
- `batch_size=8`
- `max_retries=3`，指数退避 1s/2s/4s
- 缓存键：`clip_hash + mid_step_id + schema_version`

## 质量控制
- 强制 schema 校验与枚举校验，不通过直接剔除。
- 记录 `uncertainty` 分布与 `mid_step_id` 覆盖率。
- 对每个 `mid_step_id` 设定固定比例人工抽检样本。

## 输出去向
Labeler 输出作为 Dataset Builder 的监督信号：
- Planner 数据集：`recent_clip + summary_clip + mid_step + retrieved_memory -> plan_json`
- Controller 数据集：`recent_clip + mid_step -> plan_json`（仅用于 span 对齐与短期执行监督）
