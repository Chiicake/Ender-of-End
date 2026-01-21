# VLM Labeler 技术文档
src/game_agent_training/VLM_labeler

## 目标与范围
VLM Labeler 负责把短视频片段标注为结构化 `plan_json`（含 DSL short_goal），作为 Planner/Controller 的监督信号。当前仅有视频帧与 action 数据，因此 Labeler 的上下文主要来自 `goal` / `labeling_instruct` 与视频内容。

所采集到的数据同时供 planner 和 controller 使用。

## 输入
### 通用字段
- `goal`：`<|goal_start|>长期目标/中期目标<|goal_end|>`
- `labeling_instruct`：`<|labeling_instruct_start|>.. <|labeling_instruct_end|>`

### 枚举来源（动态读取）
Labeler 与 Dataset Builder 通过以下文件动态读取枚举内容：
- `src/common/enums/dsl_ops.json`（操作符与参数约束）
- `src/common/enums/done_evidence.json`（完成证据）

### 视频片段
- **recent_clip**：`[t-7..t]`（8 帧，4 秒）
- **summary_clip**：`[t-60s..t]` 每 2 秒采 1 帧（约 30 帧）
- **lookahead_clip**：`[t..t+7]`（8 帧，4 秒）
- **lookahead_summary_clip**：`[t..t+60s]` 每 2 秒采 1 帧（约 30 帧，仅标注用）
> 采用 base64（JPEG）传输帧。
> `lookahead_clip` 与 `lookahead_summary_clip` 仅用于标注判断，不进入训练输入。

## 输出（JSON-only）
```json
{
  "goal": "<|goal_start|>长期目标/中期目标<|goal_end|>",
  "next_mid_step": "收取基建产出",
  "short_goal_dsl": [{"op": "MOVE_NAV", "args": {"...": "..."}}],
  "horizon_steps": 10,
  "done_evidence": ["dialog_open"],
  "uncertainty": "low|mid|high",
  "attempt": "接下来我应该先打开地图，寻找四号谷地，并传送"
}
```

### 字段约束
- `goal`：只允许 `<|goal_start|>长期目标/中期目标<|goal_end|>` 格式。
- `short_goal_dsl`：仅使用 DSL op 枚举表中的操作。
- `done_evidence`：仅使用证据枚举表中的条目。
- `uncertainty`：用于过滤和抽检，`high` 默认剔除。

### 自动生成字段
- `schema_version`：由 Dataset Builder 注入固定值（如 `plan_v1.0`）。
- `plan_id`：由 Dataset Builder 生成（如 `plan_{episode_id}_{t}`）。
- 若输出缺失字段，Dataset Builder 按配置注入默认值。

## Prompt 约束
**System**
- 你是游戏自动化数据标注助手。
- 只输出 JSON，严格符合 schema，不得输出解释文字。
- short_goal 必须在 1–10 秒内可执行。
- `goal` 仅供参考，不可替代对画面与时间序列的判断，可能错误或为空，请修正。
- `lookahead_clip` 与 `lookahead_summary_clip` 仅用于判断 next_mid_step / horizon_steps / done_evidence，不用于生成 short_goal_dsl。
- 任务说明：
  - 输入包含：
    - `recent_clip`（当前到过去 4 秒）
    - `summary_clip`（过去 60 秒摘要）
    - `lookahead_clip`（未来 4 秒）
    - `lookahead_summary_clip`（未来 60 秒摘要）
    - `goal`：以"<|goal_start|>长期目标/中期目标<|goal_end|>"的形式给出，长期目标（如完成每日任务、完成主线任务），中期目标（如通过副本、与npc对话）
    - `labeling_instruct`: 以"<|labeling_instruct_start|>..<|labeling_instruct_end|>"，是对于当前标注片段的参考信息
  - 输出字段含义：
    - `goal`：修正后的 goal string
    - `next_mid_step`：若当前步骤完成则输出下一步，否则保持当前步骤。
    - `short_goal_dsl`：短期可执行目标，操作符与参数必须来自 `dsl_ops_enum`。
    - `horizon_steps`：short_goal 预计持续的步数（单位=帧，2FPS，1 step=0.5s）。
    - `done_evidence`：完成证据列表，必须来自 `done_evidence_enum`。
    - `fallback_if_failed`：失败回退动作列表，必须来自枚举。
    - `uncertainty`：标注置信度（low/mid/high）。
    - `attempt`：对历史总结，当前的思考以及未来的规划。

**User**
- recent_clip
- summary_clip
- lookahead_clip
- lookahead_summary_clip
- goal
- labeling_instruct
- dsl_ops_enum / done_evidence_enum

## HTTP 接口与批处理
建议使用批量请求：
```json
{
  "items": [
    {
      "recent_clip": [{"mime": "image/jpeg", "data": "<base64>"}],
      "summary_clip": [{"mime": "image/jpeg", "data": "<base64>"}],
      "lookahead_clip": [{"mime": "image/jpeg", "data": "<base64>"}],
      "lookahead_summary_clip": [{"mime": "image/jpeg", "data": "<base64>"}],
      "goal": "<|goal_start|>长期目标/中期目标<|goal_end|>",
      "labeling_instruct": "<|labeling_instruct_start|>..<|labeling_instruct_end|>"
    }
  ]
}
```

### 推荐策略
- `batch_size=8`
- `max_retries=3`，指数退避 1s/2s/4s
- 缓存键：`clip_hash + goal + schema_version`

## 质量控制
- 强制 schema 校验与枚举校验，不通过直接剔除。
- 记录 `uncertainty` 分布与 `next_mid_step` 覆盖率。
- 对 `goal` / `next_mid_step` 设定固定比例人工抽检样本。

## 输出去向
Labeler 输出作为 Dataset Builder 的监督信号：
- Planner 数据集：`recent_clip + summary_clip + goal + labeling_instruct + retrieved_memory -> plan_json`
- Controller 数据集：`recent_clip + goal + labeling_instruct -> plan_json`（仅用于 span 对齐与短期执行监督）
