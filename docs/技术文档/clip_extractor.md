# 片段抽取器技术文档

## 目标与范围
片段抽取器负责从 **视频 + action string** 中生成用于标注/训练的样本片段，并同时输出：
- 训练用的索引文件（JSONL）
- 可抽检/可人工查看的样本文件夹（每样本一目录）

## 输入
- **视频**：2FPS（按帧序号读取）
- **动作**：每 500ms 一行 action string

## 对齐规则
不做时间戳对齐，直接按序号对齐：
- 第 i 帧对应第 i 行 action
- 缺帧/缺行动作直接丢弃该段或整段 episode

## 片段定义（与 Labeler 对齐）
以当前时刻 `t` 为锚点：
- **recent_clip**: `[t-7..t]`（8 帧，4 秒）
- **summary_clip**：`[t-60s..t]` 每 2 秒采 1 帧（约 30 帧）
- **lookahead_clip**：`[t..t+7]`（8 帧，4 秒）
- **lookahead_summary_clip**：`[t..t+60s]` 每 2 秒采 1 帧（约 30 帧，仅标注用）

> `lookahead_clip` 与 `lookahead_summary_clip` 仅用于标注判断 `next_mid_step / horizon_steps / done_evidence`，不进入训练输入。

## 输出
### 1) 索引文件（训练主产物）
`datasets/clip_index.jsonl`（示例字段）：
```json
{
  "sample_id": "ep001_t0123",
  "episode_id": "ep001",
  "anchor_t": 123,
  "recent_clip": ["frames/000123.jpg", "..."],
  "summary_clip": ["frames/000003.jpg", "..."],
  "lookahead_clip": ["frames/000123.jpg", "..."],
  "lookahead_summary_clip": ["frames/000003.jpg", "..."],
  "action_t": "<|action_start|>...<|action_end|>",
  "mid_step_id": "mid_XX",
  "mid_step_text": "..."
}
```

### 2) 样本文件夹（可选导出）
`datasets/clips/<sample_id>/`（推荐仅抽样导出）：
```
datasets/clips/ep001_t0123/
  recent/              # 8 帧
  summary/             # 30 帧
  lookahead/           # 8 帧（仅标注用）
  lookahead_summary/   # 30 帧（仅标注用）
  action.txt           # anchor 对应的动作行
  meta.json            # sample_id/anchor_t/mid_step
  label.json           # 标注输出（若已生成）
```
建议使用 **软链接/硬链接** 指向原始帧，避免重复拷贝。

## 采样策略
- **滑窗步长**：2 帧（= 1 秒）
- **边界**：只在同一 `mid_step` 区间内采样，不跨界
- **缺失处理**：若 recent/summary/lookahead/lookahead_summary 任何一段缺帧，则跳过该样本

## 训练便利性建议
- 训练读取 **以 JSONL 为主**（索引文件 + 原始帧路径）
- 文件夹只用于 **QA/抽检** 或少量人工分析
- 若必须全量导出样本文件夹，优先使用链接而不是复制
