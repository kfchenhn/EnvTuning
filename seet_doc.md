# SEET 方案介绍与在 EnvTuning 中的实现（更新版）

## 1. 方案概览

SEET（Self-Play with Environment Tuning）用于多轮工具调用智能体训练，核心是把“环境反馈”从被动评测改造成主动教学。

- Environment as Pedagogy：环境在早期提供可操作引导。
- Error as Data：失败轨迹不只是惩罚，还会沉淀为可训练样本。
- Fast/Slow Loop：在线纠偏 + 离线内化的双通道闭环。

---

## 2. 当前代码实现要点

### 2.1 Fast Loop（在线纠偏）

- 解析失败或执行失败时，`SeetRuntime.build_retry_hint` 会尝试选择锚点并生成重试提示。
- **本次更新后，诊断语句统一改为英文**（含 FPLD 诊断、Stage2 拦截提示、通用 SEET 重试提示），降低多语言混杂导致的歧义。

### 2.2 Slow Loop（反事实样本）

- `build_counterfactual_record` 生成 `fail_calls / anchor_calls / divergence_index / diagnosis`。
- 记录通过 `seet_counterfactual_records` 从 interaction 侧传到 rollout，再由 `bfcl_reward.py` 提取并形成 `seet_slow_loop_bonus`，进入最终 reward。

### 2.3 FPLD（第一逻辑分歧点）

- 支持两类轨迹输入：
  1) dict 结构调用；
  2) 字符串调用（如 `tool(a=1)`）。
- 使用 AST 做归一化后比较，输出第一处分歧与英文诊断文本。

### 2.4 Stage2 真值拦截

- 当模型调用与 ground truth 前缀不一致时，立即拦截并给出更自然的英文纠偏提示。
- 目标是降低冷启动阶段错误副作用，并更稳定地产生可学习的反事实样本。


### 2.5 Replay Buffer 持久化（可选）

- 默认行为：回放池仅在进程内保存，训练结束后释放。
- 新增能力：可通过 `SeetConfig.replay_buffer_path` + `persist_replay_buffer_on_update` 开启文件级持久化。
- 作用：在多次训练作业或中断恢复场景下复用历史锚点。

---

## 3. 关键文件映射

- `env_tuning/seet/fpld.py`：FPLD 归一化与英文诊断。
- `env_tuning/seet/runtime.py`：重试策略、锚点选择、Stage2 拦截、慢通道样本构造。
- `env_tuning/interaction/new_multi_turn_fc.py`：主流程编排（快慢通道接入点）。
- `env_tuning/bfcl_reward.py`：Slow Loop 统计与奖励加成。

---

## 4. 可读性增强（本次补充）

根据你的要求，已在关键路径添加中文注释，重点覆盖：

- `SeetRuntime`：课程策略中枢职责、锚点选择、Slow Loop 样本构造。
- `new_multi_turn_fc.py`：解析失败快通道入口、Stage2 拦截入口、执行结果处理入口。
- `bfcl_reward.py`：Slow Loop bonus 如何映射到最终 reward。

这些注释的目标是：让读代码时先理解“为什么这样做”，再看“具体怎么做”。

---

## 5. 与“完整复现”目标的对应关系

当前实现已覆盖：

1. 四阶段课程（Stage1~4）与分阶段重试/环境策略；
2. Fast Loop 在线重试；
3. Slow Loop 反事实记录；
4. FPLD 分歧定位；
5. Stage2 真值拦截；
6. Slow Loop 信号进入 reward 的端到端闭环。

如需继续提升，可在 trainer 侧加入“直接消费 counterfactual record 的监督项”，进一步强化慢通道学习强度。
