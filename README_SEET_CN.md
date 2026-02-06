# SSET / SEET 方案复现说明（EnvTuning）

> 说明：你在需求里写的是 **SSET**，结合上下文这里按论文中的 **SEET（Self-Play with Environment Tuning）** 实现。

本文档聚焦三件事：
1. 这次代码到底新增/修改了什么；
2. 这些改动如何一一对应到 SEET 核心机制；
3. Slow Loop 是否已经真正进入训练目标（端到端闭环）。

---

## 1. 实现状态（最新）

当前版本已经完成以下能力：

- ✅ **环境即教学（Environment as Pedagogy）**：按 Stage 选择增强环境/标准环境。
- ✅ **双通道提示注入（Fast/Slow Loop）**：
  - Fast Loop：在线错误重试 + FPLD 纠偏提示；
  - Slow Loop：失败-锚点反事实样本持续沉淀。
- ✅ **FPLD（第一逻辑分歧点）**：支持真实运行时字符串调用（如 `foo(a=1)`）与 dict 调用。
- ✅ **动态锚点策略**：`Peer > Historical > Induced` 优先级框架。
- ✅ **Stage2 真值拦截**：调用偏离即拦截并回注纠偏信息。
- ✅ **Slow Loop 训练闭环**：`seet_counterfactual_records` 已进入 reward 计算并影响最终优化目标。

---

## 2. 新增模块（SEET 核心）

### 2.1 `env_tuning/seet/config.py`

`SeetConfig` 提供课程配置项：

- 基础：`enabled`, `stage`, `retry_probability`, `max_retry_per_turn`
- Stage3 退火：`stage3_retry_start`, `stage3_retry_end`
- Stage2 开关：`enable_stage2_interception`

并提供课程语义属性：
- `use_augmented_env`
- `allow_peer_anchor`
- `allow_historical_anchor`
- `allow_induced_anchor`

---

### 2.2 `env_tuning/seet/fpld.py`

FPLD 逻辑诊断：

- 支持 dict + 字符串调用两种输入格式；
- 字符串调用通过 AST 归一化；
- 输出 `FPLDResult(divergence_index, diagnosis)`。

---

### 2.3 `env_tuning/seet/anchor.py`

锚点机制：

- `AnchorTrace`：单轮锚点轨迹；
- `AnchorReplayBuffer`：历史成功回放池；
- `DynamicAnchorSelector`：动态优先级选锚点。

---

### 2.4 `env_tuning/seet/runtime.py`

SEET 运行时编排：

- `should_retry`：快通道重试（Stage3 支持线性退火）；
- `build_retry_hint`：基于锚点 + FPLD 生成纠偏提示；
- `build_counterfactual_record`：构建慢通道样本；
- `stage2_ground_truth_interception`：Stage2 真值拦截。

---

## 3. 关键改动文件

### 3.1 `env_tuning/interaction/new_multi_turn_fc.py`

主交互流程已完成 SEET 接入：

1. 初始化 `SeetRuntime`；
2. 工具调用先解码，Stage2 先拦截后执行；
3. 执行失败触发 Fast Loop 提示重试；
4. 可用锚点时落地 Slow Loop 反事实样本；
5. 执行成功写入锚点池。

---

### 3.2 `env_tuning/interaction/execution_manager.py`

- 增加 `predecoded_responses`，避免重复解码；
- 保留 `decode_tool_calls` 作为独立入口；
- 执行反馈中注入 stage / env mode 语义。

---

### 3.3 `env_tuning/interaction/data_models.py` + `turn_manager.py`

- `InstanceState` 维护 `seet_counterfactual_records`；
- turn 切换时通过 `extra` 导出并清空反事实记录（避免跨轮污染）。

---

### 3.4 `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`

- 收集 interaction 返回的 `metrics`；
- 聚合为 `reward_scores["interaction_turn_metrics"]`，把 SEET 慢通道信号送入奖励侧。

---

### 3.5 `env_tuning/bfcl_reward.py`

- 从 `interaction_turn_metrics` 提取 `seet_counterfactual_records` 数量；
- 计算 `seet_slow_loop_bonus`（可配置系数与上限）；
- 将 bonus 加入最终 `score`，直接作用于 reward tensor。

---

## 4. Stage 课程映射（Stage1~4）

- **Stage1**：格式学习，高重试，增强环境。  
- **Stage2**：真值拦截冷启动，沉淀课程诱导锚点。  
- **Stage3**：标准环境，自博弈内化，重试概率线性退火（1.0→0.2）。  
- **Stage4**：关闭重试，鲁棒性实战。  

对应配置文件：
- `env_tuning/config/multi_turn_fc_interaction_stage1.yaml`
- `env_tuning/config/multi_turn_fc_interaction_stage2.yaml`
- `env_tuning/config/multi_turn_fc_interaction_stage3.yaml`
- `env_tuning/config/multi_turn_fc_interaction_stage4.yaml`

---

## 5. Slow Loop 是否已经“真闭环”？

是，当前已经闭环：

1. interaction 生成 `seet_counterfactual_records`；
2. turn manager 导出到每轮 `extra`；
3. rollout 聚合为 `interaction_turn_metrics`；
4. reward 函数提取记录数并计算 bonus；
5. bonus 融入 `score`，参与 PPO/GRPO 优化。

结论：Slow Loop 不再只是日志或附加信息，而是已成为训练目标的一部分。

---

## 6. 可读性设计说明

本实现遵循以下可读性原则：

1. **职责分层**：SEET 算法在 `env_tuning/seet/`，交互层只负责编排。  
2. **命名可读**：`build_retry_hint` / `stage2_ground_truth_interception` / `_register_success_anchor_if_needed` 等函数名即语义。  
3. **流程拆解**：主流程短小，复杂逻辑封装到私有方法。  
4. **配置优先**：课程行为尽可能由 yaml 控制，减少硬编码。  

---

## 7. 你可以直接运行的路径

- Stage1：`scripts/run_multi_turn_fc_grpo_stage1.sh`
- Stage2：`scripts/run_multi_turn_fc_grpo_stage2.sh`
- Stage3：`scripts/run_multi_turn_fc_grpo_stage3.sh`
- Stage4：`scripts/run_multi_turn_fc_grpo_stage4.sh`

如果你后续希望，我可以再补一版“训练日志解读指南”（如何从 reward 曲线中分离 `progress` 与 `seet_slow_loop_bonus` 的贡献），方便做 ablation 与论文复现实验。
