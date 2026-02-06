# SSET / SEET 方案复现说明（EnvTuning）

> 说明：你在需求里写的是 **SSET**，结合上下文这里按论文中的 **SEET（Self-Play with Environment Tuning）** 实现。

本文档聚焦两件事：
1. 这次代码到底新增/修改了什么；
2. 这些改动如何一一对应到 SEET 四个核心机制，并保证代码可读性。

---

## 1. 实现目标与达成情况

本次实现完成了以下核心复现目标：

- ✅ **环境即教学（Environment as Pedagogy）**：支持按 Stage 配置增强环境/标准环境行为。
- ✅ **双通道提示注入（Fast/Slow Loop）**：
  - Fast Loop：在线错误重试 + 结构化提示；
  - Slow Loop：失败轨迹与锚点轨迹自动生成反事实记录。
- ✅ **FPLD（第一逻辑分歧点）**：能处理真实执行中的调用字符串格式（如 `foo(a=1)`）。
- ✅ **动态锚点策略**：按 `Peer > Historical > Induced` 优先级选锚点。
- ✅ **Stage2 真值拦截（Ground Truth Interception）**：偏离真值时先纠偏再执行。

---

## 2. 新增代码

### 2.1 `env_tuning/seet/config.py`

新增 `SeetConfig`：
- `enabled`
- `stage`
- `retry_probability`
- `max_retry_per_turn`

并暴露阶段属性：
- `use_augmented_env`
- `allow_peer_anchor`
- `allow_historical_anchor`
- `allow_induced_anchor`

用途：把训练课程（Stage1~4）映射为交互行为开关。

---

### 2.2 `env_tuning/seet/fpld.py`

新增 FPLD 诊断实现，重点增强可用性：

- 支持两种输入：
  1) 结构化 dict 调用；
  2) 字符串调用（`tool(arg=...)`）。
- 用 AST 将字符串调用解析为标准节点 `ToolNode(tool_name, arguments)`；
- 返回 `FPLDResult(divergence_index, diagnosis)`，可直接用于 fast-loop 提示或 slow-loop 样本。

---

### 2.3 `env_tuning/seet/anchor.py`

新增锚点数据结构与选择器：

- `AnchorTrace`：记录单轮成功轨迹；
- `AnchorReplayBuffer`：历史成功轨迹池；
- `DynamicAnchorSelector`：实现优先级：
  - Stage3+：Peer（若有）
  - Stage3/4：Historical
  - Stage2：Induced

---

### 2.4 `env_tuning/seet/runtime.py`

新增 `SeetRuntime`，封装：

- `should_retry`：快通道重试概率控制；
- `on_success`：成功轨迹写入锚点池；
- `build_retry_hint`：结合锚点 + FPLD 生成在线提示；
- `build_counterfactual_record`：构造慢通道反事实样本；
- `stage2_ground_truth_interception`：Stage2 真值拦截逻辑。

---

## 3. 修改代码

### 3.1 `env_tuning/interaction/new_multi_turn_fc.py`

这是本次最核心改动，主要包括：

1. **SEET 初始化接入**
   - 从 interaction config 读取 `seet` 参数；
   - 启用后创建 `SeetRuntime`。

2. **Stage2 真值拦截（新增）**
   - 工具调用解码后先进行 GT 前缀一致性检查；
   - 偏离则直接返回纠偏提示，不执行错误调用；
   - 同时写入 `seet_counterfactual_records`。

3. **快通道重试完善**
   - 解析错误与执行错误都可触发 fast-loop；
   - 返回 `channel=fast`、`reason=...` 便于训练侧统计。

4. **慢通道样本落地**
   - 在执行失败且存在可用锚点时，自动构建反事实样本并保存到 state。

5. **可读性重构**
   - 引入 `_maybe_stage2_intercept`、`_register_success_anchor_if_needed`、`_get_current_turn_ground_truth` 等私有函数，减少主流程复杂度。

---

### 3.2 `env_tuning/interaction/execution_manager.py`

改动点：

- `execute_function_calls` 新增 `predecoded_responses` 参数，支持“先解码后拦截再执行”的流程，避免重复解码。
- `format_execution_response` 增加 `stage` / `augmented_env` 上下文提示。
- 保留 `decode_tool_calls` 作为独立解码入口，便于交互层调用。

---

### 3.3 `env_tuning/interaction/data_models.py`

`InstanceState` 已包含：
- `seet_counterfactual_records: List[Dict[str, Any]]`

用于慢通道样本缓存。

---

## 4. 配置与脚本

### 4.1 新增 Stage 交互配置

新增：
- `env_tuning/config/multi_turn_fc_interaction_stage1.yaml`
- `env_tuning/config/multi_turn_fc_interaction_stage2.yaml`
- `env_tuning/config/multi_turn_fc_interaction_stage3.yaml`
- `env_tuning/config/multi_turn_fc_interaction_stage4.yaml`

内容包含 `stage/retry_probability/max_retry_per_turn`。

### 4.2 修改 GRPO 配置绑定分阶段 interaction

修改：
- `multi_turn_fc_grpo_stage1.yaml`
- `multi_turn_fc_grpo_stage2.yaml`
- `multi_turn_fc_grpo_stage3.yaml`

使其分别指向对应的 stage interaction config。

### 4.3 新增 Stage4 配置与脚本

新增：
- `env_tuning/config/multi_turn_fc_grpo_stage4.yaml`
- `scripts/run_multi_turn_fc_grpo_stage4.sh`

用于关闭重试、模拟实战鲁棒性阶段。

---

## 5. 复现映射（论文 -> 代码）

- **Environment as Pedagogy**
  - `SeetConfig.use_augmented_env`
  - `format_execution_response(... stage, augmented_env)`

- **Dual-Phase Hint Injection**
  - Fast Loop：`build_retry_hint` + interaction 错误路径在线重试
  - Slow Loop：`build_counterfactual_record` + `state.seet_counterfactual_records`

- **FPLD**
  - `first_logic_divergence`（支持字符串调用 AST 解析）

- **Dynamic Anchors**
  - `DynamicAnchorSelector.choose`

- **Stage2 GT Interception**
  - `stage2_ground_truth_interception` + `_maybe_stage2_intercept`

---

## 6. 可读性设计说明

本次代码可读性改进原则：

1. **职责单一**：
   - SEET 核心算法放 `env_tuning/seet/`；
   - interaction 只做流程编排。
2. **命名直观**：
   - `build_retry_hint` / `stage2_ground_truth_interception` / `_register_success_anchor_if_needed` 等函数名即语义。
3. **流程拆解**：
   - `generate_response` 主路径变短，关键决策拆到私有函数。
4. **最小侵入**：
   - 保留现有执行/评分框架，SEET 以“可配置增强”方式接入。

---

## 7. 你可以直接怎么用

- Stage1：格式学习 + 高重试。
- Stage2：开启 GT 拦截，积累课程诱导锚点。
- Stage3：标准环境 + 低概率重试，依赖历史锚点。
- Stage4：重试关闭，验证实战鲁棒性。

如果你愿意，我下一步可以继续把 `seet_counterfactual_records` 直接接入训练数据构造器（让 Slow Loop 真正参与 loss 计算），做到完整端到端闭环。
