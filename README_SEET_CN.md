# SEET（Self-Play with Environment Tuning）实现说明

本文档说明在 EnvTuning 代码库中为实现你给出的 SEET 方案而新增/修改的核心代码。

## 1. 总体改动概览

本次实现围绕四个核心能力展开：

1. **环境即教学（Environment as Pedagogy）**：在交互层引入 `seet` 配置，按 stage 控制增强环境模式。  
2. **双通道提示注入（Fast/Slow Loop）**：
   - Fast Loop：当解析错误或执行错误时，根据重试概率注入纠错 hint 并在线重试；
   - Slow Loop：保留反事实训练记录（失败轨迹 vs 锚点轨迹 + 诊断）。
3. **FPLD（第一逻辑分歧点）诊断**：新增 AST 级别的首分歧定位与诊断文本生成。
4. **动态锚点选择（Dynamic Anchor Selection）**：支持 Stage2 的课程诱导锚点与 Stage3/4 的历史自我锚点优先级机制。

---

## 2. 新增代码

### 2.1 `env_tuning/seet/` 模块

#### `env_tuning/seet/config.py`
新增 `SeetConfig`，用于管理：
- `enabled`
- `stage`
- `retry_probability`
- `max_retry_per_turn`

并提供 `use_augmented_env`、`allow_peer_anchor`、`allow_historical_anchor`、`allow_induced_anchor` 等属性，用于在交互中按课程阶段切换逻辑。

#### `env_tuning/seet/fpld.py`
新增 FPLD 诊断实现：
- 将一步工具调用标准化为 `ToolNode(tool_name, arguments)`；
- 顺序比较失败轨迹与锚点轨迹，定位第一处 `tool_name` / `arguments` 差异；
- 输出中文诊断语句，用于重试提示或后续训练样本构建。

#### `env_tuning/seet/anchor.py`
新增锚点与回放池：
- `AnchorTrace`：记录 `entry_id / turn_index / decoded_calls / anchor_type`；
- `AnchorReplayBuffer`：维护历史成功轨迹；
- `DynamicAnchorSelector`：按 SEET 优先级（Peer > Historical > Induced）选择锚点。

#### `env_tuning/seet/runtime.py`
新增运行时调度器 `SeetRuntime`，负责：
- 按概率决定是否进行 Fast Loop 重试；
- 成功轨迹入回放池；
- 结合 FPLD 生成 retry hint；
- 生成可用于慢通道训练的 counterfactual record。

#### `env_tuning/seet/__init__.py`
导出 `SeetConfig / SeetRuntime / FPLD / Anchor` 相关接口，便于主交互模块直接使用。

---

## 3. 修改代码

### 3.1 `env_tuning/interaction/new_multi_turn_fc.py`
这是 SEET 接入的主入口，改动点：

1. **初始化接入 SEET**
   - 从 `config["seet"]` 读取 `SeetConfig`；
   - 开启时创建 `SeetRuntime`。

2. **解析错误路径接入 Fast Loop**
   - 原先解析失败直接给负分；
   - 现在在满足重试条件时，会通过 `SeetRuntime.build_retry_hint(...)` 返回系统纠错提示，触发在线重试。

3. **执行后路径接入锚点与重试**
   - 成功执行会写入 replay buffer（Stage2 作为 induced / Stage3+ 作为 standard）；
   - 执行失败时，根据 stage 与锚点可用性生成 FPLD hint 并重试。

4. **执行回包中增加 stage/environment 信息**
   - 调用 `ExecutionManager.format_execution_response(..., stage, augmented_env)`，把当前 SEET 课程状态回传给模型。

### 3.2 `env_tuning/interaction/execution_manager.py`
`format_execution_response` 扩展入参：
- `stage: int = None`
- `augmented_env: bool = False`

返回提示文本会附带当前 stage 与环境模式（augmented / standard），帮助模型显式感知课程阶段。

### 3.3 `env_tuning/interaction/data_models.py`
`InstanceState` 新增字段：
- `seet_counterfactual_records: List[Dict[str, Any]]`

用于承接慢通道（反事实训练）样本记录。

---

## 4. 配置与训练脚本改动

### 4.1 新增分阶段 interaction 配置
新增：
- `env_tuning/config/multi_turn_fc_interaction_stage1.yaml`
- `env_tuning/config/multi_turn_fc_interaction_stage2.yaml`
- `env_tuning/config/multi_turn_fc_interaction_stage3.yaml`
- `env_tuning/config/multi_turn_fc_interaction_stage4.yaml`

每个文件显式配置对应 stage 的：
- `retry_probability`
- `max_retry_per_turn`

### 4.2 修改 GRPO 配置绑定 stage interaction
修改：
- `env_tuning/config/multi_turn_fc_grpo_stage1.yaml`
- `env_tuning/config/multi_turn_fc_grpo_stage2.yaml`
- `env_tuning/config/multi_turn_fc_grpo_stage3.yaml`

将 `interaction_config_path` 改为对应 stage 的 interaction yaml。

### 4.3 新增 Stage4 配置
新增：
- `env_tuning/config/multi_turn_fc_grpo_stage4.yaml`

用于“关闭重试、标准环境、零样本鲁棒性测试”阶段。

### 4.4 新增 Stage4 启动脚本
新增：
- `scripts/run_multi_turn_fc_grpo_stage4.sh`

基于 stage3 脚本扩展，改为加载 `multi_turn_fc_grpo_stage4`。

---

## 5. 与你给出的 SEET 方案映射关系

- **环境即教学**：通过 `SeetConfig.use_augmented_env` + stage 配置化开关实现。  
- **双通道提示注入**：Fast Loop 已在在线交互中生效；Slow Loop 的数据结构与构建接口（counterfactual record）已接入。  
- **FPLD**：`first_logic_divergence` 已实现并用于 retry hint。  
- **动态锚点选择**：`DynamicAnchorSelector` 已按 stage 优先级实现（Peer/History/Induced 机制框架完备）。

---

## 6. 你接下来可以直接做的事

1. 先跑 Stage1/2 验证 Fast Loop 是否提升可执行率；
2. 在训练管线中把 `seet_counterfactual_records` 接到样本构造器（SFT/RL 混合）以完全打通 Slow Loop；
3. 在 batch 维度补充 peer anchor 检索（同 batch 成功样本索引）以强化 Stage3 的锚点质量。

如果你需要，我可以继续下一步：**把 Slow Loop 的反事实样本直接接进现有 reward/数据管线，给你一版可直接训练的 end-to-end 脚本**。
