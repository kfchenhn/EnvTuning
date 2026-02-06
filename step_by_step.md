# EnvTuning → SEET 的逐步实现说明（Step by Step）

> 目标：从原始 EnvTuning 多轮函数调用框架，逐步落地到可复现实验的 SEET（Self-Play with Environment Tuning）训练范式。  
> 结构原则：每一步都给出 **目标（Why）→ 实施（What）→ 验证（How）→ 产出（Done）**，确保工程与论文描述严格对齐。

---

## Step 0：建立可对照基线（Freeze Baseline）

### Why（目标）
在引入 SEET 机制前，先冻结“原系统能力边界”，避免后续迭代中出现“效果提升但来源不清”的问题。

### What（实施）
- 明确当前流程：`parse → decode → execute → feedback → turn advance`。
- 识别与 SEET 的能力差距：
  1. 无课程学习（Stage1~4）控制面；
  2. 无 Fast/Slow 双通道；
  3. 无 FPLD 首分歧定位；
  4. 无动态锚点回放；
  5. 无反事实样本到训练目标的闭环。

### How（验证）
- 基线任务可以跑通；
- 错误场景只有惩罚，没有结构化纠偏。

### Done（产出）
- 得到 SEET 改造的“最小可验证起点”。

---

## Step 1：先搭控制面——SEET 配置与阶段语义

### Why（目标）
把课程机制从“散乱 if/else”升级为“配置驱动”，保证可复现实验与消融。

### What（实施）
- 新增 `env_tuning/seet/config.py`：`SeetConfig`。
- 配置关键参数：
  - `stage`、`retry_probability`、`max_retry_per_turn`；
  - Stage3 退火：`stage3_retry_start/end`；
  - Stage2 拦截：`enable_stage2_interception`；
  - 回放池可选持久化：`replay_buffer_path`。
- 分阶段 YAML：
  - `env_tuning/config/multi_turn_fc_interaction_stage{1,2,3,4}.yaml`
  - `env_tuning/config/multi_turn_fc_grpo_stage{1,2,3,4}.yaml`

### How（验证）
- 切换 stage 时，环境模式/重试策略/拦截行为随配置变化。

### Done（产出）
- 课程学习具备“可控、可配、可复现”基础。

---

## Step 2：实现锚点系统——把成功轨迹变成可学习资产

### Why（目标）
SEET 的本质不是“失败重试”，而是“失败轨迹对齐成功锚点”，让错误可被修复为训练信号。

### What（实施）
- 新增 `env_tuning/seet/anchor.py`：
  - `AnchorTrace`：单轮成功轨迹；
  - `AnchorReplayBuffer`：历史成功轨迹池（支持可选持久化）；
  - `DynamicAnchorSelector`：锚点优先级选择器（Peer > Historical > Induced）。
- 在交互成功路径注册锚点：
  - Stage1/2 记为 induced；
  - Stage3/4 记为 standard。

### How（验证）
- 成功回合后回放池可增长；
- 失败样本可取到对应 turn 的历史锚点。

### Done（产出）
- 锚点体系可稳定支持“失败→对照→纠偏”。

---

## Step 3：实现 FPLD——把“失败”转化为“可定位逻辑偏差”

### Why（目标）
仅有最终奖励不足以指导长轨迹改进，必须定位第一处逻辑分歧（first mistake）。

### What（实施）
- 新增 `env_tuning/seet/fpld.py`：
  - 兼容 dict / 字符串调用（如 `tool(a=1)`）；
  - 使用 AST 进行参数标准化与比较；
  - 输出 `FPLDResult(divergence_index, diagnosis)`。

### How（验证）
- 构造 fail/anchor 轨迹：
  - 工具名不同；
  - 参数不同；
  - 轨迹长度不同；
- 均能返回正确首分歧位置与可读诊断。

### Done（产出）
- 稀疏结果信号被转换为稠密、可监督诊断信息。

---

## Step 4：实现 Runtime——把规则组装成可执行策略引擎

### Why（目标）
配置、锚点、FPLD 都是组件，需统一编排器将其变成训练时的行为策略。

### What（实施）
- 新增 `env_tuning/seet/runtime.py`：
  1. `should_retry`：按 stage/turn 的概率重试（Stage3 线性退火）；
  2. `build_retry_hint`：基于锚点 + FPLD 生成 Fast Loop 提示；
  3. `stage2_ground_truth_interception`：Stage2 偏离拦截（复用 FPLD 标准化比较）；
  4. `build_counterfactual_record`：构建 Slow Loop 反事实记录，并标记 `has_divergence`。
- 回放池支持按配置持久化更新。

### How（验证）
- Stage3 多轮下重试概率可按轮次退火；
- Stage2 错误调用会在执行前被拦截；
- 反事实记录可携带首分歧信息。

### Done（产出）
- 得到可复用的 SEET 策略运行时。

---

## Step 5：接入交互主流程——让 SEET 真正“在线生效”

### Why（目标）
只有把 Runtime 融入交互 loop，Fast/Slow 双通道才会在真实 rollout 中发生作用。

### What（实施）
- 修改 `env_tuning/interaction/new_multi_turn_fc.py`：
  1. 初始化 `SeetRuntime`；
  2. 解析失败路径接入 Fast Loop（hint 重试）；
  3. Stage2：先解码→真值拦截→再执行；
  4. 执行失败接 Fast Loop 纠偏；
  5. 执行成功写入锚点；
  6. 失败时沉淀 Slow Loop 反事实记录。
- 修改 `execution_manager.py` 支持 `predecoded_responses`，避免重复 decode。

### How（验证）
- 错误轨迹可拿到 `[SEET-FPLD]` 提示；
- Stage2 偏离不会触发错误副作用；
- 成功轨迹可在后续失败中被选为参考锚点。

### Done（产出）
- Fast Loop 在线纠偏能力完成接线。

---

## Step 6：打通 Slow Loop 导出——先“出得来”

### Why（目标）
反事实数据若只在内存里，不可被训练器消费，等同未实现。

### What（实施）
- 修改 `env_tuning/interaction/data_models.py`：
  - 新增 `seet_counterfactual_records`；
  - 新增 `pop_seet_counterfactual_records()`。
- 修改 `env_tuning/interaction/turn_manager.py`：
  - 在 `advance_to_next_turn` 时将记录写入 `extra` 并清空。

### How（验证）
- 每轮 `extra` 可看到 `seet_counterfactual_records`；
- 跨轮不会出现旧记录污染。

### Done（产出）
- Slow Loop 样本具备可传输性。

---

## Step 7：打通 rollout 聚合——再“传得到”

### Why（目标）
交互层导出后还需进入训练批次，才能影响奖励与优化。

### What（实施）
- 修改 `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`：
  - 聚合 interaction 指标到 `reward_scores["interaction_turn_metrics"]`。

### How（验证）
- rollout 结果中可读取 `interaction_turn_metrics`；
- 每条数据包含 SEET 相关字段。

### Done（产出）
- 反事实信号已从 interaction 到达训练输入端。

---

## Step 8：接入奖励函数——最后“用得上”

### Why（目标）
Slow Loop 必须进入优化目标，否则只是辅助日志。

### What（实施）
- 修改 `env_tuning/bfcl_reward.py`：
  - 提取 counterfactual 计数；
  - 计算 `seet_slow_loop_bonus`（带系数和上限）；
  - 合并到最终 `score`。

### How（验证）
- synthetic `reward_scores` 注入 counterfactual 记录后，最终 score 增加。

### Done（产出）
- Slow Loop 已转化为可学习奖励信号。

---

## Step 9：文档与入口对齐——保障可复现执行

### Why（目标）
“方案正确”还不够，必须“别人能一键复现”。

### What（实施）
- 文档更新：
  - `README_SEET_CN.md`（机制映射 + 使用方式）；
  - `seet_doc.md`（方案边界与实现状态）；
  - 本文档 `step_by_step.md`（工程拆解路径）。
- 启动脚本：
  - `scripts/run_multi_turn_fc_grpo_stage{1,2,3,4}.sh`。

### How（验证）
- 文档中每个机制都能映射到具体代码；
- 各 stage 可独立启动。

### Done（产出）
- 具备端到端复现实验基础。

---

## Step 10：与论文设定逐项对表（SEET Checklist）

### 10.1 Environment as Pedagogy
- Stage1/2 使用增强环境语义；Stage3/4 回到标准环境。

### 10.2 Dual-Phase Hint Injection
- Fast Loop：在线重试与提示；
- Slow Loop：反事实记录并进入 reward。

### 10.3 Hindsight Logic Diagnosis (FPLD)
- 支持工具名、参数、轨迹长度层面的首分歧诊断。

### 10.4 Dynamic Anchor Selection
- 策略优先级框架：Peer > Historical > Induced（当前主流程已稳定覆盖 Historical/Induced，Peer 入口已预留）。

---

## Step 11：本次优化补丁（面向“完美复现”的关键修正）

1. **Stage2 拦截比较逻辑升级为 FPLD 标准化路径**  
   - 解决参数顺序/字符串形式导致的误判风险。
2. **Slow Loop 记录增加 `has_divergence` 字段**  
   - 便于训练/分析区分“有效纠偏样本”与“长度差异或无差异样本”。
3. **步骤文档细化为 Why/What/How/Done 模式**  
   - 让工程落地路径与论文叙事一一对齐。

---

## 最终结论

当前 EnvTuning 代码已形成可运行、可观测、可优化的 SEET 闭环：

- 课程化阶段控制 ✅  
- Fast Loop 在线纠偏 ✅  
- Stage2 真值拦截 ✅  
- FPLD 首分歧诊断 ✅  
- 锚点回放与动态选择框架 ✅  
- Slow Loop 反事实记录导出与传递 ✅  
- Slow Loop 奖励注入并参与优化 ✅

若下一步追求“论文级完全等价”，建议新增一条**显式 counterfactual supervised loss**（与 reward bonus 并行），让慢通道对策略更新的影响更直接。
