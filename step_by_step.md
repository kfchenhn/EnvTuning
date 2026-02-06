# EnvTuning → SEET 的逐步实现说明（Step by Step）

> 目标：从“原始 EnvTuning 多轮函数调用训练”出发，按**功能闭环**一步步落地到当前 SEET（Self-Play with Environment Tuning）实现。  
> 写法原则：每一步都回答 3 件事——**为什么做、改了哪里、如何验证**。

---

## Step 0：建立基线（原始 EnvTuning）

### 为什么做
在改造前先明确“已有能力”和“缺口”，否则后续改动容易堆叠但不成体系。

### 基线能力
- 已有多轮交互主流程（解析模型输出 → 调工具 → 回写环境反馈）。
- 已有按轮次推进、终止、打分的基础逻辑。
- 但缺少 SEET 所需的：
  1. 课程化环境策略（Stage1~4）；
  2. 快通道（在线重试）+ 慢通道（反事实样本）双通道；
  3. FPLD 首分歧诊断；
  4. 动态锚点复用机制；
  5. 反事实样本进入训练目标的闭环通路。

### 验证
- 保留基线流程可运行，后续每步在此之上增量演进。

---

## Step 1：抽象 SEET 配置与阶段控制（先立“控制面”）

### 为什么做
不先配置化，后面所有行为会散落在 if/else 中，无法稳定复现。

### 实现
- 新增 `env_tuning/seet/config.py`，引入 `SeetConfig`：
  - `stage`、`retry_probability`、`max_retry_per_turn`；
  - Stage3 线性退火参数（`stage3_retry_start/end`）；
  - Stage2 拦截开关（`enable_stage2_interception`）。
- 新增分阶段 interaction/grpo 配置文件：
  - `env_tuning/config/multi_turn_fc_interaction_stage{1,2,3,4}.yaml`
  - `env_tuning/config/multi_turn_fc_grpo_stage{1,2,3,4}.yaml`

### 验证
- 检查各 stage 的重试/环境策略由配置驱动，而非硬编码。

---

## Step 2：实现锚点系统（先把“可学习参考答案”存起来）

### 为什么做
SEET 的核心不是只“报错重试”，而是“失败轨迹可对照成功轨迹纠偏”。锚点是这个能力的基础。

### 实现
- 新增 `env_tuning/seet/anchor.py`：
  - `AnchorTrace`：成功轨迹结构化存储；
  - `AnchorReplayBuffer`：历史成功样本池；
  - `DynamicAnchorSelector`：按优先级选锚点（Peer > Historical > Induced，当前主路径覆盖 Historical/Induced）。
- 在交互成功路径注册锚点（stage<3 标记 induced，stage>=3 标记 standard）。

### 验证
- 成功轮次后回放池长度增长；
- 失败时可取到可用 anchor 作为纠偏目标。
- （优化）支持回放池可选持久化，满足跨进程复用历史锚点的场景。

---

## Step 3：实现 FPLD（First Point of Logic Divergence）

### 为什么做
只知道“失败了”无法形成高质量学习信号；要定位“第一处逻辑分歧”。

### 实现
- 新增 `env_tuning/seet/fpld.py`：
  - 兼容两类输入：dict 调用 / 字符串调用（如 `tool(a=1)`）；
  - 用 AST 标准化参数结构后比较；
  - 输出 `divergence_index + diagnosis`。
- 诊断语句已统一为英文，避免训练日志混杂。

### 验证
- 构造 fail/anchor 轨迹，确认返回首分歧位置与可读诊断文本。

---

## Step 4：实现 SEET Runtime（把规则变成“可执行策略引擎”）

### 为什么做
配置、锚点、FPLD 都是“零件”；需要一个 runtime 在每轮交互中统一编排。

### 实现
- 新增 `env_tuning/seet/runtime.py`：
  - `should_retry`：基于轮次与 stage 的重试决策（Stage3 支持线性退火）；
  - `build_retry_hint`：失败后选择锚点 + FPLD 生成快通道提示；
  - `stage2_ground_truth_interception`：Stage2 真值拦截；
  - `build_counterfactual_record`：生成慢通道记录。
- 关键路径添加中文注释，便于二次维护。

### 验证
- 固定输入下，Fast Loop 能输出预期英文提示；
- Stage2 偏离时确实拦截；
- Slow Loop record 字段齐全。

---

## Step 5：接入主交互流程（SEET 真正“跑起来”）

### 为什么做
SEET 不是独立模块，必须嵌入 `new_multi_turn_fc` 的解析/执行/轮转路径。

### 实现
- 修改 `env_tuning/interaction/new_multi_turn_fc.py`：
  1. 初始化注入 `SeetRuntime`；
  2. 解析失败路径接 Fast Loop（重试提示）；
  3. Stage2：先解码、再真值拦截、再执行（避免错误副作用）；
  4. 执行失败路径接 Fast Loop；
  5. 执行成功时注册锚点；
  6. 失败时沉淀 counterfactual record。
- `execution_manager.py` 支持复用预解码结果，避免重复解码。

### 验证
- 失败轮可重试、成功轮可注册 anchor；
- Stage2 拦截生效后不执行错误调用。

---

## Step 6：打通 Slow Loop 数据出口（先“出得来”）

### 为什么做
只在内存里积累反事实样本，不算闭环；必须在 turn 结束时可导出。

### 实现
- 修改 `env_tuning/interaction/data_models.py`：
  - 增加 `seet_counterfactual_records` 字段；
  - 增加 `pop_seet_counterfactual_records()`。
- 修改 `env_tuning/interaction/turn_manager.py`：
  - `advance_to_next_turn` 时将记录挂入 `extra` 并清空。

### 验证
- 每轮推进时 `extra` 中可见 `seet_counterfactual_records`。

---

## Step 7：打通 rollout 聚合（再“传得到”）

### 为什么做
interaction 导出了记录，还需进入训练样本的 reward_scores。

### 实现
- 修改 `verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`：
  - 聚合 interaction 侧 metrics 到 `reward_scores["interaction_turn_metrics"]`。

### 验证
- rollout 产物中可读取 `interaction_turn_metrics`。

---

## Step 8：接入奖励函数（最后“用得上”）

### 为什么做
Slow Loop 不参与 loss，就只是日志。必须让其影响最终优化目标。

### 实现
- 修改 `env_tuning/bfcl_reward.py`：
  - `_extract_seet_counterfactual_count` 统计反事实样本数；
  - 计算 `seet_slow_loop_bonus`（系数 + 上限）；
  - 合并到最终 `score`，进入 PPO/GRPO 的 reward tensor。

### 验证
- 用 synthetic reward_scores 注入 interaction_turn_metrics，确认 score 包含 slow-loop bonus。

---

## Step 9：文档与运行入口补齐

### 为什么做
工程可复现必须有文档与启动入口。

### 实现
- 新增/更新：
  - `README_SEET_CN.md`（实现映射与使用说明）；
  - `seet_doc.md`（方案说明与当前实现边界）；
  - `scripts/run_multi_turn_fc_grpo_stage4.sh`（Stage4 启动脚本）。

### 验证
- 文档与代码路径一一对应；
- stage 启动脚本可直接复用。

---

## 本次额外复盘：发现并优化的“不合理点”

在整理 step-by-step 过程中，确认并修正/固化了以下点：

1. **诊断语句语言混杂风险**  
   - 不合理：中文/英文诊断混用会增加调试噪音，也不利于统一日志分析。  
   - 优化：FPLD 与 runtime 提示统一英文。

2. **关键路径可读性不足风险**  
   - 不合理：SEET 逻辑跨多个文件，后续维护者难以快速定位“为何这样设计”。  
   - 优化：在 runtime、interaction 主决策、reward 映射处补充中文注释，强调“设计意图”而非仅描述代码动作。

3. **Slow Loop 只记录不参与优化的风险（已闭环）**  
   - 不合理：若反事实数据不进入 reward/loss，SEET 的慢通道价值大幅下降。  
   - 优化：记录已贯通 interaction → rollout → reward，形成端到端闭环。

---

## 最终实现状态（结论）

当前代码已形成从 EnvTuning 到 SEET 的完整增量路径：

- 课程阶段控制（Stage1~4）✅
- Fast Loop 在线纠偏 ✅
- Stage2 真值拦截 ✅
- FPLD 首分歧诊断 ✅
- 动态锚点与回放 ✅
- Slow Loop 反事实记录导出 ✅
- Slow Loop 信号进入奖励与 loss ✅

如果下一步要进一步“强化完美复现”，建议在 trainer 侧再加入一条显式的 counterfactual supervised loss（与 reward bonus 并行），让慢通道学习信号更直接。
