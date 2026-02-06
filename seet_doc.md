# SEET 方案介绍与在 EnvTuning 中的实现

## 1. 什么是 SEET

SEET（Self-Play with Environment Tuning）是一个面向多轮工具调用智能体的训练范式，核心思想是：

- **环境不是被动评测器，而是主动教学器（Environment as Pedagogy）**；
- **错误不是纯惩罚信号，而是可转化的数据资产（Error as Data）**；
- 通过 **快通道（在线重试）+ 慢通道（反事实训练）**，逐步把“提示依赖”内化为“策略能力”。

SEET 的目标是破解智能体训练里的三难困境：数据稀缺、环境复杂、奖励稀疏。

---

## 2. SEET 的四个核心机制

### 2.1 环境即教学（Environment as Pedagogy）

将环境划分为两种模式：

- **增强环境（Stage1/2）**：出现错误时返回可操作提示（hint），帮助跨过冷启动；
- **标准环境（Stage3/4）**：逐步去提示，逼近真实线上场景。

在实现上由 `SeetConfig.stage` 控制，并在执行反馈中显式注入 `stage` 与 `environment mode`，让模型感知当前课程阶段。

### 2.2 双通道提示注入（Fast/Slow Loop）

- **Fast Loop（快通道）**：解析错误/执行错误时，在线注入提示并重试；
- **Slow Loop（慢通道）**：把“失败轨迹 vs 锚点轨迹”转成反事实样本（counterfactual records），用于后续训练。

### 2.3 FPLD（第一逻辑分歧点诊断）

对失败轨迹和锚点轨迹做结构化比较，定位第一处逻辑偏差：

- 在哪一步偏离；
- 偏离的是工具名还是参数；
- 给出可操作诊断文本。

本实现支持 dict 调用与字符串调用（如 `tool(a=1)`）两种格式，使用 AST 归一化后比较。

### 2.4 动态锚点选择（Dynamic Anchor Selection）

按优先级选择纠偏目标：

1. Peer Anchor（同批次成功样本，Stage3+）
2. Historical Anchor（历史自我成功样本，Stage3/4）
3. Induced Anchor（Stage2 课程诱导锚点）

当前实现已覆盖 Historical + Induced 的核心路径，Peer 保留接口位。

---

## 3. 四阶段课程学习（Stage1~4）

### Stage1：格式与语法
- 目标：输出合法工具调用格式；
- 行为：高重试、强提示。

### Stage2：真值拦截冷启动
- 目标：学会正确依赖和参数；
- 行为：先解码，再做 Ground Truth Interception，偏离即纠偏重试；
- 结果：沉淀课程诱导锚点与反事实记录。

### Stage3：自博弈内化
- 目标：减少对提示依赖；
- 行为：环境转标准模式，重试概率按课程进度线性退火（1.0→0.2）。

### Stage4：鲁棒性实战
- 目标：零提示条件下稳定完成任务；
- 行为：关闭重试。

---

## 4. 当前代码映射

- `env_tuning/seet/config.py`：SEET 配置与阶段控制。  
- `env_tuning/seet/fpld.py`：FPLD 诊断。  
- `env_tuning/seet/anchor.py`：锚点与回放池。  
- `env_tuning/seet/runtime.py`：重试策略、锚点选择、慢通道样本构造、Stage2 拦截逻辑。  
- `env_tuning/interaction/new_multi_turn_fc.py`：主交互编排（接入 Fast/Slow Loop）。  
- `env_tuning/interaction/turn_manager.py`：轮切换时导出 `seet_counterfactual_records`。  
- `env_tuning/config/multi_turn_fc_interaction_stage*.yaml`：分阶段行为配置。  

---

## 5. 为什么这个实现接近“完整复现”

1. **机制齐全**：环境调优、快慢通道、FPLD、动态锚点、Stage2 真值拦截全部具备。  
2. **训练可接入**：慢通道样本会随 turn extra 输出，便于直接接入 loss 构造。  
3. **课程可控**：每个 stage 的重试策略和环境模式配置化。  
4. **可读性优先**：核心算法在 `seet/`，交互层只做流程编排，函数职责清晰。

---

## 6. 后续可选增强（非必须）

- 在 trainer 侧直接消费 `seet_counterfactual_records`，把慢通道纳入正式训练损失；
- 增加 batch 级 Peer Anchor 检索器，补齐最高优先级锚点路径；
- 引入更细粒度的参数等价比较（例如顺序无关 dict/list 归一化）以进一步增强 FPLD 稳健性。
