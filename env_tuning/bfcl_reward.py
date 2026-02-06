from typing import Any, Dict, List


def _extract_seet_counterfactual_count(reward_scores: Dict[str, Any]) -> int:
    """从 rollout 奖励字典中提取 SEET 慢通道反事实样本数量。"""
    metrics_per_turn = reward_scores.get("interaction_turn_metrics", [])
    if not isinstance(metrics_per_turn, list):
        return 0

    total = 0
    for turn_metrics in metrics_per_turn:
        if not isinstance(turn_metrics, dict):
            continue
        records = turn_metrics.get("seet_counterfactual_records", [])
        if isinstance(records, list):
            total += len(records)
    return total


def compute_score(
    reward_scores: Dict[str, List[float]],
    ground_truth: List[List],
    extra_info=None,
    seet_slow_loop_coef: float = 0.05,
    seet_slow_loop_cap: float = 0.3,
    **kwargs,
) -> Dict[str, Any]:
    """
    BFCL 奖励函数（含 SEET 慢通道加成）。

    - 基础 score 使用任务 progress；
    - 当产生 slow-loop 反事实记录时，给予小幅奖励加成，
      让慢通道数据对最终优化目标产生直接影响。
    """

    # ---------------- 基础信息 ----------------
    user_turn_rewards = reward_scores.get("user_turn_rewards", [])
    total_interaction_rounds = len(user_turn_rewards)
    total_tool_rounds = user_turn_rewards.count(-1)

    relevant = [s for s in user_turn_rewards if s == 0 or s == 1]
    progress = sum(relevant) / len(relevant) if relevant else 0.0

    # ----------------- 工具轮指标 -----------------
    golden_tool_rounds = len(ground_truth or [])
    tool_round_diff = total_tool_rounds - golden_tool_rounds
    tool_rel_diff = abs(tool_round_diff) / max(1, golden_tool_rounds)

    correct_tool_call = user_turn_rewards.count(-1)
    error_tool_call = user_turn_rewards.count(-2)
    error_format = user_turn_rewards.count(-3)
    is_tool_call = 1.0 if (correct_tool_call + error_tool_call) > 0 else 0.0

    format_reward = ((total_interaction_rounds - error_format) / total_interaction_rounds) if total_interaction_rounds > 0 else 0.0
    tool_call_reward = correct_tool_call / (correct_tool_call + error_tool_call) if is_tool_call > 0 else 0.0

    # ----------------- SEET 慢通道加成 -----------------
    seet_counterfactual_count = _extract_seet_counterfactual_count(reward_scores)
    seet_slow_loop_bonus = min(seet_slow_loop_cap, seet_counterfactual_count * seet_slow_loop_coef)

    # 让 slow-loop 直接参与 loss：通过 score 影响最终 reward tensor
    final_score = min(1.0, progress + seet_slow_loop_bonus)

    return {
        "score": final_score,
        "progress": progress,
        "seet_slow_loop_bonus": seet_slow_loop_bonus,
        "seet_counterfactual_count": seet_counterfactual_count,
        "total_interaction_rounds": total_interaction_rounds,
        "format_reward": format_reward,
        "tool_call_reward": tool_call_reward,
        "is_tool_call": is_tool_call,
        "tool_round_diff": tool_round_diff,
        "tool_rel_diff": tool_rel_diff,
    }
