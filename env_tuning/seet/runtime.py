import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .anchor import AnchorReplayBuffer, AnchorTrace, DynamicAnchorSelector
from .config import SeetConfig
from .fpld import FPLDResult, first_logic_divergence


@dataclass
class RetryDecision:
    should_retry: bool
    hint_text: str = ""
    anchor_calls: Optional[List[Any]] = None


class SeetRuntime:
    """SEET 运行时：快通道重试 + 锚点维护 + FPLD 诊断 + 慢通道记录构造。"""

    def __init__(self, config: SeetConfig):
        self.config = config
        self.replay_buffer = AnchorReplayBuffer()
        self.selector = DynamicAnchorSelector(self.replay_buffer)

    def _effective_retry_probability(self, turn_index: int = 0, total_turns: int = 1) -> float:
        """计算当前轮次有效重试概率。Stage3 使用线性退火，其余阶段使用固定概率。"""
        if self.config.stage != 3:
            return self.config.retry_probability

        if total_turns <= 1:
            return self.config.stage3_retry_end

        progress = min(max(turn_index / float(total_turns - 1), 0.0), 1.0)
        return self.config.stage3_retry_start + progress * (self.config.stage3_retry_end - self.config.stage3_retry_start)

    def should_retry(self, attempt_count: int, turn_index: int = 0, total_turns: int = 1) -> bool:
        """按概率和轮内次数判断是否允许重试。"""
        if attempt_count >= self.config.max_retry_per_turn:
            return False
        return random.random() <= self._effective_retry_probability(turn_index, total_turns)

    def on_success(self, entry_id: str, turn_index: int, decoded_calls: List[Any], anchor_type: str) -> None:
        """把成功轨迹注册为可复用锚点。"""
        self.replay_buffer.push(
            AnchorTrace(
                entry_id=entry_id,
                turn_index=turn_index,
                decoded_calls=decoded_calls,
                anchor_type=anchor_type,
            )
        )

    def choose_anchor_calls(
        self,
        stage: int,
        entry_id: str,
        turn_index: int,
        induced_calls: Optional[List[Any]] = None,
    ) -> Optional[List[Any]]:
        induced_anchor = (
            AnchorTrace(entry_id=entry_id, turn_index=turn_index, decoded_calls=induced_calls or [], anchor_type="induced")
            if induced_calls
            else None
        )
        chosen = self.selector.choose(
            stage=stage,
            entry_id=entry_id,
            turn_index=turn_index,
            peer_anchor=None,
            induced_anchor=induced_anchor,
        )
        return chosen.decoded_calls if chosen else None

    def build_retry_hint(
        self,
        stage: int,
        entry_id: str,
        turn_index: int,
        fail_calls: Optional[List[Any]],
        induced_calls: Optional[List[Any]] = None,
    ) -> RetryDecision:
        """基于锚点和 FPLD 生成快通道提示。"""
        anchor_calls = self.choose_anchor_calls(stage, entry_id, turn_index, induced_calls)
        if anchor_calls is None:
            return RetryDecision(False, "", None)

        if not fail_calls:
            return RetryDecision(
                should_retry=True,
                hint_text="系统提示（SEET）：你上一步没有形成有效工具调用，请根据任务目标和参数约束重新调用函数。",
                anchor_calls=anchor_calls,
            )

        fpld: FPLDResult = first_logic_divergence(fail_calls, anchor_calls)
        return RetryDecision(
            should_retry=True,
            hint_text=f"系统提示（SEET-FPLD）：{fpld.diagnosis}",
            anchor_calls=anchor_calls,
        )

    def build_counterfactual_record(self, fail_calls: List[Any], anchor_calls: List[Any]) -> Dict[str, Any]:
        """构造慢通道反事实训练记录。"""
        fpld = first_logic_divergence(fail_calls, anchor_calls)
        return {
            "fail_calls": fail_calls,
            "anchor_calls": anchor_calls,
            "divergence_index": fpld.divergence_index,
            "diagnosis": fpld.diagnosis,
        }

    def stage2_ground_truth_interception(self, decoded_calls: List[Any], ground_truth_calls: List[Any]) -> Optional[str]:
        """
        Stage2 真值拦截：若当前调用与真值不一致，返回纠偏提示。
        采用前缀一致性，允许模型逐步逼近。
        """
        if not decoded_calls:
            return "系统提示（SEET-Stage2）：当前没有有效函数调用，请先发起正确的工具调用。"

        compare_len = min(len(decoded_calls), len(ground_truth_calls))
        for i in range(compare_len):
            if decoded_calls[i] != ground_truth_calls[i]:
                return (
                    f"系统提示（SEET-Stage2 真值拦截）：在第 {i + 1} 个调用处偏离目标。"
                    f"你输出的是 `{decoded_calls[i]}`，建议参考正确调用 `{ground_truth_calls[i]}` 并重试。"
                )

        if len(decoded_calls) > len(ground_truth_calls):
            return "系统提示（SEET-Stage2 真值拦截）：当前调用数量超过该轮目标，请精简后重试。"

        return None
