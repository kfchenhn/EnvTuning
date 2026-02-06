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

    # 中文注释：该类是 SEET 的策略中枢，负责把课程配置转成可执行控制逻辑。

    def __init__(self, config: SeetConfig):
        self.config = config
        # 中文注释：默认使用内存回放池；若配置了路径则在启动时尝试加载历史锚点。
        if self.config.replay_buffer_path:
            self.replay_buffer = AnchorReplayBuffer.load_from_file(self.config.replay_buffer_path)
        else:
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

        if self.config.persist_replay_buffer_on_update and self.config.replay_buffer_path:
            self.replay_buffer.save_to_file(self.config.replay_buffer_path)

    # 中文注释：按 Stage 策略选择锚点轨迹，作为失败样本的纠偏参考。
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
                hint_text="[SEET] I could not find a valid tool call in the previous step. Please retry with a function call that matches the task goal and argument constraints.",
                anchor_calls=anchor_calls,
            )

        fpld: FPLDResult = first_logic_divergence(fail_calls, anchor_calls)
        return RetryDecision(
            should_retry=True,
            hint_text=f"[SEET-FPLD] {fpld.diagnosis}",
            anchor_calls=anchor_calls,
        )

    # 中文注释：Slow Loop 核心数据结构，后续会被奖励函数或训练器消费。
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
            return "[SEET-Stage2] No valid function call was detected. Please start by issuing the expected tool call."

        compare_len = min(len(decoded_calls), len(ground_truth_calls))
        for i in range(compare_len):
            if decoded_calls[i] != ground_truth_calls[i]:
                return (
                    f"[SEET-Stage2 Interception] The trajectory diverges at call #{i + 1}. "
                    f"You produced `{decoded_calls[i]}`, while the expected call is `{ground_truth_calls[i]}`. "
                    f"Please adjust and try again."
                )

        if len(decoded_calls) > len(ground_truth_calls):
            return "[SEET-Stage2 Interception] You produced more calls than expected for this turn. Please keep only the required calls and retry."

        return None
