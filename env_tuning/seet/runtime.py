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


class SeetRuntime:
    """SEET 运行时：双通道提示注入 + 锚点维护 + FPLD。"""

    def __init__(self, config: SeetConfig):
        self.config = config
        self.replay_buffer = AnchorReplayBuffer()
        self.selector = DynamicAnchorSelector(self.replay_buffer)

    def should_retry(self, attempt_count: int) -> bool:
        if attempt_count >= self.config.max_retry_per_turn:
            return False
        return random.random() <= self.config.retry_probability

    def on_success(
        self,
        entry_id: str,
        turn_index: int,
        decoded_calls: List[Any],
        anchor_type: str,
    ) -> None:
        self.replay_buffer.push(
            AnchorTrace(
                entry_id=entry_id,
                turn_index=turn_index,
                decoded_calls=decoded_calls,
                anchor_type=anchor_type,
            )
        )

    def build_retry_hint(
        self,
        stage: int,
        entry_id: str,
        turn_index: int,
        fail_calls: Optional[List[Any]],
        induced_calls: Optional[List[Any]] = None,
    ) -> RetryDecision:
        anchor = self.selector.choose(
            stage=stage,
            entry_id=entry_id,
            turn_index=turn_index,
            peer_anchor=None,
            induced_anchor=AnchorTrace(entry_id, turn_index, induced_calls or [], "induced") if induced_calls else None,
        )
        if anchor is None:
            return RetryDecision(False, "")

        if not fail_calls:
            return RetryDecision(True, "系统提示：你上一步没有形成有效工具调用，请重试并严格遵循参数约束。")

        fpld: FPLDResult = first_logic_divergence(fail_calls, anchor.decoded_calls)
        return RetryDecision(True, f"系统提示（SEET-FPLD）：{fpld.diagnosis}")

    def build_counterfactual_record(
        self,
        fail_calls: List[Any],
        anchor_calls: List[Any],
    ) -> Dict[str, Any]:
        fpld = first_logic_divergence(fail_calls, anchor_calls)
        return {
            "fail_calls": fail_calls,
            "anchor_calls": anchor_calls,
            "divergence_index": fpld.divergence_index,
            "diagnosis": fpld.diagnosis,
        }
