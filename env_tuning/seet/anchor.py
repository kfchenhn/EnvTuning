from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AnchorTrace:
    entry_id: str
    turn_index: int
    decoded_calls: List[Any]
    anchor_type: str


@dataclass
class AnchorReplayBuffer:
    traces: Dict[str, List[AnchorTrace]] = field(default_factory=dict)

    def push(self, trace: AnchorTrace) -> None:
        self.traces.setdefault(trace.entry_id, []).append(trace)

    def latest(self, entry_id: str, turn_index: int) -> Optional[AnchorTrace]:
        candidates = [x for x in self.traces.get(entry_id, []) if x.turn_index == turn_index]
        if not candidates:
            return None
        return candidates[-1]


class DynamicAnchorSelector:
    """按 SEET 优先级选择锚点。"""

    def __init__(self, replay_buffer: AnchorReplayBuffer):
        self.replay_buffer = replay_buffer

    def choose(
        self,
        stage: int,
        entry_id: str,
        turn_index: int,
        peer_anchor: Optional[AnchorTrace] = None,
        induced_anchor: Optional[AnchorTrace] = None,
    ) -> Optional[AnchorTrace]:
        # Priority 1: Peer Anchor (Stage 3+)
        if stage >= 3 and peer_anchor is not None:
            return peer_anchor

        # Priority 2: Historical Self Anchor (Stage 3/4)
        if stage >= 3:
            historical = self.replay_buffer.latest(entry_id=entry_id, turn_index=turn_index)
            if historical is not None:
                return historical

        # Priority 3: Curriculum induced anchor (Stage 2)
        if stage == 2 and induced_anchor is not None:
            return induced_anchor

        return None
