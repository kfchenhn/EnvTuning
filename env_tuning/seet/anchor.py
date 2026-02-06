from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class AnchorTrace:
    """锚点轨迹（单轮）。"""

    entry_id: str
    turn_index: int
    decoded_calls: List[Any]
    anchor_type: str


@dataclass
class AnchorReplayBuffer:
    """
    历史成功轨迹回放池。

    中文注释：默认仅内存保存；可通过 save_to_file/load_from_file 做可选持久化。
    """

    traces: Dict[str, List[AnchorTrace]] = field(default_factory=dict)

    def push(self, trace: AnchorTrace) -> None:
        self.traces.setdefault(trace.entry_id, []).append(trace)

    def latest(self, entry_id: str, turn_index: int) -> Optional[AnchorTrace]:
        candidates = [x for x in self.traces.get(entry_id, []) if x.turn_index == turn_index]
        return candidates[-1] if candidates else None

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            entry_id: [
                {
                    "entry_id": t.entry_id,
                    "turn_index": t.turn_index,
                    "decoded_calls": t.decoded_calls,
                    "anchor_type": t.anchor_type,
                }
                for t in traces
            ]
            for entry_id, traces in self.traces.items()
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, List[Dict[str, Any]]]) -> "AnchorReplayBuffer":
        buffer = cls()
        if not isinstance(payload, dict):
            return buffer

        for entry_id, traces in payload.items():
            if not isinstance(traces, list):
                continue
            for t in traces:
                if not isinstance(t, dict):
                    continue
                buffer.push(
                    AnchorTrace(
                        entry_id=str(t.get("entry_id", entry_id)),
                        turn_index=int(t.get("turn_index", 0)),
                        decoded_calls=t.get("decoded_calls", []) if isinstance(t.get("decoded_calls", []), list) else [],
                        anchor_type=str(t.get("anchor_type", "standard")),
                    )
                )
        return buffer

    def save_to_file(self, file_path: str) -> None:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_from_file(cls, file_path: str) -> "AnchorReplayBuffer":
        p = Path(file_path)
        if not p.exists():
            return cls()
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        return cls.from_dict(payload)


class DynamicAnchorSelector:
    """按 SEET 优先级选择锚点：Peer > Historical > Induced。"""

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

        # Priority 3: Curriculum Induced Anchor (Stage 2)
        if stage == 2 and induced_anchor is not None:
            return induced_anchor

        return None
