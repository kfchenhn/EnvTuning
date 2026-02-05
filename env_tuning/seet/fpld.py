from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ToolNode:
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class FPLDResult:
    divergence_index: Optional[int]
    diagnosis: str


def _normalize_step(step: Any) -> ToolNode:
    if not isinstance(step, dict) or len(step) != 1:
        return ToolNode(tool_name="<invalid>", arguments={})
    tool_name = next(iter(step.keys()))
    args = step.get(tool_name, {})
    if not isinstance(args, dict):
        args = {}
    return ToolNode(tool_name=tool_name, arguments=args)


def first_logic_divergence(fail_path: List[Any], anchor_path: List[Any]) -> FPLDResult:
    """定位第一逻辑分歧点（FPLD）。"""
    shortest = min(len(fail_path), len(anchor_path))
    for idx in range(shortest):
        fail_node = _normalize_step(fail_path[idx])
        anchor_node = _normalize_step(anchor_path[idx])
        if fail_node.tool_name != anchor_node.tool_name or fail_node.arguments != anchor_node.arguments:
            diagnosis = (
                f"在第 {idx + 1} 步出现第一逻辑分歧："
                f"你执行了 {fail_node.tool_name}({fail_node.arguments})，"
                f"锚点轨迹执行的是 {anchor_node.tool_name}({anchor_node.arguments})。"
            )
            return FPLDResult(divergence_index=idx, diagnosis=diagnosis)

    if len(fail_path) != len(anchor_path):
        idx = shortest
        diagnosis = (
            f"在第 {idx + 1} 步出现长度分歧：失败轨迹长度={len(fail_path)}，"
            f"锚点轨迹长度={len(anchor_path)}。"
        )
        return FPLDResult(divergence_index=idx, diagnosis=diagnosis)

    return FPLDResult(divergence_index=None, diagnosis="未检测到逻辑分歧。")
