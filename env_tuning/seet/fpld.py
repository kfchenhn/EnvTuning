from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import ast


@dataclass
class ToolNode:
    """标准化后的工具调用节点。"""

    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class FPLDResult:
    """第一逻辑分歧诊断结果。"""

    divergence_index: Optional[int]
    diagnosis: str


def _safe_eval_node(node: ast.AST) -> Any:
    """尽可能把 AST 节点转回 Python 值。"""
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return str(node)


def _parse_call_string(call_str: str) -> ToolNode:
    """将形如 `foo(a=1, b='x')` 的字符串解析为 ToolNode。"""
    try:
        expr = ast.parse(call_str.strip(), mode="eval")
        if not isinstance(expr.body, ast.Call):
            return ToolNode(tool_name="<invalid>", arguments={})

        func = expr.body.func
        if isinstance(func, ast.Name):
            tool_name = func.id
        elif isinstance(func, ast.Attribute):
            tool_name = ast.unparse(func)
        else:
            tool_name = "<invalid>"

        kwargs: Dict[str, Any] = {}
        for kw in expr.body.keywords:
            if kw.arg is None:
                continue
            kwargs[kw.arg] = _safe_eval_node(kw.value)

        return ToolNode(tool_name=tool_name, arguments=kwargs)
    except Exception:
        return ToolNode(tool_name="<invalid>", arguments={})


def _normalize_step(step: Any) -> ToolNode:
    """
    兼容两种调用表示：
    1) 结构化 dict: {tool_name: {arg: value}}
    2) 字符串: "tool_name(a=1)"
    """
    if isinstance(step, str):
        return _parse_call_string(step)

    if isinstance(step, dict) and len(step) == 1:
        tool_name = next(iter(step.keys()))
        args = step.get(tool_name, {})
        if not isinstance(args, dict):
            args = {}
        return ToolNode(tool_name=tool_name, arguments=args)

    return ToolNode(tool_name="<invalid>", arguments={})


def first_logic_divergence(fail_path: List[Any], anchor_path: List[Any]) -> FPLDResult:
    """定位第一逻辑分歧点（FPLD）。"""
    # 中文说明：按最短轨迹对齐比较，优先找到“第一处”可定位偏差。
    shortest = min(len(fail_path), len(anchor_path))

    for idx in range(shortest):
        fail_node = _normalize_step(fail_path[idx])
        anchor_node = _normalize_step(anchor_path[idx])

        if fail_node.tool_name != anchor_node.tool_name or fail_node.arguments != anchor_node.arguments:
            diagnosis = (
                f"The first logic mismatch appears at step {idx + 1}. "
                f"You called {fail_node.tool_name}({fail_node.arguments}), "
                f"but the anchor trajectory uses {anchor_node.tool_name}({anchor_node.arguments})."
            )
            return FPLDResult(divergence_index=idx, diagnosis=diagnosis)

    if len(fail_path) != len(anchor_path):
        idx = shortest
        diagnosis = (
            f"The trajectories start to differ in length at step {idx + 1}: "
            f"fail trace has {len(fail_path)} step(s), while the anchor trace has {len(anchor_path)} step(s)."
        )
        return FPLDResult(divergence_index=idx, diagnosis=diagnosis)

    return FPLDResult(divergence_index=None, diagnosis="No logic mismatch was detected between the two trajectories.")
