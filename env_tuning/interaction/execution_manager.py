# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import List, Any, Tuple, Optional
from .data_models import InstanceState, ExecutionResult
from .utils import (
    parse_tool_calls,
    default_decode_execute_prompting,
    is_empty_execute_response,
    has_execution_error,
)
from bfcl_env.multi_turn_utils import execute_multi_turn_func_call


class ExecutionManager:
    """管理函数执行相关逻辑。"""

    def execute_function_calls(
        self,
        tool_content: str,
        state: InstanceState,
        instance_id: str,
        entry_id: str,
        predecoded_responses: Optional[List[Any]] = None,
    ) -> ExecutionResult:
        """执行函数调用。"""
        try:
            decoded_responses = predecoded_responses
            if decoded_responses is None:
                model_responses = parse_tool_calls(tool_content)
                decoded_responses = default_decode_execute_prompting(model_responses)

            if is_empty_execute_response(decoded_responses):
                return ExecutionResult(
                    execution_results=[],
                    new_instances=state.involved_instances,
                    has_error=False,
                    should_continue=False,
                    decoded_responses=decoded_responses,
                )

            state.single_turn_model_response_decode_list.append(decoded_responses)

            execution_results, new_instances = execute_multi_turn_func_call(
                decoded_responses,
                state.initial_config,
                state.involved_classes,
                instance_id,
                entry_id,
                long_context=("long_context" in entry_id or "composite" in entry_id),
                is_evaL_run=False,
            )

            has_error = has_execution_error(execution_results)

            return ExecutionResult(
                execution_results=execution_results,
                new_instances=new_instances,
                has_error=has_error,
                should_continue=True,
                decoded_responses=decoded_responses,
            )

        except Exception:
            return ExecutionResult(
                execution_results=[],
                new_instances=state.involved_instances,
                has_error=True,
                should_continue=False,
                decoded_responses=predecoded_responses,
            )

    def format_execution_response(
        self,
        execution_results: List[Any],
        has_error: bool,
        stage: int = None,
        augmented_env: bool = False,
    ) -> Tuple[str, float]:
        """格式化执行结果响应。"""
        response_content = json.dumps(execution_results, ensure_ascii=False)
        score = -2.0 if has_error else -1.0
        stage_text = f" Current SEET stage: {stage}." if stage is not None else ""
        env_text = " Environment mode: augmented." if augmented_env else " Environment mode: standard."

        user_hint = (
            f"Here are the function's execution results. Execution results:{response_content}\n"
            f"{stage_text}{env_text} "
            f"If you believe you have already fulfilled the user's request, please first outline "
            f"your thought process in a <think></think>pair, and then give a brief summary of the "
            f"result in an <answer></answer> pair. Otherwise, you should continue to call until "
            f"fulfilling user's request."
        )
        return user_hint, score

    def decode_tool_calls(self, tool_content: str) -> List[Any]:
        """仅做工具调用解码，不执行。"""
        try:
            model_responses = parse_tool_calls(tool_content)
            return default_decode_execute_prompting(model_responses)
        except Exception:
            return []

    def check_execution_limits(self, state: InstanceState, max_limit: int) -> bool:
        """检查执行次数是否超过限制。"""
        return state.current_turn_attempt_counts > max_limit
