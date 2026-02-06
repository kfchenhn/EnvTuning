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
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4

from verl.interactions.base import BaseInteraction
from bfcl_env.multi_turn_utils import execute_multi_turn_func_call

from .data_models import InstanceState, ResponseData, ResponseType, ExecutionResult
from .response_handler import ResponseHandler
from .execution_manager import ExecutionManager
from .score_calculator import ScoreCalculator
from .turn_manager import TurnManager
from env_tuning.seet import SeetConfig, SeetRuntime


class MultiTurnFunctionCallInteraction(BaseInteraction):
    """多轮函数调用交互主类（集成 SEET 快慢双通道）。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = config.get("name", "multi_turn_function_call")
        self._instance_dict: Dict[str, InstanceState] = {}
        self.max_step_limit = 5

        # SEET 运行时（可选）
        self.seet_config = SeetConfig(**config.get("seet", {}))
        self.seet_runtime = SeetRuntime(self.seet_config) if self.seet_config.enabled else None

        self.response_handler = ResponseHandler()
        self.execution_manager = ExecutionManager()
        self.score_calculator = ScoreCalculator()
        self.turn_manager = TurnManager(self.score_calculator)

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """创建交互实例。"""
        if instance_id is None:
            instance_id = str(uuid4())

        entry_id: str = kwargs["id"]
        initial_config: Dict[str, Any] = json.loads(kwargs["initial_config"])
        involved_classes: Dict[str, Any] = kwargs["involved_classes"]
        ground_truth: List[Any] = kwargs["ground_truth"]
        processed_question: List[str] = kwargs["processed_question"]
        question: List[str] = kwargs["question"]

        _, model_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            instance_id,
            entry_id,
            long_context=("long_context" in entry_id or "composite" in entry_id),
            is_evaL_run=False,
        )

        execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            instance_id + "_ground_truth",
            entry_id,
            long_context=("long_context" in entry_id or "composite" in entry_id),
            is_evaL_run=True,
        )

        self._instance_dict[instance_id] = InstanceState(
            initial_config=initial_config,
            involved_classes=involved_classes,
            ground_truth=ground_truth,
            processed_question=processed_question,
            question=question,
            involved_instances=model_instances,
            total_turns=len(question),
        )
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """生成交互响应。"""
        state = self._instance_dict[instance_id]
        entry_id = kwargs["id"]

        response_data = self.response_handler.parse_and_validate(messages)
        if response_data.has_error:
            return await self._handle_response_error(instance_id, response_data, state, entry_id)

        special_case_result = self._handle_special_cases(response_data, state, entry_id)
        if special_case_result:
            return special_case_result

        predecoded_calls: Optional[List[Any]] = None
        if response_data.response_type == ResponseType.TOOL_CALL:
            predecoded_calls = self.execution_manager.decode_tool_calls(response_data.content)
            stage2_intercept = self._maybe_stage2_intercept(state, predecoded_calls)
            if stage2_intercept is not None:
                return stage2_intercept

        execution_result = self._execute_function_calls(
            response_data,
            state,
            instance_id,
            entry_id,
            predecoded_calls,
        )
        return self._determine_next_action(execution_result, state, entry_id)

    async def _handle_response_error(
        self,
        instance_id: str,
        response_data: ResponseData,
        state: InstanceState,
        entry_id: str,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """处理解析错误，并在 SEET 下尝试快通道重试。"""
        state.current_turn_attempt_counts += 1

        if self.turn_manager.should_force_quit(state, self.max_step_limit):
            should_term, content, score, extra = self.turn_manager.advance_to_next_turn(state, entry_id)
            if should_term:
                await self.finalize_interaction(instance_id=instance_id)

            prev_gt = self.turn_manager._get_ground_truth_calls(state, state.current_turn_index - 1)
            if not prev_gt:
                score = 0.0
            return should_term, content, score, extra

        if self.seet_runtime and self.seet_runtime.should_retry(
            state.current_turn_attempt_counts,
            turn_index=state.current_turn_index,
            total_turns=state.total_turns,
        ):
            retry = self.seet_runtime.build_retry_hint(
                stage=self.seet_config.stage,
                entry_id=entry_id,
                turn_index=state.current_turn_index,
                fail_calls=[],
                induced_calls=self._get_current_turn_ground_truth(state),
            )
            if retry.should_retry:
                return False, retry.hint_text, -1.0, {"seet_fast_loop": True, "channel": "fast"}

        return False, response_data.error_message or "Parse error", -3.0, {}

    def _handle_special_cases(
        self,
        response_data: ResponseData,
        state: InstanceState,
        entry_id: str,
    ) -> Optional[Tuple[bool, str, float, Dict[str, Any]]]:
        """处理无 GT 的轮次。"""
        ground_truth_calls = self._get_current_turn_ground_truth(state)
        if ground_truth_calls:
            return None

        should_term, content, base_score, extra = self.turn_manager.advance_to_next_turn(state, entry_id)
        assert base_score == -1.0

        if response_data.response_type == ResponseType.ANSWER:
            return should_term, content, 1.0, extra

        warning_hint = (
            "(SYSTEM WARNING: You should not call any function in this turn because certain function "
            "description(s) or parameter(s) is missing in this turn. Previous turn is forced quit. "
            "Current function(s) will not be executed.) Next turn question:\n"
        )
        return should_term, warning_hint + content, 0.0, extra

    def _maybe_stage2_intercept(
        self,
        state: InstanceState,
        decoded_calls: List[Any],
    ) -> Optional[Tuple[bool, str, float, Dict[str, Any]]]:
        """Stage2 真值拦截：偏离即提示重试，不执行错误调用。"""
        if not self.seet_runtime or self.seet_config.stage != 2 or not self.seet_config.enable_stage2_interception:
            return None

        gt_calls = self._get_current_turn_ground_truth(state)
        if not gt_calls:
            return None

        hint = self.seet_runtime.stage2_ground_truth_interception(decoded_calls, gt_calls)
        if hint is None:
            return None

        # 记录慢通道样本（反事实：失败调用 -> GT 调用）
        state.seet_counterfactual_records.append(
            self.seet_runtime.build_counterfactual_record(decoded_calls, gt_calls)
        )
        state.current_turn_attempt_counts += 1
        return False, hint, -1.0, {"seet_fast_loop": True, "channel": "fast", "reason": "stage2_interception"}

    def _execute_function_calls(
        self,
        response_data: ResponseData,
        state: InstanceState,
        instance_id: str,
        entry_id: str,
        predecoded_calls: Optional[List[Any]] = None,
    ) -> ExecutionResult:
        """执行函数调用。"""
        return self.execution_manager.execute_function_calls(
            response_data.content,
            state,
            instance_id,
            entry_id,
            predecoded_responses=predecoded_calls,
        )

    def _determine_next_action(
        self,
        execution_result: ExecutionResult,
        state: InstanceState,
        entry_id: str,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """根据执行结果决定是否继续。"""
        if not execution_result.should_continue:
            return self.turn_manager.advance_to_next_turn(state, entry_id)

        state.involved_instances = execution_result.new_instances
        state.add_exec_results(execution_result.execution_results)
        state.current_turn_attempt_counts += 1

        if self.turn_manager.should_force_quit(state, self.max_step_limit):
            return self.turn_manager.advance_to_next_turn(state, entry_id)

        user_hint, score = self.execution_manager.format_execution_response(
            execution_result.execution_results,
            execution_result.has_error,
            stage=self.seet_config.stage if self.seet_config.enabled else None,
            augmented_env=self.seet_config.use_augmented_env if self.seet_config.enabled else False,
        )

        self._register_success_anchor_if_needed(state, entry_id, execution_result)

        if self.seet_runtime and execution_result.has_error and self.seet_runtime.should_retry(
            state.current_turn_attempt_counts,
            turn_index=state.current_turn_index,
            total_turns=state.total_turns,
        ):
            retry = self.seet_runtime.build_retry_hint(
                stage=self.seet_config.stage,
                entry_id=entry_id,
                turn_index=state.current_turn_index,
                fail_calls=execution_result.decoded_responses or [],
                induced_calls=self._get_current_turn_ground_truth(state),
            )
            if retry.should_retry:
                if retry.anchor_calls is not None:
                    state.seet_counterfactual_records.append(
                        self.seet_runtime.build_counterfactual_record(
                            execution_result.decoded_responses or [],
                            retry.anchor_calls,
                        )
                    )
                return (
                    False,
                    user_hint + "\n\n" + retry.hint_text,
                    min(score, -1.0),
                    {"seet_fast_loop": True, "channel": "fast", "reason": "execution_error"},
                )

        return False, user_hint, score, {}

    def _register_success_anchor_if_needed(self, state: InstanceState, entry_id: str, execution_result: ExecutionResult) -> None:
        if not self.seet_runtime or execution_result.has_error:
            return

        self.seet_runtime.on_success(
            entry_id=entry_id,
            turn_index=state.current_turn_index,
            decoded_calls=execution_result.decoded_responses or [],
            anchor_type="standard" if self.seet_config.stage >= 3 else "induced",
        )

    def _get_current_turn_ground_truth(self, state: InstanceState) -> List[Any]:
        return self.turn_manager._get_ground_truth_calls(state, state.current_turn_index)

    async def calculate_score(self) -> float:
        return 0.0

    async def finalize_interaction(self, instance_id: str = None, **kwargs) -> None:
        if instance_id and instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
