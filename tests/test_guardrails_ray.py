"""
Ensure Ray tasks can be used as input and output guardrail functions with the default backend.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

try:
    import ray
except Exception:  # pragma: no cover - ray is optional
    ray = None  # type: ignore[assignment]

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
)

from .fake_model import FakeModel
from .test_responses import get_text_message

RUN_RAY_GUARDRAIL_TESTS = os.environ.get("RUN_RAY_GUARDRAIL_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    (ray is None) or (not RUN_RAY_GUARDRAIL_TESTS),
    reason="Ray guardrail tests require RUN_RAY_GUARDRAIL_TESTS=1 and ray installed.",
)


@pytest.fixture(scope="module", autouse=True)
def _ensure_local_ray_runtime():
    """
    These tests intentionally exercise Ray tasks while the Agents SDK still uses the
    default (non-Ray) backend. Rather than spawning separate worker processes (which
    requires packaging the repo and replicating the virtualenv), we rely on Ray's
    local_mode so that "remote" tasks execute in-process. This keeps the environment
    identical to the test runner and avoids hanging workers.
    """

    if ray is None:
        yield
        return

    original_runtime_env_dir = os.environ.pop("RAY_RUNTIME_ENV_WORKING_DIR", None)
    already_running = ray.is_initialized()
    if not already_running:
        working_dir_uri = Path(__file__).resolve().parents[1].as_uri()
        ray.init(
            local_mode=True,
            include_dashboard=False,
            runtime_env={"working_dir": working_dir_uri},
        )

    try:
        yield
    finally:
        if not already_running and ray.is_initialized():
            ray.shutdown()
        if original_runtime_env_dir is not None:
            os.environ["RAY_RUNTIME_ENV_WORKING_DIR"] = original_runtime_env_dir


@pytest.mark.asyncio
async def test_ray_task_as_input_guardrail_function_works_with_default_backend():
    """
    Verifies that an input guardrail can invoke a ray task and works under the default backend
    (i.e., without enabling the Ray backend for guardrail orchestration).
    """

    @ray.remote
    def input_guardrail_remote_check(text: str) -> dict[str, Any]:
        should_block = "BLOCK" in text
        return {
            "tripwire_triggered": should_block,
            "output_info": "ray_input_check",
        }

    @input_guardrail(run_in_parallel=False)
    async def ray_wrapped_input_guardrail(
        ctx: RunContextWrapper[Any], agent: Agent[Any], input: str | list[TResponseInputItem]
    ) -> GuardrailFunctionOutput:
        if isinstance(input, str):
            text = input
        else:
            # Concatenate any text contents present in list input form.
            parts: list[str] = []
            for item in input:
                if isinstance(item, dict):
                    parts.append(str(item.get("content", "")))
            text = " ".join(parts)

        ref = input_guardrail_remote_check.remote(text)
        result = await asyncio.to_thread(ray.get, ref)
        return GuardrailFunctionOutput(
            output_info=result["output_info"],
            tripwire_triggered=bool(result["tripwire_triggered"]),
        )

    model = FakeModel()
    agent = Agent(
        name="agent_with_ray_input_guardrail",
        instructions="Reply with 'ok'.",
        input_guardrails=[ray_wrapped_input_guardrail],
        model=model,
    )
    # Case 1: Guardrail trips and prevents model call.
    model.set_next_output([get_text_message("ok")])
    with pytest.raises(InputGuardrailTripwireTriggered):
        await Runner.run(agent, "please BLOCK this request")
    assert model.first_turn_args is None, "Model should not have been called when guardrail trips"

    # Case 2: Guardrail passes and model is called.
    model = FakeModel()
    agent = Agent(
        name="agent_with_ray_input_guardrail",
        instructions="Reply with 'ok'.",
        input_guardrails=[ray_wrapped_input_guardrail],
        model=model,
    )
    model.set_next_output([get_text_message("ok")])
    result = await Runner.run(agent, "hello there")
    assert result.final_output is not None
    assert model.first_turn_args is not None, "Model should have been called when guardrail passes"


@pytest.mark.asyncio
async def test_ray_task_as_output_guardrail_function_works_with_default_backend():
    """
    Verifies that an output guardrail can invoke a ray task and works under the default backend
    (i.e., without enabling the Ray backend for guardrail orchestration).
    """

    @ray.remote
    def output_guardrail_remote_check(text: str) -> dict[str, Any]:
        should_block = "BAD" in text
        return {
            "tripwire_triggered": should_block,
            "output_info": "ray_output_check",
        }

    @output_guardrail(name="ray_output_guardrail")
    async def ray_wrapped_output_guardrail(
        ctx: RunContextWrapper[Any], agent: Agent[Any], output: Any
    ) -> GuardrailFunctionOutput:
        text = str(output)
        ref = output_guardrail_remote_check.remote(text)
        result = await asyncio.to_thread(ray.get, ref)
        return GuardrailFunctionOutput(
            output_info=result["output_info"],
            tripwire_triggered=bool(result["tripwire_triggered"]),
        )

    model = FakeModel()
    agent = Agent(
        name="agent_with_ray_output_guardrail",
        instructions="Reply exactly with the provided keywords.",
        output_guardrails=[ray_wrapped_output_guardrail],
        model=model,
    )
    # Case 1: Guardrail trips after model generation.
    model.set_next_output([get_text_message("this is BAD output")])
    with pytest.raises(OutputGuardrailTripwireTriggered) as excinfo:
        await Runner.run(agent, "produce BAD")
    assert excinfo.value.guardrail_result.output.tripwire_triggered is True
    assert model.first_turn_args is not None, (
        "Model should have been called before output guardrail trips"
    )

    # Case 2: Guardrail passes and final output is returned.
    model = FakeModel()
    agent = Agent(
        name="agent_with_ray_output_guardrail",
        instructions="Reply exactly with the provided keywords.",
        output_guardrails=[ray_wrapped_output_guardrail],
        model=model,
    )
    model.set_next_output([get_text_message("all good")])
    result = await Runner.run(agent, "ok")
    assert result.final_output is not None
    assert model.first_turn_args is not None, "Model should have been called when guardrail passes"
