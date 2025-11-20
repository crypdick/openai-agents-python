"""
Ensure Ray tasks can be used as input and output guardrail functions with the default backend.
"""

from __future__ import annotations

import asyncio
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

pytestmark = pytest.mark.skipif(ray is None, reason="ray is not installed")


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
