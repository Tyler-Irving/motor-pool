"""Planner implementations behind the Planner protocol.

ScriptedPlanner replays pre-baked decisions for tests (no network, the agent
analog of httpx.MockTransport). LlmPlanner is the real impl: it renders the tool
catalog + history into a grounding-style prompt and parses a FLAT PlannerDecision
from the model via TypeAdapter, replicating inference.prompt.generate_grounded's
proven complete(json_mode=True) + extract_json + one corrective retry. It does
NOT use ChatClient.complete_json: that calls schema.model_validate, which a
discriminated-union alias like PlannerDecision does not have (the same reason
inference/prompt.py uses a TypeAdapter). On final parse failure it returns a
PlannerError so the loop degrades gracefully instead of crashing.

llm is imported lazily so importing this module (for ScriptedPlanner) does not
pull in ChatClient/httpx.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from ..schemas import AgentStep, PlannerDecision, PlannerError, ToolSpec

if TYPE_CHECKING:
    from ..config import AgentConfig
    from ..llm import ChatClient
    from .registry import ToolRegistry

_PLANNER_ADAPTER = TypeAdapter(PlannerDecision)


class ScriptedPlanner:
    """Replays a fixed list of decisions, one per decide() call. The test driver."""

    def __init__(self, script: list[PlannerDecision]) -> None:
        self._script = list(script)
        self._i = 0

    def decide(self, question, history, registry) -> PlannerDecision:
        if self._i >= len(self._script):
            raise AssertionError(
                "ScriptedPlanner exhausted: more decide() calls than scripted decisions"
            )
        decision = self._script[self._i]
        self._i += 1
        return decision


_SYSTEM_PROMPT = (
    "You are a diagnostic agent for the HMMWV operator technical manual. You answer "
    "by calling tools and then finishing with a grounded, cited answer. Use ONLY "
    "facts returned by the tools; if the tools do not cover the question, finish "
    "with a refusal. Respond with ONE JSON object and nothing else."
)

_DECISION_SPEC = (
    "Respond with ONE JSON object, exactly one of:\n"
    'Call a tool: {"kind":"call_tool","tool":"<name>","args":{...},"rationale":"<why>"}\n'
    'Finish (answer): {"kind":"finish","answer":{"answer_type":"answer",'
    '"summary":"<one sentence>","steps":[{"order":1,"text":"<step>","cited":[1]}],"cited":[1]}}\n'
    'Finish (refusal): {"kind":"finish","answer":{"answer_type":"refusal",'
    '"reason":"not_in_corpus|out_of_scope|insufficient_context|ambiguous","message":"<why>"}}\n'
    "The cited arrays hold the [C#] numbers from the most recent retrieve result."
)


def _render_specs(specs: list[ToolSpec]) -> str:
    lines = []
    for spec in specs:
        props = ", ".join((spec.args_schema.get("properties") or {}).keys())
        lines.append(f"- {spec.name}({props}): {spec.description}")
    return "\n".join(lines)


def _render_history(history: list[AgentStep]) -> str:
    if not history:
        return "(no tool calls yet)"
    lines = []
    for step in history:
        decision = step.decision
        if decision.kind == "call_tool":
            result = step.result
            status = "ok" if (result and result.ok) else "error"
            body = (result.content if result else "")[:700]
            lines.append(
                f"[{step.index}] called {decision.tool}({json.dumps(decision.args)}) "
                f"-> {status}\n{body}"
            )
        elif decision.kind == "error":
            lines.append(f"[{step.index}] planner error: {decision.message}")
    return "\n".join(lines)


class LlmPlanner:
    """The real planner: one model turn per decide(), parsed into a PlannerDecision."""

    def __init__(
        self,
        client: "ChatClient",
        *,
        system_prompt: str = _SYSTEM_PROMPT,
        retries: int = 1,
    ) -> None:
        self._client = client
        self._system = system_prompt
        self._retries = retries

    @classmethod
    def from_config(cls, cfg: "AgentConfig") -> "LlmPlanner":
        from ..llm import ChatClient

        return cls(ChatClient.from_config(cfg.planner))

    def decide(self, question, history, registry: "ToolRegistry") -> PlannerDecision:
        from ..llm import extract_json

        user = (
            f"QUESTION: {question}\n\n"
            f"TOOLS:\n{_render_specs(registry.specs())}\n\n"
            f"HISTORY:\n{_render_history(history)}\n\n"
            f"{_DECISION_SPEC}"
        )
        prompt = user
        last = ""
        for _ in range(self._retries + 1):
            text = self._client.complete(self._system, prompt, json_mode=True)
            try:
                return _PLANNER_ADAPTER.validate_python(extract_json(text))
            except (ValidationError, ValueError) as exc:
                last = str(exc)[:150]
                prompt = (
                    f"{user}\n\nReturn ONLY one valid decision JSON object. "
                    f"Your last reply failed: {last}"
                )
        return PlannerError(message=f"planner did not return a valid decision: {last}")
