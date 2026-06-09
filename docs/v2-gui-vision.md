# V2 GUI vision (way down the line)

This is a forward design note, not built in V2.0. No web framework, no async, no
streaming, and no new dependencies are added now. The point is to record the seam
so a GUI can be built later without reworking the agent.

## One serializable contract is the API

`run_agent(...)` is a pure function returning `AgentResult`, a fully JSON-dumpable
pydantic trace (`src/motor_pool/schemas/agent.py`). `AgentResult.model_dump_json()`,
already emitted by the CLI `motor-pool agent --json` flag, IS the exact payload a
future GUI renders. It contains:

- the `question`;
- the ordered `trace.steps` list; per step the `decision` (a `call_tool` with the
  tool name, validated `args`, and `rationale`; or a `finish`; or an `error`), the
  `result` (`ok`, `error_kind`, `content`, and `data` carrying the serialized
  retrieved chunks with their 1-based `[C#]` indices and full `Citation`s), and the
  per-step `elapsed_ms`;
- the final `answer` (a `ModelAnswer` with summary and `[C#]`-cited steps, or a
  `Refusal`);
- `stop_reason`, `used_tools`, `total_ms`, and `trace_version`.

Because each `Citation` carries `source_doc_id`, `locator.paragraph`,
`tm_page_label`, and `pdf_page_index`, the GUI can deep-link a `[C#]` chip straight
to the exact manual page.

## Renderable as-is

The trace maps cleanly onto a timeline / stepper:

- each step is a card (tool icon + args + rationale);
- its result is an expandable panel (the chunk list with scores and clickable
  citations);
- the answer panel shows `[C#]` chips that highlight the originating chunk cards;
- `elapsed_ms` drives a latency view;
- `error_kind` (unknown_tool / invalid_args / tool_error) and `stop_reason` drive
  error and refusal badges.

`ToolRegistry.specs()` (name + description + args JSON-schema) lets the GUI render a
live tool catalog and argument forms with no extra plumbing.

## The seam, not a server

Three forward-compatible moves are taken now and nothing else:

1. Keep `AgentResult` / `AgentStep` / `ToolResult` pydantic and JSON-serializable,
   versioned via `trace_version`, living in `schemas/agent.py`, so the GUI contract
   is decoupled from internal loop refactors (exactly as the V1 answer schema
   decoupled training from inference).
2. Keep `Planner` a Protocol and `run_agent` a pure synchronous function over
   `(question, planner, registry, config)`, so a future thin transport (for example
   FastAPI plus server-sent events) can wrap it without touching agent logic. A
   later streaming planner that yields decisions, or an optional `on_step` callback
   for live progress, can be added without changing the return contract.
3. Keep tools self-describing via `specs()`. The `ScriptedPlanner` path doubles as
   the GUI's deterministic fixture/replay source: a captured `AgentResult.trace`
   replays with no model.

## Explicitly out of scope for V2.0

Any web framework, async, or streaming; and any tool that needs interactive user
input. A future fault-tree tool (V2.2) would surface an `awaiting_user` result that
the GUI renders as a prompt; that is not part of the V2.0 shell.
