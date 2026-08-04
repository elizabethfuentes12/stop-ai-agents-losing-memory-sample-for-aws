# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Research-backed demos showing why AI agents lose memory and how to fix it using Strands Agents. Part of the larger `sample-why-agents-fail` monorepo (sibling folders: `stop-ai-agent-hallucinations`, `stop-ai-agents-wasting-tokens`).

Progressive demo series: key-value memory → vector memory → graph memory → selective memory → memory hygiene → reasoning memory → hybrid memory (design) → production deploy (pending).

## Commands

```bash
# Run a demo (from its directory)
# Using uv — a fast Python package manager (https://docs.astral.sh/uv/)
cd 01-key-value-memory-demo
uv venv && uv pip install -r requirements.txt
uv run python test_key_value_memory.py

# Run any demo the same way
cd 04-selective-memory-demo && uv run python test_selective_memory.py
cd 02-vector-memory-demo && uv run python test_vector_memory.py

# Generate charts (requires matplotlib)
cd 01-key-value-memory-demo/images && python3 generate_chart.py

# Syntax check all Python files
python3 -c "import ast; ast.parse(open('tools.py').read())"
```

Each demo is self-contained with its own `requirements.txt`. There is no shared venv — create one per demo or reuse across demos (deps are identical).

## Architecture

### Demo Pattern (consistent across all demos)

Each demo follows the same structure inherited from the sibling `stop-ai-agents-wasting-tokens` project:

- `tools.py` — Strands tool definitions using `@tool` and `@tool(context=True)`. This is where the core logic lives.
- `test_<name>.py` — Runnable script with multiple test functions (`run_test_1_*`, `run_test_2_*`, etc.) and a comparison table at the end.
- `test_<name>.ipynb` — Jupyter notebook mirroring the same tests with markdown explanations between cells.
- `images/generate_chart.py` — Matplotlib script that generates the `.png` chart for the README.

### Strands APIs Used

All demos use the OpenAI-compatible interface via Strands SDK:

```python
from strands import Agent, tool, ToolContext
from strands.models.openai import OpenAIModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.session import FileSessionManager, S3SessionManager
```

Key state mechanisms:
- `agent.messages` — conversation history, kept between calls on the same Agent instance and sent to the model every turn. NOT durable memory: unstructured, trimmed by conversation managers, gone on restart. Never describe an in-process agent as "forgetting between turns".
- `agent.state.set(key, value)` / `.get(key)` — per-agent key-value store, invisible to the model, read/written by tools (used across most demos)
- `FileSessionManager(session_id, storage_dir)` / `S3SessionManager(session_id, bucket, prefix)` — persist state across agent restarts (Demo 01 tests 3-4)
- `SlidingWindowConversationManager(window_size=N)` — limits conversation history length
- `HookProvider` hooks — record decision traces without touching tools (Demo 06)

### How Tools Access Memory

Tools use `@tool(context=True)` to receive a `ToolContext` parameter, which provides `tool_context.agent.state` for reading/writing memory. Stateless tools use plain `@tool` with no context access.

### Demo Data

All demos share a flights domain (travel assistant): live Duffel sandbox offers via `flights_api.py` (with `fallback_offers.json` captured once for offline resilience) and real Open-Meteo climate data via `weather_api.py`. A brand-new user starts with empty memory; demos that need pre-existing memories seed them explicitly (e.g. `seed_memory()` in demo 05).

### Teaching Pattern

Each test script isolates ONE variable while keeping the query/model constant:
- Test 1 = baseline (problem)
- Test 2 = solution
- Tests 3-4 = variations (evolution, persistence, multi-turn)

Comparison table printed at the end shows measured differences.

## Conventions

- All demos use `OpenAIModel(model_id="gpt-4o-mini")` as default. Swappable to any Strands-supported provider.
- `OPENAI_API_KEY` env var required. Validated at startup with descriptive error.
- System prompts are concise ("2-3 sentences maximum") to keep demo output readable.
- Session files created during tests are cleaned up with `shutil.rmtree()` at the end.
- Images: `.jpg` files are Canva conceptual diagrams, `.png` files are matplotlib-generated charts.
- The `# Using OpenAI-compatible interface via Strands SDK (not direct OpenAI usage)` comment is required above OpenAI imports to avoid Semgrep false positives.
- Each demo README must include research paper citations with arxiv links.
- The main README and each demo README must end with Contributing, Security, and License sections linking to the root repo files.
