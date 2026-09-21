"""Reasoning memory on Neo4j, using Neo4j's OFFICIAL agent-memory SDK.

Demos 01-05 store WHAT the agent knows. This stores WHY it decided: the question ->
reasoning steps -> tool calls -> touched sources chain. We do NOT hand-roll the graph
schema. We use `neo4j-agent-memory` (Neo4j Labs), the official reasoning-memory SDK,
so the schema, the writes, and the audit traversal are Neo4j's, not ours.

Everything is LIVE: the agent takes real decisions with real tools, and a Strands
HookProvider (Neo4jDecisionRecorder) records each one into Neo4j as it happens. No
hardcoded history.

Neo4j creates and manages this schema:

    (:ReasoningTrace)-[:HAS_STEP]->(:ReasoningStep)-[:USES_TOOL]->(:ToolCall)-[:INSTANCE_OF]->(:Tool)
    (:ReasoningStep)-[:TOUCHED]->(:Entity)

The reverse audit ("evidence source S was wrong, which decisions touched it?") is one
traversal over the `:TOUCHED` edges the SDK records. See VISUALIZE_QUERIES for the
Neo4j Browser queries that draw the graph.

Docs: https://neo4j.com/labs/agent-memory/how-to/reasoning-traces/
"""

import asyncio as _asyncio
import concurrent.futures as _futures
import os
import re

from dotenv import load_dotenv

load_dotenv()

from neo4j import GraphDatabase
from neo4j_agent_memory import MemoryClient, MemorySettings
from neo4j_agent_memory.memory.reasoning import EntityRef
from strands.hooks import (
    AfterInvocationEvent, AfterToolCallEvent, BeforeInvocationEvent,
    HookProvider, HookRegistry,
)

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
# Its own isolated database, kept clean for this demo (the SDK creates its own vector
# indexes, which must not collide with the 1024-dim indexes other demos use).
NEO4J_DATABASE = os.getenv("REASONING_NEO4J_DATABASE", "reasoningdemo")

# The evidence source we later declare compromised, for the reverse audit.
COMPROMISED_SOURCE = "fare_alerts_feed"

# Which external source each tool reads from. This is tool-catalog metadata (a fact
# about each tool), not a scripted history: it lets the recorder tag what each live
# tool call actually touched, so the audit can traverse a decision to that source.
TOOL_SOURCE = {
    "search_flights": "flight_search",
    "check_fare_alert": "fare_alerts_feed",
    "check_weather": "weather_api",
    "best_time_to_visit": "weather_api",
    "find_restaurants": "dining_guide",
}


# ── Neo4j Browser queries to SEE the graph (paste into the reasoningdemo database) ──
# Return paths so the Browser draws the edges; if you only RETURN nodes, turn on
# "Connect result nodes" in the Browser settings.
VISUALIZE_QUERIES = {
    "full_graph": (
        "// The whole reasoning graph: each decision's trace, its steps and tool calls,\n"
        "// and the sources those calls touched.\n"
        "MATCH p = (:ReasoningTrace)-[:HAS_STEP]->(:ReasoningStep)-[:USES_TOOL]->(:ToolCall)\n"
        "RETURN p\n"
        "UNION\n"
        "MATCH p = (:ReasoningStep)-[:TOUCHED]->(:Entity)\n"
        "RETURN p"
    ),
    "reverse_audit": (
        "// Reverse audit: every decision whose steps touched the compromised source.\n"
        "MATCH p = (:ReasoningTrace)-[:HAS_STEP]->(:ReasoningStep)"
        "-[:TOUCHED]->(:Entity {name: \"fare_alerts_feed\"})\n"
        "RETURN p"
    ),
    "one_decision": (
        "// One decision end to end: its steps, tools, and touched sources.\n"
        "MATCH p = (t:ReasoningTrace)-[*1..3]-()\n"
        "WHERE t.task CONTAINS \"Madrid\"\n"
        "RETURN p"
    ),
}


def _settings() -> MemorySettings:
    return MemorySettings(neo4j={
        "uri": NEO4J_URI, "username": NEO4J_USER,
        "password": NEO4J_PASSWORD, "database": NEO4J_DATABASE,
    })


def _valid_identifier(name: str) -> bool:
    """Database names cannot be parameterized in Cypher, so they are interpolated.
    Validate first (same guard as demos 03 and 05) so only a safe name reaches
    CREATE/DROP DATABASE."""
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name))


def memory_client() -> MemoryClient:
    """A Neo4j agent-memory client bound to this demo's isolated database. Use as an
    async context manager: `async with memory_client() as client: ...`."""
    return MemoryClient(_settings())


# ── Database lifecycle (plain driver; the SDK does not create databases) ─────
def ensure_clean_database() -> None:
    """Create the isolated reasoningdemo database and clear it, so each run starts
    fresh. The SDK builds its own schema/indexes inside it on first connect."""
    if not _valid_identifier(NEO4J_DATABASE):
        raise ValueError(f"Invalid database name: {NEO4J_DATABASE!r}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    try:
        with driver.session(database="system") as s:
            s.run(f"CREATE DATABASE {NEO4J_DATABASE} IF NOT EXISTS")
        import time
        time.sleep(2)
        with driver.session(database=NEO4J_DATABASE) as s:
            s.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()


def teardown_database() -> None:
    """Drop the isolated database entirely, so nothing this demo created is left."""
    if not _valid_identifier(NEO4J_DATABASE):
        raise ValueError(f"Invalid database name: {NEO4J_DATABASE!r}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database="system") as s:
            s.run(f"DROP DATABASE {NEO4J_DATABASE} IF EXISTS")
    finally:
        driver.close()


# ── The queries the demo measures (deterministic Cypher over the SDK's schema) ──
async def replay_why(client: MemoryClient, topic: str) -> dict | None:
    """Replay "why did I decide X?" by reading the recorded trace whose task mentions
    the topic: its outcome and the tools it used. One traversal, no model call, so the
    answer is the REAL recorded chain, not a reconstruction."""
    rows = await client.query.cypher(
        "MATCH (t:ReasoningTrace) WHERE toLower(t.task) CONTAINS toLower($topic) "
        "OPTIONAL MATCH (t)-[:HAS_STEP]->(:ReasoningStep)-[:USES_TOOL]->(tc:ToolCall) "
        "WITH t, collect(DISTINCT tc.tool_name) AS tools "
        "RETURN t.task AS question, t.outcome AS outcome, tools "
        "ORDER BY size(t.task) LIMIT 1",
        {"topic": topic},
    )
    return rows[0] if rows else None


async def find_affected(client: MemoryClient,
                        source_name: str = COMPROMISED_SOURCE) -> list[str]:
    """The reverse audit as ONE traversal over the SDK's `:TOUCHED` edges: every
    decision whose reasoning touched the compromised source. A flat log would need a
    scan of each record; the graph answers it with one query, at read time."""
    rows = await client.query.cypher(
        "MATCH (t:ReasoningTrace)-[:HAS_STEP]->(:ReasoningStep)"
        "-[:TOUCHED]->(:Entity {name: $source}) "
        "RETURN DISTINCT t.task AS decision ORDER BY decision",
        {"source": source_name},
    )
    return [r["decision"] for r in rows]


async def sources_touched(client: MemoryClient) -> list[dict]:
    """Which sources each decision touched, the raw map behind the audit."""
    rows = await client.query.cypher(
        "MATCH (t:ReasoningTrace)-[:HAS_STEP]->(:ReasoningStep)-[:TOUCHED]->(e:Entity) "
        "RETURN t.task AS decision, collect(DISTINCT e.name) AS sources ORDER BY decision"
    )
    return rows


async def all_decisions(client: MemoryClient) -> list[str]:
    """Every decision recorded, for reporting the total."""
    rows = await client.query.cypher(
        "MATCH (t:ReasoningTrace) RETURN t.task AS decision ORDER BY decision"
    )
    return [r["decision"] for r in rows]


# ── Live recorder: a Strands HookProvider that writes to Neo4j as the agent runs ──
def _run_async(coro):
    """Run an async SDK call from Strands' synchronous hook callbacks. If a loop is
    already running (notebook), run on a worker thread; otherwise asyncio.run.

    We drain any pending tasks before the loop closes: the Neo4j SDK's HTTP client
    schedules its teardown as a task, and letting asyncio.run close the loop first
    prints a harmless 'Event loop is closed' traceback. Draining avoids that noise."""
    async def _driver():
        result = await coro
        pending = [t for t in _asyncio.all_tasks() if t is not _asyncio.current_task()]
        if pending:
            await _asyncio.gather(*pending, return_exceptions=True)
        return result

    try:
        _asyncio.get_running_loop()
    except RuntimeError:
        return _asyncio.run(_driver())
    with _futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: _asyncio.run(_driver())).result()


def _last_user_text(messages) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            for b in m.get("content", []):
                if isinstance(b, dict) and "text" in b:
                    return b["text"]
    return ""


class Neo4jDecisionRecorder(HookProvider):
    """Records each live agent invocation as a reasoning trace in Neo4j, via the
    official SDK. Attach with ``Agent(hooks=[Neo4jDecisionRecorder()])``.

    This is where Strands and Neo4j meet. Strands emits lifecycle events with the data
    already attached: ``AfterToolCallEvent`` carries ``event.tool_use`` (tool + input),
    so the recorder never touches tool code. Neo4j's SDK takes those and stores them as
    a connected, queryable trace. Zero changes to the tools; it captures whatever the
    agent actually did.

    Strands runs hook callbacks synchronously, and each ``agent(...)`` runs on its own
    event loop. A Neo4j async client is bound to the loop that opened it, so the recorder
    does NOT hold a long-lived client: it opens a fresh one per write, inside the same
    loop that performs that write. That keeps every await on one loop.
    """

    def __init__(self, session_id: str = "live"):
        self._session_id = session_id
        self._question = None
        self._tool_calls = []

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeInvocationEvent, self._on_start)
        registry.add_callback(AfterToolCallEvent, self._on_tool)
        registry.add_callback(AfterInvocationEvent, self._on_end)

    def _on_start(self, event: BeforeInvocationEvent) -> None:
        self._question = _last_user_text(event.messages)
        self._tool_calls = []

    def _on_tool(self, event: AfterToolCallEvent) -> None:
        name = event.tool_use["name"]
        self._tool_calls.append({
            "tool": name,
            "input": event.tool_use.get("input") or {},
            "source": TOOL_SOURCE.get(name),   # which external source this tool read
        })

    def _on_end(self, event: AfterInvocationEvent) -> None:
        if not self._question:
            return
        outcome = str(event.result).strip()[:400]
        _run_async(self._write(self._question, list(self._tool_calls), outcome))
        self._question = None

    async def _write(self, question, tool_calls, outcome) -> None:
        # Open a client inside THIS loop (the one _run_async created), so every await
        # runs on a single event loop. Never reuse a client from another loop.
        async with memory_client() as client:
            trace = await client.reasoning.start_trace(
                session_id=self._session_id, task=question)
            for call in tool_calls:
                rs = await client.reasoning.add_step(
                    trace.id, thought=f"Call {call['tool']}", action=f"invoke {call['tool']}")
                touched = None
                src = call.get("source")
                if src:
                    ent, _ = await client.long_term.add_entity(src, entity_type="Source")
                    touched = [EntityRef(id=str(ent.id), name=src, type="Source")]
                await client.reasoning.record_tool_call(
                    rs.id, call["tool"], call["input"], touched_entities=touched)
            await client.reasoning.complete_trace(
                trace.id, outcome=outcome, success=True)
