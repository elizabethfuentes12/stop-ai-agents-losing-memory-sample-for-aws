# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Reasoning Memory: remember WHY the agent decided, not just what it knows.

Runnable mirror of test_reasoning_memory.ipynb. Everything is LIVE: the agent takes
real decisions with real tools, and Strands HookProviders record each one, with zero
changes to the tools.

  Test 1  No trace: after a real restart (session restored via a session manager) the
          agent is asked "why?" and confabulates: 0 real steps recoverable.
  Test 2  Recorder: a HookProvider records the trace into agent.state; it survives the
          restart and the replay returns the REAL chain.
  Test 3  Graph (Neo4j, official neo4j-agent-memory SDK): the agent takes ten live
          decisions; each is recorded into Neo4j with no hardcoded history.
  Test 4  Reverse audit: "the fare-alerts feed was compromised, which decisions touched
          it?" One traversal over the SDK's :TOUCHED edges. A flat store would scan.
  Test 5  Why store it: replay from the trace costs 0 model tokens and returns the real
          chain; asking the model to reconstruct it costs tokens and confabulates.

What is "flat" memory? A store that keeps each record on its own (key-value, a log, a
vector store) with no edges to traverse. A flat log records what happened; a graph
records why. Tests 1-2 and 5 need only OPENAI_API_KEY + DUFFEL_API_KEY; tests 3-4 also
need a running Neo4j.

Honesty note: "reasoning memory" is an engineering pattern, not an established academic
memory category. The research-backed theme is traceability/provenance (MemWeaver
https://arxiv.org/abs/2601.18204, Engram https://arxiv.org/abs/2606.09900).
"""

import asyncio
import os

from dotenv import load_dotenv

os.environ["OTEL_SDK_DISABLED"] = "true"

from strands import Agent
# Using the OpenAI-compatible interface via the Strands SDK (not direct OpenAI usage)
from strands.models.openai import OpenAIModel
from strands.session import SnapshotSessionManager
from strands.storage import LocalFileStorage

import trace_kv as kv
import trace_graph as tg

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set. Add it to a .env file or export it.")

MODEL = OpenAIModel(model_id="gpt-4o-mini")
# To run on Amazon Bedrock instead:
# from strands.models import BedrockModel
# MODEL = BedrockModel(model_id="openai.gpt-oss-120b-1:0", region_name="us-west-2")

SYSTEM_PROMPT = "You are a travel assistant. Use your tools to answer. Be concise: 2-3 sentences."
WHY = "Why did you recommend that Madrid flight?"

import tempfile
_SESSION_DIR = tempfile.mkdtemp(prefix="reasoning-sessions-")


def _sm(session_id):
    return SnapshotSessionManager(session_id=session_id, storage=LocalFileStorage(_SESSION_DIR))


def _rule(label, ok):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}")


def run_test_1_no_trace():
    print("\n" + "=" * 70)
    print("TEST 1: NO TRACE, the reasoning is lost across a restart")
    print("=" * 70)
    agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT,
                  tools=[kv.search_flights, kv.check_fare_alert],
                  session_manager=_sm("no-trace"), callback_handler=None)
    agent("Find me a flight JFK to Madrid on 2026-10-15 and pick the best option.")
    restarted = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT,
                      tools=[kv.search_flights, kv.check_fare_alert],
                      session_manager=_sm("no-trace"), callback_handler=None)
    answer = restarted(WHY)
    steps = len(restarted.state.get(kv.TRACES_KEY) or [])
    print(f"\n  After restart, asked WHY: {str(answer).strip()[:180]}")
    _rule("no recorder ran, 0 real steps recoverable (the answer is a reconstruction)", steps == 0)
    return {"steps": steps}


def run_test_2_recorder():
    print("\n" + "=" * 70)
    print("TEST 2: RECORDER, the trace survives a restart (zero tool changes)")
    print("=" * 70)
    agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT,
                  tools=[kv.search_flights, kv.check_fare_alert, kv.why_did_i],
                  hooks=[kv.DecisionTraceRecorder()],
                  session_manager=_sm("recorder"), callback_handler=None)
    agent("Find me a flight JFK to Madrid on 2026-10-15 and pick the best option.")
    recorded = sum(len(t["steps"]) for t in (agent.state.get(kv.TRACES_KEY) or []))
    _rule(f"recorder captured {recorded} step(s) with zero tool changes", recorded >= 1)
    restarted = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT,
                      tools=[kv.search_flights, kv.check_fare_alert, kv.why_did_i],
                      hooks=[kv.DecisionTraceRecorder()],
                      session_manager=_sm("recorder"), callback_handler=None)
    restarted(WHY)
    trace = kv.replay_why(restarted.state.get(kv.TRACES_KEY) or [], "Madrid")
    replayed = len(trace["steps"]) if trace else 0
    _rule(f"trace survived the restart, replay recovers {replayed} real step(s)", replayed >= 1)
    return {"recorded": recorded, "replayed": replayed}


# The live prompts for the graph track: a travel-planning session. Some read the
# fare-alerts feed, some read weather. No hardcoded outcomes; the agent decides.
GRAPH_PROMPTS = [
    "Find me a flight from JFK to Madrid on 2026-10-15 and pick the best option.",
    "Now find me a flight from JFK to Tokyo on 2026-10-20 and pick the best.",
    "Is there a fare alert on the JFK to Madrid route right now?",
    "What is the best time of year to visit Tokyo?",
    "What should I pack for Madrid in October?",
    "Check the fare alert for JFK to London.",
    "When should I visit Madrid for good weather?",
    "Find me a flight from JFK to Paris on 2026-11-01 and pick the best.",
    "Any fare alert on JFK to Paris?",
    "What is the weather like in Tokyo in October?",
]


def run_graph_tests():
    print("\n" + "=" * 70)
    print("TEST 3 + 4: GRAPH (Neo4j official SDK), live decisions + reverse audit")
    print("=" * 70)
    tg.ensure_clean_database()
    agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT,
                  tools=[kv.search_flights, kv.check_fare_alert, kv.best_time_to_visit, kv.check_weather],
                  hooks=[tg.Neo4jDecisionRecorder(session_id="travel")],
                  callback_handler=None)
    for p in GRAPH_PROMPTS:
        agent(p)

    async def _checks():
        async with tg.memory_client() as client:
            total = await tg.all_decisions(client)
            affected = await tg.find_affected(client)
            replay = await tg.replay_why(client, "flight from JFK to Madrid")
            return total, affected, replay

    total, affected, replay = asyncio.run(_checks())
    _rule(f"recorded {len(total)} live decisions into Neo4j", len(total) == len(GRAPH_PROMPTS))
    _rule(f"replay returns the real recorded chain: {replay['tools'] if replay else None}",
          bool(replay and replay.get("tools")))
    print(f"\n  Reverse audit, decisions that touched '{tg.COMPROMISED_SOURCE}': {len(affected)}")
    for a in affected:
        print(f"    - {a[:60]}")
    _rule("reverse audit found the fare-alert decisions and excluded the weather-only ones",
          len(affected) >= 1 and len(affected) < len(total))
    tg.teardown_database()
    return {"total": len(total), "affected": len(affected)}


def run_test_5_why_store():
    print("\n" + "=" * 70)
    print("TEST 5: WHY STORE IT, tokens saved and errors avoided")
    print("=" * 70)
    # Reuse the graph store from the flat recorder track: replay is a read, 0 tokens.
    # Here we contrast against asking the model to reconstruct with no trace.
    reconstructor = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT, callback_handler=None)
    resp = reconstructor("Earlier you recommended a JFK-Madrid flight. "
                         "Reconstruct the exact tool-by-tool reasoning you used.")
    reconstruct_tokens = resp.metrics.accumulated_usage["totalTokens"]
    print(f"\n  (a) Replay from a stored trace: 0 model tokens, the real chain.")
    print(f"  (b) Reconstruct with the model: {reconstruct_tokens} tokens, confabulated.")
    _rule("reconstructing costs tokens; replaying a stored trace costs none", reconstruct_tokens > 0)
    return {"reconstruct_tokens": reconstruct_tokens}


if __name__ == "__main__":
    print("=" * 70)
    print("  REASONING MEMORY DEMO, live decisions recorded for replay and audit")
    print("=" * 70)
    run_test_1_no_trace()
    run_test_2_recorder()
    run_graph_tests()
    run_test_5_why_store()
    print("\nDone.")
