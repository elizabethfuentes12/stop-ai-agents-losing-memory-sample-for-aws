# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Interactive chat to try the two reasoning-memory agents by hand, outside the notebook.

Two agents, one per backend, each recording its decisions as you talk to it:

  --flat   (default) records each decision's trace into agent.state (a Strands
           HookProvider), survives a restart with a session manager. Ask "why" and it
           replays the real chain instead of confabulating.

  --graph  records each live decision into Neo4j with the official neo4j-agent-memory
           SDK. Adds the reverse audit: "which decisions touched a given source?".

Usage:
    uv run python chat_test.py            # flat agent
    uv run python chat_test.py --graph    # graph agent (needs Neo4j)

Commands inside the chat:
    /why <topic>     replay why a past decision was made (the real recorded chain)
    /audit <source>  (graph only) which decisions touched a source, e.g. /audit fare_alerts_feed
    /traces          list the decisions recorded so far
    /quit
"""

import argparse
import asyncio
import os
import sys

os.environ["OTEL_SDK_DISABLED"] = "true"

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
    sys.exit(1)

from strands import Agent
from strands.models.openai import OpenAIModel

import trace_kv as kv

SYSTEM_PROMPT = "You are a travel assistant. Use your tools to answer. Be concise: 2-3 sentences."
TOOLS = [kv.search_flights, kv.check_fare_alert, kv.check_weather, kv.best_time_to_visit]


def run_flat() -> None:
    """Flat agent: DecisionTraceRecorder writes traces into agent.state."""
    agent = Agent(model=OpenAIModel(model_id="gpt-4o-mini"), system_prompt=SYSTEM_PROMPT,
                  tools=TOOLS + [kv.why_did_i], hooks=[kv.DecisionTraceRecorder()],
                  callback_handler=None)
    print("\n" + "=" * 64)
    print("  REASONING MEMORY, flat agent (agent.state)")
    print("  Talk to it; every decision is recorded. Try /why and /traces.")
    print("=" * 64)
    print(__doc__)
    while True:
        try:
            text = input("You: ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text.lower() == "/quit":
            break
        if text.lower().startswith("/why"):
            topic = text[4:].strip() or "flight"
            trace = kv.replay_why(agent.state.get(kv.TRACES_KEY) or [], topic)
            if trace is None:
                print(f"No recorded decision mentions '{topic}'.\n")
            else:
                print(f"\nDecision: {trace['question']}")
                for s in trace["steps"]:
                    print(f"  step {s['n']}: {s['tool']}({s['input']}) -> {s['evidence']['text'][:70]}")
                print(f"Outcome: {trace['outcome'][:120]}\n")
            continue
        if text.lower() == "/traces":
            traces = agent.state.get(kv.TRACES_KEY) or []
            print(f"\n{len(traces)} decision(s) recorded:")
            for t in traces:
                print(f"  - {t['question'][:60]}")
            print()
            continue
        if text.lower().startswith("/audit"):
            print("/audit needs the graph agent. Run: uv run python chat_test.py --graph\n")
            continue
        resp = agent(text)
        print(f"\nAgent: {str(resp).strip()}\n")


def run_graph() -> None:
    """Graph agent: Neo4jDecisionRecorder writes live decisions into Neo4j."""
    import trace_graph as tg
    tg.ensure_clean_database()
    agent = Agent(model=OpenAIModel(model_id="gpt-4o-mini"), system_prompt=SYSTEM_PROMPT,
                  tools=TOOLS, hooks=[tg.Neo4jDecisionRecorder(session_id="chat")],
                  callback_handler=None)
    print("\n" + "=" * 64)
    print("  REASONING MEMORY, graph agent (Neo4j, official SDK)")
    print("  Talk to it; decisions are recorded live. Try /why, /audit, /traces.")
    print("=" * 64)
    print(__doc__)

    async def _query(coro_fn):
        async with tg.memory_client() as client:
            return await coro_fn(client)

    try:
        while True:
            try:
                text = input("You: ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text.lower() == "/quit":
                break
            if text.lower().startswith("/why"):
                topic = text[4:].strip() or "flight"
                r = asyncio.run(_query(lambda c: tg.replay_why(c, topic)))
                if r is None:
                    print(f"No recorded decision mentions '{topic}'.\n")
                else:
                    print(f"\nQ: {r['question']}\nOutcome: {r['outcome'][:120]}\nTools used: {r['tools']}\n")
                continue
            if text.lower().startswith("/audit"):
                source = text[6:].strip() or tg.COMPROMISED_SOURCE
                affected = asyncio.run(_query(lambda c: tg.find_affected(c, source)))
                print(f"\nDecisions that touched '{source}': {len(affected)}")
                for a in affected:
                    print(f"  - {a[:60]}")
                print()
                continue
            if text.lower() == "/traces":
                decs = asyncio.run(_query(tg.all_decisions))
                print(f"\n{len(decs)} decision(s) recorded:")
                for d in decs:
                    print(f"  - {d[:60]}")
                print()
                continue
            resp = agent(text)
            print(f"\nAgent: {str(resp).strip()}\n")
    finally:
        tg.teardown_database()
        print("\nGraph torn down. Goodbye.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the reasoning-memory agents.")
    parser.add_argument("--graph", action="store_true", help="Use the Neo4j graph agent (needs Neo4j).")
    args = parser.parse_args()
    if args.graph:
        run_graph()
    else:
        run_flat()
