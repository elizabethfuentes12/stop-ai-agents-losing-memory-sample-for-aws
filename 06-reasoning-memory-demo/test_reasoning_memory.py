"""
Demo: Reasoning Memory, Remember WHY You Decided, Not Just What You Know

Demos 01-04 store what the agent knows. This demo stores why it decided: the
question → tool steps → evidence → outcome chain, with provenance. Without it, "why did
you recommend X?" gets a confabulated answer, the agent invents a plausible
justification because the real reasoning was never kept.

Two tracks, same traces:

  - Key-value (agent.state): a Strands HookProvider records each invocation's trace
    automatically, zero changes to the tools. Replaying "why did I decide X?" works
    great from the flat store.
  - Graph (Neo4j): the same traces stored as node chains with evidence provenance as
    edges. Identical replay, plus the REVERSE audit a flat store can't express:
    "evidence source S was wrong, which decisions depended on it?" at any depth.

  Test 1: The problem, no trace kept, the agent confabulates a justification
  Test 2: Standalone, HookProvider records the trace; replay answers the why
  Test 3: Graph, same trace as a node chain; replay + provenance per edge
  Test 4: The reverse audit, flat scan finds only direct dependents; the graph
          traversal finds them ALL (including through other decisions' outputs)
  Comparison: decisions found by the audit, flat scan vs graph traversal

Honesty note: "reasoning memory" is an engineering pattern, not an established category
in academic memory taxonomies. Closest verified research (traceability/provenance):
  - MemWeaver (https://arxiv.org/abs/2601.18204), traceable long-horizon agentic reasoning
  - "Less Context, More Accuracy" (https://arxiv.org/abs/2606.09900), the Engram system;
    every stored fact keeps provenance and a supersession chain (single-author preprint)

Tests 1-2 need only an OPENAI_API_KEY. Tests 3-4 also need a running Neo4j. See README.
"""

import os
from dotenv import load_dotenv

from strands import Agent
# Using OpenAI-compatible interface via Strands SDK (not direct OpenAI usage)
from strands.models.openai import OpenAIModel

import trace_kv as kv
import trace_graph as tg

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not set. Get your API key from https://platform.openai.com/api-keys "
        "then either: 1) Add OPENAI_API_KEY=your-key to a .env file, or "
        "2) Run: export OPENAI_API_KEY=your-key"
    )

# --- Model: OpenAI by default so it runs locally with just an OPENAI_API_KEY (no AWS setup) ---
# Using the OpenAI-compatible interface via the Strands SDK (not direct OpenAI usage)
MODEL = OpenAIModel(model_id="gpt-4o-mini")  # api_key read from the OPENAI_API_KEY env var

# To run on Amazon Bedrock instead (no OpenAI key; uses your AWS credentials), comment the
# two lines above and uncomment these:
# from strands.models import BedrockModel
# MODEL = BedrockModel(model_id="openai.gpt-oss-120b-1:0", region_name="us-west-2")

SYSTEM_PROMPT = (
    "You are a travel assistant. Use your tools to answer. Be concise, 2-3 sentences maximum."
)

WHY_QUESTION = "Why did you recommend that Madrid flight?"


def run_test_1_no_trace():
    """Test 1: The problem, the reasoning is gone, so the agent confabulates.

    One agent, no recorder. It decides with its tools, then is asked WHY. Nothing
    persisted the reasoning chain, so there is no trace to consult and the answer is a
    plausible reconstruction, not the real steps. (A session manager would carry the
    conversation forward across restarts, but not the tool-by-tool reasoning, that is
    exactly the gap the recorder in Test 2 fills.)
    """
    print("\n" + "=" * 70)
    print("TEST 1: NO TRACE, the reasoning is lost")
    print("=" * 70)

    # No DecisionTraceRecorder attached: the agent runs its tools but keeps no trace.
    agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT,
                  tools=[kv.search_flights, kv.check_fare_alert], callback_handler=None)
    decision = agent("Find me a flight JFK to Madrid on 2026-10-10 and pick the best option.")
    print(f"\n  Decision made: {str(decision).strip()[:160]}")

    # Ask the same agent WHY. With no recorder, the reasoning chain was never stored.
    answer = agent(WHY_QUESTION)
    print(f"\n  Asked WHY: {str(answer).strip()[:220]}")

    # Deterministic check: how many real reasoning steps can be recovered? No recorder
    # ran, so no trace was written, so the answer is necessarily 0.
    steps_recoverable = len(agent.state.get(kv.TRACES_KEY) or [])
    print(f"\n  Real reasoning steps recoverable from the store: {steps_recoverable}")
    print("  Whatever the answer says, it is a plausible reconstruction, not the real chain.")
    return {"steps_recoverable": steps_recoverable}


def run_test_2_recorder():
    """Test 2: The standalone fix, a HookProvider records the trace automatically.

    Same tools, same question. The only change is Agent(hooks=[DecisionTraceRecorder()]).
    Afterwards the trace is in agent.state, and why_did_i replays the REAL chain.
    """
    print("\n" + "=" * 70)
    print("TEST 2: RECORDER, the HookProvider captures the trace (zero tool changes)")
    print("=" * 70)

    agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT,
                  tools=[kv.search_flights, kv.check_fare_alert, kv.why_did_i],
                  hooks=[kv.DecisionTraceRecorder()], callback_handler=None)
    decision = agent("Find me a flight JFK to Madrid on 2026-10-10 and pick the best option.")
    print(f"\n  Decision made: {str(decision).strip()[:160]}")

    traces = agent.state.get(kv.TRACES_KEY) or []
    recorded_steps = sum(len(t["steps"]) for t in traces)
    print(f"  Recorded automatically: {len(traces)} trace(s), {recorded_steps} step(s) with evidence")

    answer = agent(WHY_QUESTION)
    print(f"\n  Asked WHY (agent replays its own trace): {str(answer).strip()[:220]}")

    # Deterministic check: the replay recovers the actual recorded chain.
    trace = kv.replay_why(agent.state.get(kv.TRACES_KEY) or [], "Madrid")
    replayed_steps = len(trace["steps"]) if trace else 0
    print(f"\n  Real reasoning steps recoverable from the store: {replayed_steps}/{recorded_steps}")
    return {"recorded_steps": recorded_steps, "replayed_steps": replayed_steps}


def run_test_3_graph_replay():
    """Test 3: The graph track, same traces as node chains, replay works the same."""
    print("\n" + "=" * 70)
    print("TEST 3: GRAPH, traces as node chains with provenance edges")
    print("=" * 70)

    driver = tg.get_driver()
    db = tg.ensure_database(driver)
    try:
        tg.reset_graph(driver, db)
        summary = tg.seed_graph(driver, db)
        print(f"  Seeded: {summary['decisions']} decisions, {summary['evidence']} evidence records, "
              f"{summary['sources']} external sources")

        trace = tg.replay_why_graph(driver, db, "Iberia")
        print(f"\n  Replay 'why the Madrid flight?': {trace['question']}")
        for step in trace["steps"]:
            print(f"    step {step['n']}: {step['tool']}({step['input']}) -> {step['evidence']}")
        print(f"  Outcome: {trace['outcome']}")

        replayed_steps = len(trace["steps"])
        print(f"\n  Real reasoning steps recoverable from the graph: {replayed_steps}")
        return {"replayed_steps": replayed_steps, "driver": driver, "db": db}
    except Exception:
        driver.close()
        raise


def run_test_4_reverse_audit(driver, db):
    """Test 4: The reverse audit, 'this source was wrong; which decisions relied on it?'

    Ground truth (by construction of the seed): 4 of 5 decisions depend on the fare-alerts
    feed, 2 directly, 1 through the flight decisions' outputs, 1 two hops away. The
    flat scan sees only direct citations; the graph traversal follows the provenance
    chain to any depth. The packing-list control depends on the weather API and must
    NOT be flagged by either.
    """
    print("\n" + "=" * 70)
    print("TEST 4: REVERSE AUDIT, the fare-alerts feed turned out to be compromised")
    print("=" * 70)

    try:
        ground_truth = sorted(kv.AFFECTED_IDS)
        print(f"  Ground truth (by construction): {len(ground_truth)} affected decisions {ground_truth}")

        # Flat store: linear scan over each trace's own evidence blobs.
        kv_found = sorted(kv.find_affected_decisions_kv(kv.SEED_TRACES, kv.COMPROMISED_SOURCE))
        print(f"\n  Flat scan (key-value):   found {len(kv_found)}/{len(ground_truth)}  {kv_found}")
        missed = sorted(set(ground_truth) - set(kv_found))
        print(f"  Missed (indirect deps):  {missed}, their blobs never mention '{kv.COMPROMISED_SOURCE}'")

        # Graph: one traversal, any depth.
        graph_found = sorted(tg.find_affected_decisions_graph(driver, db))
        print(f"\n  Graph traversal:         found {len(graph_found)}/{len(ground_truth)}  {graph_found}")

        control_clean = kv.CONTROL_ID not in kv_found and kv.CONTROL_ID not in graph_found
        print(f"  Control '{kv.CONTROL_ID}' correctly NOT flagged by either: {control_clean}")

        print("\n  The receipts, provenance path from each indirect decision to the source:")
        for decision_id in missed:
            hops = tg.provenance_path(driver, db, decision_id)
            print(f"    {decision_id}: {' -> '.join(hops)}")

        return {"total": len(ground_truth), "kv_found": len(kv_found),
                "graph_found": len(graph_found), "control_clean": control_clean,
                "kv_ids": kv_found, "graph_ids": graph_found}
    finally:
        # Full teardown: clear the demo's nodes and DROP the isolated database
        # (guarded so the shared default DB is never dropped).
        try:
            tg.teardown_graph(driver, db)
        finally:
            driver.close()


if __name__ == "__main__":
    print("=" * 70)
    print("  REASONING MEMORY DEMO")
    print("  Store WHY the agent decided, replay it, and audit it in reverse")
    print("=" * 70)

    r1 = run_test_1_no_trace()
    r2 = run_test_2_recorder()
    r3 = run_test_3_graph_replay()
    r4 = run_test_4_reverse_audit(r3.pop("driver"), r3.pop("db"))

    print("\n" + "=" * 70)
    print("  COMPARISON, the reverse audit: 'source S was wrong, what did it touch?'")
    print("=" * 70)
    print(f"\n  {'Store':<26} {'Affected decisions found':>26}")
    print("  " + "-" * 54)
    print(f"  {'Key-value (flat scan)':<26} {str(r4['kv_found']) + '/' + str(r4['total']):>26}")
    print(f"  {'Graph (traversal)':<26} {str(r4['graph_found']) + '/' + str(r4['total']):>26}")

    print("\n  Key insight: both stores replay 'why did I decide X?' equally well.")
    print("  The graph earns its keep on the REVERSE question, a flat scan only finds")
    print(f"  decisions that cite the source directly ({r4['kv_found']}/{r4['total']}); the provenance")
    print(f"  traversal follows evidence through other decisions' outputs ({r4['graph_found']}/{r4['total']}).")
    print("\n  Note: 'reasoning memory' is an engineering pattern, not an established")
    print("  academic memory category, see README for what the research does support.")
    print("\n  Research: https://arxiv.org/abs/2601.18204 (MemWeaver)")
    print("  Research: https://arxiv.org/abs/2606.09900 (Engram system, provenance chains)")
    print("  Strands:  https://github.com/strands-agents/sdk-python")
