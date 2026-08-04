"""
Demo: Graph Memory — Reasoning Over Relationships, Not Just Similarity

Based on research in graph-structured agent memory:
  - GAAMA: Graph Augmented Associative Memory for Agents (https://arxiv.org/abs/2603.27910) — 2026
  - MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents (https://arxiv.org/abs/2601.03236) — 2026
  - GRAVITY: Architecture-Agnostic Structured Anchoring for Long-Horizon Conversational Memory (https://arxiv.org/abs/2605.01688) — 2026

Semantic memory (Demo 02) retrieves by *similarity* but cannot reason over
*relationships*. A multi-hop question needs to find an entry point by similarity and
then TRAVERSE the graph to the answer. This demo contrasts the two on the same graph:

  Test 1: Semantic recall (before)    — pure vector similarity, misses the multi-hop chain
  Test 2: Graph recall (after)        — vector similarity + graph traversal, recovers it
  Test 3: Full Strands agent          — the agent uses graph memory to answer, and writes
                                        a new fact back into the graph (the harness)
  Test 4: Deterministic scorecard     — 4 multi-hop questions, before vs after (feeds the chart)

The graph is a KNOWN, seeded graph, so the before/after scores are reproducible.
Both strategies receive the SAME facts; the graph wins because it stores them as
connected nodes, not because it is handed the answer.

Requires a running Neo4j and an OPENAI_API_KEY. See README for setup.
"""

import os
from dotenv import load_dotenv

from strands import Agent
# Using OpenAI-compatible interface via Strands SDK (not direct OpenAI usage)
from strands.models.openai import OpenAIModel

import graph_memory as gm
import travel_tools as tt

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

# The headline multi-hop question the demo is built around.
QUESTION = gm.MULTIHOP_QUESTION

# Deterministic scorecard: multi-hop questions whose correct answer is the PERSON,
# reachable only by following relationships. Same known graph → reproducible scores.
SCORECARD = [
    ("Who do I know that's connected to flights to Spain?", "Maya Torres"),
    ("Who do I know connected to an airline that flies to Madrid?", "Maya Torres"),
    ("Who works at the Oneworld airline I know?", "Maya Torres"),
    ("Which person is linked to airlines in Spain?", "Maya Torres"),
]


def run_test_1_semantic(driver, db, embedder):
    """Test 1: Semantic recall only (the 'before'). Pure vector similarity, no traversal."""
    print("\n" + "=" * 70)
    print("TEST 1: SEMANTIC RECALL (before) — pure vector similarity")
    print("=" * 70)

    retriever = gm.make_before_retriever(driver, db, embedder)
    result = retriever.search(query_text=QUESTION, top_k=3)

    print(f"\nQuestion: {QUESTION}\n")
    print("Top-3 most similar memory nodes:")
    for item in result.items:
        print(f"  - {item.content}")

    recovered = any("Maya Torres" in item.content for item in result.items)
    print(f"\n  Recovers the person (Maya Torres)? {recovered}")
    print("  It surfaces Iberia / Madrid / Spain as separate pieces but cannot connect")
    print("  them to a person — similarity has no notion of a relationship.")
    return {"strategy": "semantic (before)", "recovered": recovered}


def run_test_2_graph(driver, db, embedder):
    """Test 2: Graph recall (the 'after'). Vector similarity to an entry node, then traversal."""
    print("\n" + "=" * 70)
    print("TEST 2: GRAPH RECALL (after) — vector similarity + graph traversal")
    print("=" * 70)

    retriever = gm.make_after_retriever(driver, db, embedder)
    result = retriever.search(query_text=QUESTION, top_k=3)

    print(f"\nQuestion: {QUESTION}\n")
    print("Traversal results (person + the relationship chain):")
    for item in result.items:
        print(f"  - {item.content}")

    recovered = any("Maya Torres" in item.content for item in result.items)
    print(f"\n  Recovers the person (Maya Torres)? {recovered}")
    print("  Vector search finds an entry node, then Cypher walks the edges back to the")
    print("  person: Maya Torres -> Iberia -> Madrid -> Spain.")
    return {"strategy": "graph (after)", "recovered": recovered}


def run_test_3_agent(driver, db, embedder):
    """Test 3: A full Strands agent that uses graph memory — and writes a new fact back."""
    print("\n" + "=" * 70)
    print("TEST 3: FULL STRANDS AGENT WITH GRAPH MEMORY")
    print("=" * 70)

    tt.init_memory(driver=driver, db=db, embedder=embedder)

    agent = Agent(
        model=MODEL,
        system_prompt=(
            "You are a travel assistant with graph memory. Use recall_graph to answer "
            "questions about people and places the user has mentioned, and remember_fact "
            "to store new durable facts the user tells you. Be concise."
        ),
        tools=[tt.recall_graph, tt.recall_semantic, tt.remember_fact],
        callback_handler=None,
    )

    # Turn 1: answer the multi-hop question from graph memory.
    print(f"\nTurn 1 (recall): {QUESTION}")
    resp = agent(QUESTION)
    answer = resp.message["content"][0]["text"]
    print(f"  Agent: {answer.strip()[:220]}")

    # Turn 2: teach the agent a new fact — it writes an edge into the graph.
    teach = "By the way, remember that Maya Torres works at Iberia — she's my contact there."
    print(f"\nTurn 2 (write): {teach}")
    resp = agent(teach)
    print(f"  Agent: {resp.message['content'][0]['text'].strip()[:220]}")

    remembered = agent.state.get("remembered_facts") or []
    print(f"\n  Facts the agent logged to agent.state this session: {len(remembered)}")
    for f in remembered:
        print(f"    {f['subject']} -[{f['relation']}]-> {f['object']}")

    recovered = "Maya" in answer
    return {"strategy": "agent + graph", "recovered": recovered, "facts_written": len(remembered)}


def run_test_4_scorecard(driver, db, embedder):
    """Test 4: Deterministic scorecard over several multi-hop questions (feeds the chart)."""
    print("\n" + "=" * 70)
    print("TEST 4: DETERMINISTIC SCORECARD — before vs after on multi-hop questions")
    print("=" * 70)

    before = gm.make_before_retriever(driver, db, embedder)
    after = gm.make_after_retriever(driver, db, embedder)

    before_hits = after_hits = 0
    print(f"\n  {'Question':<52} {'before':>7} {'after':>7}")
    print("  " + "-" * 68)
    for question, target in SCORECARD:
        b = any(target in it.content for it in before.search(query_text=question, top_k=3).items)
        a = any(target in it.content for it in after.search(query_text=question, top_k=3).items)
        before_hits += b
        after_hits += a
        print(f"  {question[:52]:<52} {'✓' if b else '✗':>7} {'✓' if a else '✗':>7}")

    total = len(SCORECARD)
    print(f"\n  Correct answers recovered — before: {before_hits}/{total} | after: {after_hits}/{total}")
    return {"total": total, "before_hits": before_hits, "after_hits": after_hits}


if __name__ == "__main__":
    print("=" * 70)
    print("  GRAPH MEMORY DEMO")
    print("  Semantic recall vs graph traversal — same graph, one multi-hop question")
    print("=" * 70)

    driver, db, embedder = gm.build()
    try:
        r1 = run_test_1_semantic(driver, db, embedder)
        r2 = run_test_2_graph(driver, db, embedder)
        r3 = run_test_3_agent(driver, db, embedder)
        r4 = run_test_4_scorecard(driver, db, embedder)

        print("\n" + "=" * 70)
        print("  COMPARISON")
        print("=" * 70)
        print(f"\n  {'Test':<42} {'Recovers multi-hop answer?':>26}")
        print("  " + "-" * 68)
        print(f"  {'Test 1 — Semantic recall (before)':<42} {str(r1['recovered']):>26}")
        print(f"  {'Test 2 — Graph recall (after)':<42} {str(r2['recovered']):>26}")
        print(f"  {'Test 3 — Strands agent + graph memory':<42} {str(r3['recovered']):>26}")
        print(
            f"\n  Scorecard (Test 4): before {r4['before_hits']}/{r4['total']} correct, "
            f"after {r4['after_hits']}/{r4['total']} correct."
        )
        print("\n  Key insight: similarity finds related pieces; only traversal connects them.")
        print("  Graph memory answers multi-hop questions that flat/semantic memory cannot.")
        print("\n  Research: https://arxiv.org/abs/2601.03236 (MAGMA — multi-graph agentic memory)")
        print("  Strands:  https://github.com/strands-agents/sdk-python")
        print("  Neo4j:    https://neo4j.com/docs/neo4j-graphrag-python/")
    finally:
        driver.close()
