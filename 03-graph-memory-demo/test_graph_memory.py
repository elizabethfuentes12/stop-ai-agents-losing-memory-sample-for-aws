"""
Demo: Graph Memory, Reasoning Over Relationships, Not Just Similarity

Based on research in graph-structured agent memory:
  - GAAMA: Graph Augmented Associative Memory for Agents (https://arxiv.org/abs/2603.27910), 2026
  - MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents (https://arxiv.org/abs/2601.03236), 2026
  - GRAVITY: Architecture-Agnostic Structured Anchoring for Long-Horizon Conversational Memory (https://arxiv.org/abs/2605.01688), 2026

Semantic memory (Demo 02) retrieves by *similarity* but cannot reason over
*relationships*. A multi-hop question needs to find an entry point by similarity and
then TRAVERSE the graph to the answer. This demo contrasts the two on the same graph:

  Test 1: Semantic recall (before)   , pure vector similarity, misses the multi-hop chain
  Test 2: Graph recall (after)       , vector similarity + graph traversal, recovers it
  Test 3: Full Strands agent         , the agent uses graph memory to answer, and writes
                                        a new fact back into the graph (the harness)
  Test 4: Deterministic scorecard    , 4 multi-hop questions, before vs after (feeds the chart)

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

# The multi-hop question the demo is built around.
QUESTION = "Who do I know that's connected to flights to Spain?"

# Deterministic scorecard: multi-hop questions whose correct answer is the PERSON,
# reachable only by following relationships. Same known graph → reproducible scores.
SCORECARD = [
    ("Who do I know that's connected to flights to Spain?", "Maya Torres"),
    ("Who do I know connected to an airline that flies to Madrid?", "Maya Torres"),
    ("Who works at the Oneworld airline I know?", "Maya Torres"),
    ("Which person is linked to airlines in Spain?", "Maya Torres"),
]


def run_test_1_semantic(driver, db, embedder):
    """Test 1: Agent with semantic-only recall. Vector similarity alone cannot answer
    a multi-hop question, it surfaces related pieces but never connects them to a person."""
    print("\n" + "=" * 70)
    print("TEST 1: AGENT WITH SEMANTIC RECALL, vector similarity only, no traversal")
    print("=" * 70)

    tt.init_memory(driver=driver, db=db, embedder=embedder)

    agent = Agent(
        model=MODEL,
        system_prompt="You are a personal travel assistant with access to the user's travel memory. Be concise.",
        tools=[tt.recall_semantic],
        callback_handler=None,
    )

    print(f"\nQuestion: {QUESTION}\n")
    resp = agent(QUESTION)
    answer = resp.message["content"][0]["text"]
    print(f"  Agent: {answer.strip()[:220]}")

    recovered = "Maya Torres" in answer
    print(f"\n  Recovers the person (Maya Torres)? {recovered}")
    print("  Similarity surfaces Iberia / Madrid / Spain as separate pieces but cannot")
    print("  connect them to a person, no notion of a relationship.")
    return {"strategy": "semantic recall", "recovered": recovered}


def run_test_2_graph(driver, db, embedder):
    """Test 2: Agent with graph recall. Similarity finds an entry node, then Cypher
    traversal walks the relationships back to the person."""
    print("\n" + "=" * 70)
    print("TEST 2: AGENT WITH GRAPH RECALL, vector similarity + graph traversal")
    print("=" * 70)

    tt.init_memory(driver=driver, db=db, embedder=embedder)

    agent = Agent(
        model=MODEL,
        system_prompt="You are a personal travel assistant with access to the user's travel memory. Be concise.",
        tools=[tt.recall_graph],
        callback_handler=None,
    )

    print(f"\nQuestion: {QUESTION}\n")
    resp = agent(QUESTION)
    answer = resp.message["content"][0]["text"]
    print(f"  Agent: {answer.strip()[:220]}")

    recovered = "Maya Torres" in answer
    print(f"\n  Recovers the person (Maya Torres)? {recovered}")
    print("  Graph traversal walks Maya Torres -> Iberia -> Madrid -> Spain and returns")
    print("  the person, the answer similarity alone could not reach.")
    return {"strategy": "graph recall", "recovered": recovered}


def run_test_3_agent(driver, db, embedder):
    """Test 3: A full Strands agent that uses graph memory, and writes a new fact back."""
    print("\n" + "=" * 70)
    print("TEST 3: FULL STRANDS AGENT WITH GRAPH MEMORY")
    print("=" * 70)

    tt.init_memory(driver=driver, db=db, embedder=embedder)

    agent = Agent(
        model=MODEL,
        system_prompt="You are a personal travel assistant with access to the user's travel memory. Always store new facts the user shares. Be concise.",
        tools=[tt.search_flights, tt.book_flight, tt.best_time_to_visit,
               tt.recall_graph, tt.recall_semantic, tt.remember_fact],
        callback_handler=None,
    )

    # Turn 1: answer the multi-hop question from graph memory.
    print(f"\nTurn 1 (recall): {QUESTION}")
    resp = agent(QUESTION)
    answer = resp.message["content"][0]["text"]
    print(f"  Agent: {answer.strip()[:220]}")

    # Turn 2: teach the agent a new fact, it writes an edge into the graph.
    teach = "By the way, remember that Maya Torres works at Iberia, she's my contact there."
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
    """Test 4: Deterministic scorecard, semantic vs graph on 4 multi-hop questions.
    Uses retrievers directly (not agents) so scoring is deterministic and feeds the chart."""
    print("\n" + "=" * 70)
    print("TEST 4: SCORECARD, semantic retrieval vs graph traversal, 4 multi-hop questions")
    print("=" * 70)

    semantic_retriever = gm.make_semantic_retriever(driver, db, embedder)
    graph_retriever = gm.make_graph_retriever(driver, db, embedder)

    semantic_hits = graph_hits = 0
    print(f"\n  {'Question':<52} {'semantic':>9} {'graph':>7}")
    print("  " + "-" * 70)
    for question, target in SCORECARD:
        s = any(target in it.content for it in semantic_retriever.search(query_text=question, top_k=3).items)
        g = any(target in it.content for it in graph_retriever.search(query_text=question, top_k=3).items)
        semantic_hits += s
        graph_hits += g
        print(f"  {question[:52]:<52} {'✓' if s else '✗':>9} {'✓' if g else '✗':>7}")

    total = len(SCORECARD)
    print(f"\n  Correct answers recovered, semantic: {semantic_hits}/{total} | graph: {graph_hits}/{total}")
    return {"total": total, "before_hits": semantic_hits, "after_hits": graph_hits}


if __name__ == "__main__":
    print("=" * 70)
    print("  GRAPH MEMORY DEMO")
    print("  Semantic recall vs graph traversal, same graph, one multi-hop question")
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
        print(f"  {'Test 1, Agent: semantic recall only':<42} {str(r1['recovered']):>26}")
        print(f"  {'Test 2, Agent: graph traversal':<42} {str(r2['recovered']):>26}")
        print(f"  {'Test 3, Agent: graph memory + write':<42} {str(r3['recovered']):>26}")
        print(
            f"\n  Scorecard (Test 4): semantic {r4['before_hits']}/{r4['total']} correct, "
            f"graph {r4['after_hits']}/{r4['total']} correct."
        )
        print("\n  Key insight: similarity finds related pieces; only traversal connects them.")
        print("  Graph memory answers multi-hop questions that flat/semantic memory cannot.")
        print("\n  Research: https://arxiv.org/abs/2601.03236 (MAGMA, multi-graph agentic memory)")
        print("  Strands:  https://github.com/strands-agents/sdk-python")
        print("  Neo4j:    https://neo4j.com/docs/neo4j-graphrag-python/")
    finally:
        driver.close()
