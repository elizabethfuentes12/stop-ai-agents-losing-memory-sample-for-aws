"""
Interactive travel assistant, GRAPH MEMORY.

Memory is stored as a knowledge graph. The agent can traverse relationships
between people, airlines, cities, and countries to answer multi-hop questions.
Every booking writes new edges into the graph. Compare with chat_semantic.py.

Pre-loaded memory (what the agent already knows):
  Maya Torres   works at  Iberia
  Iberia        member of Oneworld
  Iberia        flies to  Madrid
  Madrid        located in Spain

Try these to see graph memory in action:
  "Who do I know connected to flights to Spain?"    ← traverses the graph!
  "Find me a flight from JFK to MAD on 2026-10-15"
  "Book [offer_id from search results]"             ← writes to graph
  "When is the best time to visit Madrid?"
  "Remember that Carlos works at Air France"
  "Who do I know connected to France?"              ← uses edge you just wrote

Commands: /memory  /graph  /quit
"""

import os
import sys

os.environ["OTEL_SDK_DISABLED"] = "true"

from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not set. Add it to your .env file.")
    sys.exit(1)

from strands import Agent
# Using OpenAI-compatible interface via Strands SDK (not direct OpenAI usage)
from strands.models.openai import OpenAIModel

import graph_memory as gm
import travel_tools as tt

print("Connecting to Neo4j and loading memory...")
driver, db, embedder = gm.build()
tt.init_memory(driver=driver, db=db, embedder=embedder)

MODEL = OpenAIModel(model_id="gpt-4o-mini")

agent = Agent(
    model=MODEL,
    system_prompt=(
        "You are a personal travel assistant with access to the user's travel memory. "
        "Always store new facts the user shares. Be concise."
    ),
    tools=[
        tt.search_flights,
        tt.book_flight,
        tt.best_time_to_visit,
        tt.recall_graph,
        tt.recall_semantic,
        tt.remember_fact,
    ],
    callback_handler=None,
)

print("\n" + "=" * 60)
print("  TRAVEL ASSISTANT, graph memory")
print("  recall_graph · recall_semantic · remember_fact")
print("  search_flights · book_flight · best_time_to_visit")
print("=" * 60)
print(__doc__)


def _show_graph():
    """Print all nodes and edges currently in the graph."""
    with driver.session(database=db) as session:
        nodes = session.run(f"MATCH (n:{gm.NODE_LABEL}) RETURN n.name AS name, n.type AS type ORDER BY n.type, n.name").data()
        edges = session.run(
            f"MATCH (a:{gm.NODE_LABEL})-[r]->(b:{gm.NODE_LABEL}) "
            f"RETURN a.name AS src, type(r) AS rel, b.name AS dst ORDER BY a.name"
        ).data()

    print(f"\nGraph, {len(nodes)} nodes, {len(edges)} edges:")
    for e in edges:
        print(f"  ({e['src']}) -[{e['rel']}]-> ({e['dst']})")
    if not edges:
        print("  (empty)")
    print()


try:
    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            break

        if user_input.lower() == "/memory":
            remembered = agent.state.get("remembered_facts") or []
            if remembered:
                print("Facts stored this session:")
                for f in remembered:
                    print(f"  {f['subject']} -[{f['relation']}]-> {f['object']}")
            else:
                print("No facts stored this session yet.")
            continue

        if user_input.lower() == "/graph":
            _show_graph()
            continue

        if user_input.lower() == "/help":
            print("Commands: /memory (session log)  /graph (full Neo4j graph)  /quit")
            print("Try: 'Who do I know connected to Spain?' to see graph traversal.")
            continue

        resp = agent(user_input)
        print(f"\nAgent: {resp.message['content'][0]['text'].strip()}\n")

finally:
    driver.close()
    print("\nGoodbye.")
