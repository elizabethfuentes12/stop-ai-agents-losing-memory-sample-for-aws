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
from strands.memory import MemoryManager, ModelExtractor, ExtractionConfig, IntervalTrigger

import asyncio

import graph_memory as gm
import travel_tools as tt
from graph_memory_store import GraphMemoryStore

print("Connecting to Neo4j and building memory (LLM extraction)...")
driver, db, embedder = asyncio.run(gm.build())

MODEL = OpenAIModel(model_id="gpt-4o-mini")

# Graph memory as a native Strands MemoryStore, wired through the MemoryManager,
# exactly like the vector store in Demo 04. The manager registers a search tool,
# runs extraction after each turn (facts the user reveals become graph nodes/edges),
# and injects recalled memories into the model. mode="graph" recalls by traversal.
SELECTION_PROMPT = (
    "Extract durable facts worth keeping about the traveler and their network: "
    "who knows whom, who works where, which airline flies where, where a city is. "
    'Return ONLY a JSON array of {"content": string}, or [] if nothing is worth keeping.'
)
graph_store = GraphMemoryStore(
    name="traveler_graph", driver=driver, db=db, embedder=embedder, mode="graph",
    extraction=ExtractionConfig(
        trigger=[IntervalTrigger(turns=1)],
        extractor=ModelExtractor(model=MODEL, system_prompt=SELECTION_PROMPT),
    ),
)

agent = Agent(
    model=MODEL,
    system_prompt="You are a personal travel assistant. Be concise: at most 3 sentences.",
    tools=[tt.search_flights, tt.book_flight, tt.best_time_to_visit],
    memory_manager=MemoryManager(stores=[graph_store]),
    callback_handler=None,
)

print("\n" + "=" * 60)
print("  TRAVEL ASSISTANT, graph memory (MemoryManager)")
print("  search_memory (traversal) · auto-extraction · injection")
print("  search_flights · book_flight · best_time_to_visit")
print("=" * 60)
print(__doc__)


def _show_graph():
    """Print the entities and edges the LLM pipeline extracted into the graph."""
    with driver.session(database=db) as session:
        nodes = session.run(
            "MATCH (n:__Entity__) RETURN n.name AS name, head(labels(n)) AS type "
            "ORDER BY type, name"
        ).data()
        edges = session.run(
            "MATCH (a:__Entity__)-[r]->(b:__Entity__) "
            "RETURN a.name AS src, type(r) AS rel, b.name AS dst ORDER BY a.name"
        ).data()

    print(f"\nGraph, {len(nodes)} entities, {len(edges)} edges:")
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
            _show_graph()
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
