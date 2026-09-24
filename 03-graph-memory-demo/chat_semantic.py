"""
Interactive travel assistant, SEMANTIC MEMORY only.

Memory is stored as flat vectors. The agent can recall individual facts by
similarity, but cannot follow relationships between them to answer multi-hop
questions. Compare with chat_graph.py to see the difference.

Pre-loaded memory (what the agent already knows):
  Maya Torres   works at  Iberia
  Iberia        member of Oneworld
  Iberia        flies to  Madrid
  Madrid        located in Spain

Try these to explore the limits:
  "Who do I know connected to flights to Spain?"    ← fails, no traversal
  "What airlines do you know about?"                ← works, direct similarity
  "Find me a flight from JFK to MAD on 2026-10-15"
  "When is the best time to visit Madrid?"
  "Remember that Maya Torres is my travel agent"

Commands: /memory  /quit
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

import graph_memory as gm
import travel_tools as tt
from graph_memory_store import GraphMemoryStore
import asyncio

print("Connecting to Neo4j and building memory (LLM extraction)...")
driver, db, embedder = asyncio.run(gm.build())

MODEL = OpenAIModel(model_id="gpt-4o-mini")

# Same GraphMemoryStore and MemoryManager as chat_graph.py, but mode="semantic":
# recall is similarity over chunks only, no traversal. This is the contrast, the
# agent can recall individual facts but cannot follow relationships to connect them.
SELECTION_PROMPT = (
    "Extract durable facts worth keeping about the traveler and their network. "
    'Return ONLY a JSON array of {"content": string}, or [] if nothing is worth keeping.'
)
semantic_store = GraphMemoryStore(
    name="traveler_semantic", driver=driver, db=db, embedder=embedder, mode="semantic",
    extraction=ExtractionConfig(
        trigger=[IntervalTrigger(turns=1)],
        extractor=ModelExtractor(model=MODEL, system_prompt=SELECTION_PROMPT),
    ),
)

agent = Agent(
    model=MODEL,
    system_prompt="You are a personal travel assistant. Be concise: at most 3 sentences.",
    tools=[tt.search_flights, tt.best_time_to_visit],
    memory_manager=MemoryManager(stores=[semantic_store]),
    callback_handler=None,
)

print("\n" + "=" * 60)
print("  TRAVEL ASSISTANT, semantic memory (MemoryManager)")
print("  search_memory (similarity only) · auto-extraction · injection")
print("=" * 60)
print(__doc__)

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
            with driver.session(database=db) as session:
                rows = session.run(
                    "MATCH (c:Chunk) RETURN c.text AS text ORDER BY text"
                ).data()
            print(f"\nStored chunks ({len(rows)}):")
            for row in rows:
                print(f"  - {row['text']}")
            if not rows:
                print("  (empty)")
            continue

        if user_input.lower() == "/help":
            print("Commands: /memory  /quit")
            print("Try: 'Who do I know connected to Spain?' to see semantic memory's limit.")
            continue

        resp = agent(user_input)
        print(f"\nAgent: {resp.message['content'][0]['text'].strip()}\n")

finally:
    driver.close()
    print("\nGoodbye.")
