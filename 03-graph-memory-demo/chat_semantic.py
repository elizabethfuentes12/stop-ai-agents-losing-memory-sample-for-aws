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
        tt.best_time_to_visit,
        tt.recall_semantic,
        tt.remember_fact,
    ],
    callback_handler=None,
)

print("\n" + "=" * 60)
print("  TRAVEL ASSISTANT, semantic memory")
print("  recall_semantic · remember_fact · search_flights")
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
            remembered = agent.state.get("remembered_facts") or []
            if remembered:
                print("Facts stored this session:")
                for f in remembered:
                    print(f"  {f['subject']} -[{f['relation']}]-> {f['object']}")
            else:
                print("No facts stored this session yet.")
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
