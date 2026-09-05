"""
Interactive flight assistant, SELECTIVE MEMORY, one store (Mechanism A).

Chat with an agent that remembers across turns using Strands' native
`MemoryManager` with ONE vector-backed store and one general selection prompt.
After every turn you see the memory grow: the `ModelExtractor` distills what is
worth keeping and stores it; small talk is discarded.

What the agent keeps: durable facts, stated preferences, and notable events.
What it ignores: small talk, weather, passing opinions.

Try these, one per line, and watch the memory panel after each:
  "Hi, I'm Sam. I'm vegetarian with a shellfish allergy."   -> stored (facts)
  "Gorgeous weather today!"                                 -> ignored (decoy)
  "I refuse overnight layovers and keep fares under $1,500." -> stored (preferences)
  "What do you know about me?"                              -> recalls from memory

Commands:  /memory  (show everything stored)   /quit
Backend:   VECTOR_BACKEND=s3 (default) or dynamodb , set in .env.
"""

import os
import sys
import asyncio

os.environ["OTEL_SDK_DISABLED"] = "true"
# Bearer-token env vars would override the AWS profile; drop them before boto3 loads.
os.environ.pop("AWS_BEARER_TOKEN", None)
os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not set. Add it to .env, or switch to Bedrock "
          "(uncomment the BedrockModel line below).")
    sys.exit(1)

from strands import Agent
# Using the OpenAI-compatible interface via the Strands SDK (not direct OpenAI usage).
from strands.models.openai import OpenAIModel
from strands.memory import MemoryManager, ModelExtractor, ExtractionConfig, IntervalTrigger

from vector_memory_store import VectorMemoryStore
import memory_stores as ms
from tools import search_flights, book_flight, best_time_to_visit

MODEL = OpenAIModel(model_id="gpt-4o-mini")
# To use Amazon Bedrock instead of OpenAI (no OpenAI key needed), comment the line
# above and uncomment these (embeddings always use Bedrock Titan regardless):
# from strands.models import BedrockModel
# MODEL = BedrockModel(model_id="openai.gpt-oss-120b-1:0", region_name="us-west-2")

SELECTION_PROMPT = (
    "You extract durable memories worth keeping about a traveler, from a transcript. "
    "KEEP durable facts (name, allergies), stated preferences (cabin, layovers, budget), "
    "and notable events (a booking, a cancellation). DISCARD small talk, weather, "
    "passing opinions, questions. "
    'Return ONLY a JSON array of {"content": string}, or [] if nothing is worth keeping.'
)

RECALL_QUERY = "traveler facts preferences bookings and trip details"


def show_memory(store: VectorMemoryStore) -> None:
    """Print everything currently stored, so you watch memory accumulate."""
    print(f"\n  🧠 memory now holds {store.count()} item(s):")
    hits = store._backend.query(ms.embed(RECALL_QUERY), top_k=20)
    if not hits:
        print("     (empty, nothing worth keeping yet)")
    for text, _ in hits:
        print(f"     - {text}")
    print()


def main() -> None:
    print(f"Provisioning the vector store ({ms.VECTOR_BACKEND})...")
    store = VectorMemoryStore(
        name="traveler_memory",
        partition="selective-single",
        extraction=ExtractionConfig(
            trigger=[IntervalTrigger(turns=1)],   # extract after every turn
            extractor=ModelExtractor(model=MODEL, system_prompt=SELECTION_PROMPT),
        ),
    )
    agent = Agent(
        model=MODEL,
        system_prompt="You are a helpful flight assistant. Be concise, 2-3 sentences max.",
        memory_manager=MemoryManager(stores=[store]),
        callback_handler=None,
    )

    # Start from an empty store so you can watch memory fill from zero. In
    # production you would NOT clear it, the whole point is that memory persists
    # across sessions (that survival is what the notebook's Step 4 demonstrates).
    store.clear()

    print("Flight assistant ready (Mechanism A: one store). Type a message, /memory, or /quit.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/memory":
            show_memory(store)
            continue

        reply = agent(user)   # extraction runs and is flushed within this call
        print(f"\nassistant> {reply}")
        show_memory(store)    # watch what the turn added

    print("Bye. Memory persists in the vector store for next time.")


if __name__ == "__main__":
    main()
