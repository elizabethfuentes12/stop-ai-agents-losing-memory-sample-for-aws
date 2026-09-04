"""
Interactive flight assistant — SELECTIVE MEMORY, four typed stores (Mechanism B).

Chat with an agent whose memory is partitioned by TYPE: facts, preferences,
trip_summary, episodes. Each type has its own vector store and its own
`ModelExtractor` selection prompt, all orchestrated by one native Strands
`MemoryManager`. After every turn you see what landed in each type, so you watch
selection AND partitioning happen live. This is AgentCore's per-strategy split,
built with the native SDK; the backend (S3 Vectors or DynamoDB) is your choice.

Try these, one per line, and watch which TYPE each lands in:
  "Hi, I'm Sam. I'm vegetarian with a shellfish allergy."   -> facts
  "I refuse overnight layovers and keep fares under $1,500." -> preferences
  "I just booked the Iberia flight JFK to Madrid Oct 10."    -> episodes / trip_summary
  "Gorgeous weather today!"                                  -> ignored (decoy)
  "What do you know about me?"                               -> recalls across types

Commands:  /memory  (show everything stored, by type)   /quit
Backend:   VECTOR_BACKEND=s3 (default) or dynamodb  — set in .env.
"""

import os
import sys

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

# One selection prompt per memory type. Each keeps ITS type and rejects the rest.
JSON_CONTRACT = ' Return ONLY a JSON array of {"content": string}, or [] if none.'
TYPED = {
    "facts":        ("selective-facts",   "Extract durable FACTS (identity, home airport, allergies). "
                                          "NOT preferences, NOT opinions, NOT small talk, NOT weather."),
    "preferences":  ("selective-prefs",   "Extract stated PREFERENCES (cabin, seats, layovers, budget). "
                                          "NOT facts like allergies, NOT events, NOT small talk."),
    "trip_summary": ("selective-summary", "If the turn advances the CURRENT trip, return one summary sentence. "
                                          "Otherwise return []."),
    "episodes":     ("selective-episodes", "If the turn is a notable EVENT (a booking, a cancellation), "
                                           "describe it in one sentence. Otherwise return []."),
}

RECALL_QUERY = "traveler facts preferences bookings and trip details"


def show_memory(stores: dict) -> None:
    """Print what's stored per type, so you watch selection AND partitioning."""
    print("\n  🧠 memory by type (count is transparency, not a quality score):")
    for name, store in stores.items():
        hits = store._backend.query(ms.embed(RECALL_QUERY), top_k=20)
        if hits:
            for text, _ in hits:
                print(f"     [{name}] {text}")
        else:
            print(f"     [{name}] (empty)")
    print()


def main() -> None:
    print(f"Provisioning four typed vector stores ({ms.VECTOR_BACKEND})...")
    stores = {}
    for mem_type, (partition, rules) in TYPED.items():
        stores[mem_type] = VectorMemoryStore(
            name=mem_type,
            partition=partition,
            extraction=ExtractionConfig(
                trigger=[IntervalTrigger(turns=1)],
                extractor=ModelExtractor(model=MODEL, system_prompt=rules + JSON_CONTRACT),
            ),
        )
    # Start clean so you watch each type fill from zero (in production you would not clear).
    for s in stores.values():
        s.clear()

    agent = Agent(
        model=MODEL,
        system_prompt="You are a helpful flight assistant. Be concise, 2-3 sentences max.",
        memory_manager=MemoryManager(stores=list(stores.values())),
        callback_handler=None,
    )

    print("Flight assistant ready (Mechanism B: four typed stores). Type a message, /memory, or /quit.\n")
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
            show_memory(stores)
            continue

        reply = agent(user)   # each type's ModelExtractor runs off the turn, flushed within the call
        print(f"\nassistant> {reply}")
        show_memory(stores)   # watch which type the turn fed

    print("Bye. Memory persists in the vector stores for next time.")


if __name__ == "__main__":
    main()
