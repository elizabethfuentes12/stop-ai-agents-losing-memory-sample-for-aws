"""
Demo: What Should Your AI Agent Actually Remember? 3 Ways to Build Selective Memory

A real conversation mixes durable facts, throwaway small talk, preferences, and
events. Store everything -> expensive, dirty memory (Demo 05 shows it's also
dangerous). Store nothing -> Demo 01's amnesiac agent. The missing capability is
SELECTION: deciding what deserves to persist, in which memory type, and what to ignore.

Three mechanisms, same planted conversation, measured against the same ground truth.
The first two run on Strands' NATIVE memory framework (`MemoryManager` + a
`MemoryStore` + a `ModelExtractor`) — no hand-rolled memory tools, no memory logic
in the chat agent's system prompt. The framework orchestrates; you own the
selection prompt and the store:

  A — Native MemoryManager, ONE vector store, ONE general selection prompt.
      The framework runs extraction off the turn (IntervalTrigger), distills with
      a ModelExtractor, and injects recalled memory into the model. Simplest setup.
  B — Native MemoryManager, FOUR typed stores (facts/preferences/trip_summary/
      episodes), each with its OWN specialized ModelExtractor prompt and its OWN
      vector partition — AgentCore's per-strategy partitioning, built with the
      native SDK, on the backend you pick (VECTOR_BACKEND=s3|dynamodb).
  C — Amazon Bedrock AgentCore Memory: send raw turns (create_event); the four
      built-in strategies extract, embed, and index them managed. The fully
      managed counterpart to B.

Ground truth planted in the conversation: 5 items that MUST be kept (2 facts,
2 preferences, 1 episode) and 3 decoys that must NOT (small talk, ephemeral
weather, a passing opinion).

Measured per mechanism, on the axes that actually matter for agent memory
(PrecisionMemBench / Future AGI, 2026), not vanity counts:
  - selection recall: did the keepers get stored?
  - noise isolation:  did the decoys stay out? (storing everything is NOT a win)
  - conversation fluidity: per-turn latency (secondary, but it decides whether a
    chat feels responsive)
  - availability lag: how long until a stored memory is queryable?

All AWS resources are self-provisioned (vector indexes/tables, AgentCore memory).
Requires: OPENAI_API_KEY + AWS credentials (Bedrock Titan, S3 Vectors/DynamoDB, AgentCore).

Native Strands memory docs:
  https://strandsagents.com/docs/api/python/strands.memory.memory_manager/
"""

import os

# Bearer-token env vars would override the AWS profile — drop them before boto3 loads.
os.environ.pop("AWS_BEARER_TOKEN", None)
os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

import asyncio
import time

from dotenv import load_dotenv
load_dotenv()

from strands import Agent
# Using the OpenAI-compatible interface via the Strands SDK (not direct OpenAI usage).
from strands.models.openai import OpenAIModel
# The native Strands memory framework — MemoryManager orchestrates; you own the
# selection prompt (ModelExtractor) and the store (VectorMemoryStore).
from strands.memory import MemoryManager, ModelExtractor, ExtractionConfig, IntervalTrigger

import agentcore_memory as acm
from vector_memory_store import VectorMemoryStore
from tools import search_flights, book_flight, best_time_to_visit

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not set. Get your API key from https://platform.openai.com/api-keys "
        "and add OPENAI_API_KEY=your-key to a .env file."
    )

MODEL = OpenAIModel(model_id="gpt-4o-mini")

# To run on Amazon Bedrock instead (no OpenAI key; uses your AWS credentials),
# comment the MODEL line above and uncomment these:
# from strands.models import BedrockModel
# MODEL = BedrockModel(model_id="openai.gpt-oss-120b-1:0", region_name="us-west-2")

# The chat agent has NO memory instructions — the MemoryManager owns memory. The
# system prompt is just the agent's persona and rules (its proper job).
SYSTEM_PROMPT = (
    "You are a helpful flight assistant. Help the user plan and book travel using "
    "your tools. Be concise — answer in 2-3 sentences maximum."
)

# ── The planted conversation (same input for all three mechanisms) ───────────
# 5 keepers: 2 facts (F), 2 preferences (P), 1 episode (E). 3 decoys (D).
CONVERSATION = [
    "Hi! I'm Sam. I'm vegetarian with a severe shellfish allergy.",              # keep: vegetarian + shellfish allergy
    "Gorgeous weather out here today, hope your day is going great!",            # decoy: small talk
    "For flights: I refuse overnight layovers, and I keep fares under $1,500.",  # keep: no layovers + $1,500 budget
    "I watched a documentary about airplanes last night, it was okay I guess.",  # decoy: passing opinion
    "I just booked the Iberia flight JFK to Madrid for October 10th!",           # keep: Madrid booking
    "Apparently it might drizzle here later this afternoon.",                    # decoy: ephemeral weather
]

# Query used to read every store back for scoring (semantic recall).
RECALL_QUERY = "traveler dietary restrictions travel preferences bookings and trip"

# Deterministic ground truth: a memory that stores everything relevant contains
# each KEEPER word and none of the DECOY words. One readable word per item — no
# label/marker duplication.
KEEPERS = ["vegetarian", "shellfish", "layover", "1,500", "madrid"]
DECOYS = ["gorgeous", "documentary", "drizzle"]

# The selection policy — the ONLY thing you write for native memory. The
# ModelExtractor runs this prompt off the turn; returning [] is "discard".
# Mechanism A uses one general prompt; Mechanism B uses one specialized prompt
# per memory type (the same criteria AgentCore ships built-in, here yours to tune).
JSON_CONTRACT = ' Return ONLY a JSON array of {"content": string}, or [] if none.'

GENERAL_SELECTION_PROMPT = (
    "You extract durable memories worth keeping about a traveler, from a transcript. "
    "KEEP durable facts (name, allergies), stated preferences (cabin, layovers, budget), "
    "and notable events (a booking, a cancellation). DISCARD small talk, weather, "
    "passing opinions, questions." + JSON_CONTRACT
)

# {memory_type: (vector_partition, selection_prompt)}
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


def score(all_memory_text: str) -> dict:
    """Deterministic selection scoring against the planted ground truth.

    Two axes, because memory quality is not one number (PrecisionMemBench, 2026):
      - selection recall: did the keepers make it in?  (kept / kept_of)
      - noise isolation:  did the decoys stay out?      (leaked)
    A system that stores everything scores perfect recall but terrible isolation,
    so we score both. 'Stored count' is deliberately NOT a quality metric here:
    more is not better, keeping the right things and dropping the rest is.
    """
    text = all_memory_text.lower()
    kept = [w for w in KEEPERS if w in text]
    leaked = [w for w in DECOYS if w in text]
    return {"kept": len(kept), "kept_of": len(KEEPERS),
            "missed": [w for w in KEEPERS if w not in kept],
            "leaked": len(leaked), "leaked_names": leaked}


async def _read_all(manager) -> str:
    """Concatenate every stored memory's text via the manager's native search."""
    entries = await manager.search(RECALL_QUERY)
    return " ".join(e.content for e in entries)


def _run_native(manager, stores, label_counts) -> dict:
    """Shared driver for the two native-MemoryManager mechanisms (A and B).

    The synchronous `Agent(...)` entry point awaits the manager's flush after each
    invocation, so background extraction is drained before the turn returns: the
    extractor model call is included in `turn_ms`, and memories are queryable the
    moment the turn ends (availability lag ~0). This is the trade-off of the
    synchronous path — extraction rides on the turn. (Driving the agent through
    your own event loop lets extraction stay in the background; you would then
    await `manager.flush()` at a shutdown boundary.)
    """
    agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT,
                  memory_manager=manager, callback_handler=None)

    turn_ms = []
    for turn in CONVERSATION:
        start = time.perf_counter()
        agent(turn)                                   # extraction runs + is flushed within the turn
        turn_ms.append((time.perf_counter() - start) * 1000)

    stored_text = asyncio.run(_read_all(manager))
    result = score(stored_text)
    counts = label_counts()
    result.update({
        "turn_ms": sum(turn_ms) / len(turn_ms),
        "avail_s": 0.0,                               # queryable the moment the turn ends
        "counts": counts,
    })
    print(f"  stored per partition: {counts}  (count is not a quality signal)")
    print(f"  selection recall: {result['kept']}/{result['kept_of']} kept (missed: {result['missed'] or 'none'})")
    print(f"  noise isolation:  {len(DECOYS) - result['leaked']}/{len(DECOYS)} decoys rejected (leaked: {result['leaked_names'] or 'none'})")
    print(f"  conversation fluidity: {result['turn_ms']:.0f} ms/turn (incl. extraction) | available: immediately")
    return result


def run_mechanism_a():
    """A — native MemoryManager, one vector store, one general selection prompt."""
    store = VectorMemoryStore(
        name="traveler_memory",
        partition="selective-single",
        extraction=ExtractionConfig(
            trigger=[IntervalTrigger(turns=1)],                          # extract every turn, off the turn
            extractor=ModelExtractor(model=MODEL, system_prompt=GENERAL_SELECTION_PROMPT),
        ),
    )
    store.clear()
    manager = MemoryManager(stores=[store])
    return _run_native(manager, [store], lambda: {store.name: store.count()})


def run_mechanism_b():
    """B — native MemoryManager, four typed vector stores (one prompt + partition each)."""
    stores = {
        mem_type: VectorMemoryStore(
            name=mem_type,
            partition=partition,
            extraction=ExtractionConfig(
                trigger=[IntervalTrigger(turns=1)],
                extractor=ModelExtractor(model=MODEL, system_prompt=prompt + JSON_CONTRACT),
            ),
        )
        for mem_type, (partition, prompt) in TYPED.items()
    }
    for s in stores.values():
        s.clear()
    manager = MemoryManager(stores=list(stores.values()))
    return _run_native(manager, list(stores.values()),
                       lambda: {name: s.count() for name, s in stores.items()})


def run_mechanism_c():
    """C — AgentCore Memory: raw turns in, managed strategies extract (unchanged)."""
    info = acm.ensure_memory()
    memory_id, sids = info["memory_id"], info["strategy_ids"]
    actor = f"sam-{int(time.time())}"      # fresh actor per run -> clean namespaces
    session = "selective-run"

    turn_ms = []
    for turn in CONVERSATION:
        turn_ms.append(acm.send_turn(memory_id, actor, session, "USER", turn))

    # Extraction is async — poll until facts appear, and report the real lag.
    lag = acm.wait_for_extraction(memory_id, sids["facts"], actor,
                                  "dietary restrictions travel preferences", timeout_s=420)

    stored_text = ""
    per_strategy = {}
    for name, sid in sids.items():
        session_scoped = name in ("tripSummary", "episodes")
        records = acm.retrieve(memory_id, sid, actor, "traveler profile and trip",
                               session_id=session if session_scoped else None, top_k=10)
        per_strategy[name] = len(records)
        stored_text += " ".join(records) + " "

    result = score(stored_text)
    result.update({
        "avail_s": lag if lag is not None else -1.0,
        "turn_ms": sum(turn_ms) / len(turn_ms),   # create_event only — no extraction inline
        "records": per_strategy,
    })
    print(f"  records per strategy: {per_strategy}  (count is not a quality signal)")
    print(f"  selection recall: {result['kept']}/{result['kept_of']} kept (missed: {result['missed'] or 'none'})")
    print(f"  noise isolation:  {len(DECOYS) - result['leaked']}/{len(DECOYS)} decoys rejected (leaked: {result['leaked_names'] or 'none'})")
    print(f"  conversation fluidity: {result['turn_ms']:.0f} ms/turn (create_event only) | availability lag: {lag:.0f}s" if lag else "  extraction timed out")
    return result


if __name__ == "__main__":
    start = time.time()
    print("Mechanism A — native MemoryManager, single store + general extractor:")
    ra = run_mechanism_a()
    print("\nMechanism B — native MemoryManager, 4 typed stores + typed extractors:")
    rb = run_mechanism_b()
    print("\nMechanism C — AgentCore Memory (4 built-in strategies):")
    rc = run_mechanism_c()

    print(f"\n{'Mechanism':<32} {'Recall':>8} {'NoiseIso':>9} {'Turn':>10} {'Avail after':>14}")
    print(f"{'A: native MemoryManager (1)':<32} {str(ra['kept']) + '/' + str(ra['kept_of']):>8} {str(len(DECOYS)-ra['leaked']) + '/' + str(len(DECOYS)):>9} {ra['turn_ms']:>7.0f} ms {'immediately':>14}")
    print(f"{'B: native MemoryManager (4 typed)':<32} {str(rb['kept']) + '/' + str(rb['kept_of']):>8} {str(len(DECOYS)-rb['leaked']) + '/' + str(len(DECOYS)):>9} {rb['turn_ms']:>7.0f} ms {'immediately':>14}")
    print(f"{'C: AgentCore managed':<32} {str(rc['kept']) + '/' + str(rc['kept_of']):>8} {str(len(DECOYS)-rc['leaked']) + '/' + str(len(DECOYS)):>9} {rc['turn_ms']:>7.0f} ms {str(round(rc['avail_s'])) + ' s':>14}")
    print("\nRecall = keepers stored; NoiseIso = decoys rejected (higher is better on both).")
    print("Storing everything would score perfect Recall and zero NoiseIso, so both matter.")
    print(f"wall time: {time.time() - start:.0f}s")
