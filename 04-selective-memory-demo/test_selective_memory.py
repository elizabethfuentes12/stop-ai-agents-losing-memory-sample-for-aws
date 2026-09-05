"""
Demo: What Should Your AI Agent Actually Remember? 3 Ways to Build Selective Memory

A real conversation mixes durable facts, throwaway small talk, preferences, and
events. Store everything -> expensive, dirty memory (Demo 05 shows it's also
dangerous). Store nothing -> Demo 01's amnesiac agent. The missing capability is
SELECTION: deciding what deserves to persist, in which memory type, and what to ignore.

Three mechanisms, same planted conversation, measured against the same ground truth.
The first two run on Strands' NATIVE memory framework (`MemoryManager` + a
`MemoryStore` + a `ModelExtractor`), no hand-rolled memory tools, no memory logic
in the chat agent's system prompt. The framework orchestrates; you own the
selection prompt and the store:

  A: Native MemoryManager, ONE vector store, ONE general selection prompt.
      The framework runs extraction off the turn (IntervalTrigger), distills with
      a ModelExtractor, and injects recalled memory into the model. Simplest setup.
  B: Native MemoryManager, FOUR typed stores (facts/preferences/trip_summary/
      episodes), each with its OWN specialized ModelExtractor prompt and its OWN
      vector partition, AgentCore's per-strategy partitioning, built with the
      native SDK, on the backend you pick (VECTOR_BACKEND=s3|dynamodb).
  C: Amazon Bedrock AgentCore Memory: send raw turns (create_event); the four
      built-in strategies extract, embed, and index them managed. The fully
      managed counterpart to B.

Ground truth planted in the conversation: 5 items that MUST be kept (2 facts,
2 preferences, 1 episode) and 3 decoys that must NOT (small talk, ephemeral
weather, a passing opinion).

Measured per mechanism:
  - selection recall: how many of the 5 keepers got stored (the score)
  - decoys kept out: the 3 decoys confirm a mechanism is selecting, not hoarding
  - turn latency: per-turn cost (secondary; decides whether the chat feels responsive)
  - when queryable: how long until a stored memory can be read back

All AWS resources are self-provisioned (vector indexes/tables, AgentCore memory).
Requires: OPENAI_API_KEY + AWS credentials (Bedrock Titan, S3 Vectors/DynamoDB, AgentCore).

Native Strands memory docs:
  https://strandsagents.com/docs/api/python/strands.memory.memory_manager/
"""

import os

# Bearer-token env vars would override the AWS profile, drop them before boto3 loads.
os.environ.pop("AWS_BEARER_TOKEN", None)
os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

import asyncio
import time

from dotenv import load_dotenv
load_dotenv()

from strands import Agent
# Using the OpenAI-compatible interface via the Strands SDK (not direct OpenAI usage).
from strands.models.openai import OpenAIModel
# The native Strands memory framework, MemoryManager orchestrates; you own the
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

# The chat agent has NO memory instructions, the MemoryManager owns memory. The
# system prompt is just the agent's persona and rules (its proper job).
SYSTEM_PROMPT = (
    "You are a helpful flight assistant. Help the user plan and book travel using "
    "your tools. Be concise, answer in 2-3 sentences maximum."
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
# each KEEPER word and none of the DECOY words. One readable word per item, no
# label/marker duplication.
KEEPERS = ["vegetarian", "shellfish", "layover", "1,500", "madrid"]
DECOYS = ["gorgeous", "documentary", "drizzle"]

# The selection policy, the ONLY thing you write for native memory. The
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
    "facts":        ("selective-facts",   "Extract ONLY durable FACTS about the traveler (name, home airport, "
                                          "dietary restrictions, allergies). A fact is stable and identity-level. "
                                          "DISCARD preferences, opinions, small talk, weather, and one-off events. "
                                          "If the turn has no such fact, return []."),
    "preferences":  ("selective-prefs",   "Extract ONLY stated travel PREFERENCES (cabin, seat, layover rules, "
                                          "budget limits). A preference is a standing rule the traveler wants applied. "
                                          "DISCARD facts like allergies, one-off bookings, opinions, small talk, weather. "
                                          "If the turn has no such preference, return []."),
    "trip_summary": ("selective-summary", "Maintain a one-sentence summary of the CONFIRMED current trip ONLY "
                                          "(route, airline, date once booked). Summarize only concrete trip facts the "
                                          "traveler has committed to. DISCARD small talk, weather, opinions, and anything "
                                          "not part of the booked trip. If the turn does not change the confirmed trip, return []."),
    "episodes":     ("selective-episodes", "Record ONLY a concrete completed ACTION the traveler took this turn "
                                           "(a booking made, a cancellation, a change confirmed). It must be a real event, "
                                           "not a comment, question, opinion, weather remark, or small talk. "
                                           "If the turn is not such an action, return []."),
}


def score(all_memory_text: str) -> dict:
    """Deterministic selection scoring against the planted ground truth.

    The score is selection recall (kept / kept_of): how many keepers got stored.
    We also track which decoys leaked in, as a check that a mechanism is selecting
    rather than hoarding, since a store that keeps everything would ace recall and
    still be useless. 'Stored count' is not a quality metric: more is not better.
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
    synchronous path, extraction rides on the turn. (Driving the agent through
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
    print(f"  decoys kept out:  {len(DECOYS) - result['leaked']}/{len(DECOYS)} (leaked: {result['leaked_names'] or 'none'})")
    print(f"  turn latency: {result['turn_ms']:.0f} ms/turn (incl. extraction) | queryable: on return")
    return result


def run_mechanism_a():
    """A, native MemoryManager, one vector store, one general selection prompt."""
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
    """B, native MemoryManager, four typed vector stores (one prompt + partition each)."""
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
    """C, Amazon Bedrock AgentCore Memory via the OFFICIAL Strands integration.

    Uses AgentCoreMemorySessionManager + AgentCoreMemoryConfig + RetrievalConfig
    (bedrock_agentcore.memory.integrations.strands): the agent persists each turn
    and AgentCore's managed strategies extract into long-term records server-side.
    RetrievalConfig.relevance_score is the managed noise filter (there is no prompt
    to tune here, the criteria are AWS's, not yours).
    """
    from bedrock_agentcore.memory.integrations.strands.config import (
        AgentCoreMemoryConfig, RetrievalConfig)
    from bedrock_agentcore.memory.integrations.strands.session_manager import (
        AgentCoreMemorySessionManager)

    info = acm.ensure_memory()
    memory_id = info["memory_id"]
    actor = f"sam-{int(time.time())}"          # fresh actor per run -> clean namespaces
    session = f"selective-{int(time.time())}"

    # One RetrievalConfig per strategy namespace. relevance_score filters out
    # weakly-related records (the managed equivalent of tightening a prompt).
    config = AgentCoreMemoryConfig(
        memory_id=memory_id, actor_id=actor, session_id=session,
        retrieval_config={
            "/facts/{actorId}/":                    RetrievalConfig(top_k=10, relevance_score=0.3),
            "/preferences/{actorId}/":              RetrievalConfig(top_k=10, relevance_score=0.3),
            "/summaries/{actorId}/{sessionId}/":    RetrievalConfig(top_k=5,  relevance_score=0.5),
            "/episodes/{actorId}/{sessionId}/":     RetrievalConfig(top_k=5,  relevance_score=0.5),
        },
    )

    turn_ms = []
    with AgentCoreMemorySessionManager(config, region_name=acm.REGION) as sm:
        agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT,
                      tools=[search_flights, book_flight, best_time_to_visit],
                      session_manager=sm)
        for turn in CONVERSATION:
            t0 = time.time()
            agent(turn)                            # turn persisted by the session manager
            turn_ms.append((time.time() - t0) * 1000)

    # Extraction is async and managed. Poll a strategy namespace until records
    # appear, measuring how long until the memory is queryable.
    lag = acm.wait_for_extraction(memory_id, info["strategy_ids"]["facts"], actor,
                                  "dietary restrictions travel preferences", timeout_s=420)

    NAMESPACES = {
        "facts":       f"/facts/{actor}/",
        "preferences": f"/preferences/{actor}/",
        "tripSummary": f"/summaries/{actor}/{session}/",
        "episodes":    f"/episodes/{actor}/{session}/",
    }

    def read_all():
        """Read every strategy namespace (score-filtered) and score it."""
        text, per = "", {}
        for name, ns in NAMESPACES.items():
            recs = acm.retrieve_by_namespace(memory_id, ns,
                                             "traveler profile trip preferences",
                                             top_k=10, min_score=0.3)
            per[name] = len(recs)
            text += " ".join(recs) + " "
        return text, per

    # Extraction across the four strategies completes at different times. The
    # waiter (in agentcore_memory, using a proper wait, not a bare sleep) re-reads
    # and re-scores until the score stabilizes, so we score what AgentCore actually
    # extracts, not a premature first read.
    best, stored_text, per_ns = acm.wait_until_stable(read_all, score)

    result = best
    result.update({
        "avail_s": lag if lag is not None else -1.0,
        "turn_ms": sum(turn_ms) / len(turn_ms),
        "records": per_ns,
    })
    print(f"  records per strategy: {per_ns}  (count is not a quality signal)")
    print(f"  selection recall: {result['kept']}/{result['kept_of']} kept (missed: {result['missed'] or 'none'})")
    print(f"  decoys kept out:  {len(DECOYS) - result['leaked']}/{len(DECOYS)} (leaked: {result['leaked_names'] or 'none'})")
    print(f"  turn latency: {result['turn_ms']:.0f} ms/turn (create_event only) | queryable after: {lag:.0f}s" if lag else "  extraction timed out")
    return result


if __name__ == "__main__":
    start = time.time()
    print("Mechanism A, native MemoryManager, single store + general extractor:")
    ra = run_mechanism_a()
    print("\nMechanism B, native MemoryManager, 4 typed stores + typed extractors:")
    rb = run_mechanism_b()
    print("\nMechanism C, AgentCore Memory (4 built-in strategies):")
    rc = run_mechanism_c()

    print(f"\n{'Mechanism':<32} {'Recall':>8} {'DecoysOut':>10} {'Turn':>10} {'Queryable':>14}")
    print(f"{'A: native MemoryManager (1)':<32} {str(ra['kept']) + '/' + str(ra['kept_of']):>8} {str(len(DECOYS)-ra['leaked']) + '/' + str(len(DECOYS)):>10} {ra['turn_ms']:>7.0f} ms {'on return':>14}")
    print(f"{'B: native MemoryManager (4 typed)':<32} {str(rb['kept']) + '/' + str(rb['kept_of']):>8} {str(len(DECOYS)-rb['leaked']) + '/' + str(len(DECOYS)):>10} {rb['turn_ms']:>7.0f} ms {'on return':>14}")
    print(f"{'C: AgentCore managed':<32} {str(rc['kept']) + '/' + str(rc['kept_of']):>8} {str(len(DECOYS)-rc['leaked']) + '/' + str(len(DECOYS)):>10} {rc['turn_ms']:>7.0f} ms {str(round(rc['avail_s'])) + ' s':>14}")
    print("\nRecall = keepers stored (the score). DecoysOut = decoys correctly left out (confirms selecting, not hoarding).")
    print(f"wall time: {time.time() - start:.0f}s")
