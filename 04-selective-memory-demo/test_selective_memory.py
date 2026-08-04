"""
Demo: What Should Your AI Agent Actually Remember? 3 Ways to Build Selective Memory

A real conversation mixes durable facts, throwaway small talk, preferences, and
events. Store everything -> expensive, dirty memory (Demo 05 shows it's also
dangerous). Store nothing -> Demo 01's amnesiac agent. The missing capability is
SELECTION: deciding what deserves to persist, in which memory type, and what to ignore.

Three mechanisms, same planted conversation, measured against the same ground truth:

  A — Agent-managed: the conversational agent files memories itself via tools
      (selection inline, stores to agent.state).
  B — Own extractor: four specialized LLM prompts run OFF the conversation path,
      one per memory type, writing survivors to Amazon S3 Vectors (one index per
      type). This is AgentCore's pipeline, built by hand.
  C — AgentCore Memory: send raw turns (create_event); the four built-in
      strategies (semantic/userPreference/summary/episodic) extract, embed, and
      index them managed. Retrieval via retrieve_memory_records.

Ground truth planted in the conversation: 5 items that MUST be kept (2 facts,
2 preferences, 1 episode) and 3 decoys that must NOT (small talk, ephemeral
weather, a passing opinion).

Measured per mechanism: keep/discard score, write-path latency, availability
lag (when is the memory queryable?), and extractor token cost.

All AWS resources are self-provisioned (S3 Vectors indexes, AgentCore memory).
Requires: OPENAI_API_KEY + AWS credentials (Bedrock Titan, S3 Vectors, AgentCore).
"""

import os

# Bearer-token env vars would override the AWS profile — drop them before boto3 loads.
os.environ.pop("AWS_BEARER_TOKEN", None)
os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

import json
import time

from dotenv import load_dotenv
load_dotenv()

from strands import Agent
# Using OpenAI-compatible interface via Strands SDK (not direct OpenAI usage)
from strands.models.openai import OpenAIModel

import agentcore_memory as acm
import extractor
from tools import core_memory_read, core_memory_write, core_memory_update, core_memory_list

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

SYSTEM_PROMPT_TYPED_MEMORY = (
    "You are a flight assistant with self-managed memory. "
    "Organize what you learn into these memory sections:\n"
    "- 'facts': durable facts about the user's world (name, home airport, allergies)\n"
    "- 'preferences': likes/dislikes they reveal (cabin, layovers, budget)\n"
    "- 'trip_summary': one rolling summary of the trip being planned\n"
    "- 'episodes': notable events, one entry per event (bookings, cancellations)\n"
    "Only store what is durable — ignore small talk and passing remarks. "
    "Be concise — answer in 2-3 sentences maximum."
)

# ── The planted conversation (same input for all three mechanisms) ───────────
# 5 keepers: 2 facts (F), 2 preferences (P), 1 episode (E). 3 decoys (D).
CONVERSATION = [
    "Hi! I'm Sam. I'm vegetarian with a severe shellfish allergy.",              # F1, F2
    "Gorgeous weather out here today, hope your day is going great!",            # D1 small talk
    "For flights: I refuse overnight layovers, and I keep fares under $1,500.",  # P1, P2
    "I watched a documentary about airplanes last night, it was okay I guess.",  # D2 opinion
    "I just booked the Iberia flight JFK to Madrid for October 10th!",           # E1 episode
    "Apparently it might drizzle here later this afternoon.",                    # D3 ephemeral
]

# Deterministic ground truth: markers that must appear in kept memories, and
# decoy markers that must NOT appear anywhere.
KEEPERS = {
    "F1 vegetarian": "vegetarian",
    "F2 shellfish allergy": "shellfish",
    "P1 no overnight layovers": "layover",
    "P2 fare ceiling": "1,500",
    "E1 Madrid booking": "madrid",
}
DECOYS = {
    "D1 small talk": "gorgeous",
    "D2 documentary opinion": "documentary",
    "D3 drizzle": "drizzle",
}


def score(all_memory_text: str) -> dict:
    """Deterministic keep/discard scoring against the planted ground truth."""
    text = all_memory_text.lower()
    kept = [name for name, marker in KEEPERS.items() if marker in text]
    leaked = [name for name, marker in DECOYS.items() if marker in text]
    return {"kept": len(kept), "kept_of": len(KEEPERS),
            "missed": sorted(set(KEEPERS) - set(kept)),
            "leaked": len(leaked), "leaked_names": leaked}


def run_mechanism_a():
    """A — the conversational agent selects inline, via memory tools."""
    agent = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT_TYPED_MEMORY,
        tools=[core_memory_read, core_memory_write, core_memory_update, core_memory_list],
        callback_handler=None,
    )
    turn_ms = []
    for turn in CONVERSATION:
        start = time.perf_counter()
        agent(turn)
        turn_ms.append((time.perf_counter() - start) * 1000)

    memory = agent.state.get("core_memory") or {}
    result = score(json.dumps(memory))
    result.update({
        "avail_s": 0.0,                            # queryable the moment the turn ends
        "turn_ms": sum(turn_ms) / len(turn_ms),    # selection happens INSIDE the turn
        "sections": sorted(memory),
    })
    print(f"  sections: {result['sections']}")
    print(f"  kept {result['kept']}/{result['kept_of']} (missed: {result['missed'] or 'none'}) | decoys leaked: {result['leaked']}")
    print(f"  avg turn latency (selection inline): {result['turn_ms']:.0f} ms")
    return result


def run_mechanism_b():
    """B — own extraction pipeline -> S3 Vectors, one index per memory type."""
    typed = extractor.TypedVectorMemory()   # self-provisions the 4 indexes
    typed.clear()

    total_tokens = 0
    pipeline_ms = []
    for n, turn in enumerate(CONVERSATION):
        kept, cost = extractor.extract(turn)          # off the conversation path
        total_tokens += cost["tokens"]
        write_ms = 0.0
        for mem_type, items in kept.items():
            write_ms += typed.write(mem_type, items, f"turn{n}")
        pipeline_ms.append(cost["ms"] + write_ms)

    counts = typed.counts()
    stored_text = ""
    for mem_type in extractor.INDEXES:
        for text, _ in typed.query(mem_type, "traveler profile and trip", top_k=10):
            stored_text += " " + text
    result = score(stored_text)
    result.update({
        "avail_s": sum(pipeline_ms) / 1000 / len(CONVERSATION),  # per-turn lag
        "turn_ms": 0.0,                                          # conversation path untouched
        "tokens": total_tokens,
        "counts": counts,
    })
    print(f"  stored per type: {counts}")
    print(f"  kept {result['kept']}/{result['kept_of']} (missed: {result['missed'] or 'none'}) | decoys leaked: {result['leaked']}")
    print(f"  extractor cost: {total_tokens:,} tokens | avg availability lag: {result['avail_s']:.1f}s per turn")
    return result


def run_mechanism_c():
    """C — AgentCore Memory: raw turns in, managed strategies extract."""
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
    print(f"  records per strategy: {per_strategy}")
    print(f"  kept {result['kept']}/{result['kept_of']} (missed: {result['missed'] or 'none'}) | decoys leaked: {result['leaked']}")
    print(f"  create_event avg: {result['turn_ms']:.0f} ms | extraction lag: {lag:.0f}s" if lag else "  extraction timed out")
    return result


if __name__ == "__main__":
    start = time.time()
    print("Mechanism A — agent-managed (tools inline):")
    ra = run_mechanism_a()
    print("\nMechanism B — own extractor -> S3 Vectors (4 indexes):")
    rb = run_mechanism_b()
    print("\nMechanism C — AgentCore Memory (4 built-in strategies):")
    rc = run_mechanism_c()

    print(f"\n{'Mechanism':<30} {'Kept':>6} {'Leaked':>7} {'Turn overhead':>14} {'Available after':>16}")
    print(f"{'A — agent tools (inline)':<30} {str(ra['kept']) + '/' + str(ra['kept_of']):>6} {ra['leaked']:>7} {ra['turn_ms']:>11.0f} ms {'immediately':>16}")
    print(f"{'B — own extractor + S3V':<30} {str(rb['kept']) + '/' + str(rb['kept_of']):>6} {rb['leaked']:>7} {'0 (off-path)':>14} {rb['avail_s']:>13.1f} s")
    print(f"{'C — AgentCore managed':<30} {str(rc['kept']) + '/' + str(rc['kept_of']):>6} {rc['leaked']:>7} {rc['turn_ms']:>11.0f} ms {rc['avail_s']:>13.0f} s")
    print(f"\nB extractor cost: {rb['tokens']:,} tokens for {len(CONVERSATION)} turns")
    print(f"wall time: {time.time() - start:.0f}s")
