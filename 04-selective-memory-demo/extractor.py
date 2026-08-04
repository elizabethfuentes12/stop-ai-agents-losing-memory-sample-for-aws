"""Mechanism B — a hand-built extraction pipeline: AgentCore's strategies, manual.

This is what "build AgentCore Memory yourself" actually means. For each raw
conversation turn, FOUR specialized extractor prompts run — one per memory
type — and each decides independently: extract structured items, or answer
NOTHING. What survives is embedded (Titan V2) and written to Amazon S3 Vectors,
one index per memory type — the same partitioning AgentCore gives you managed.

  | Extractor      | Keeps                                   | Index               |
  |----------------|------------------------------------------|---------------------|
  | facts          | durable facts about the user's world     | selective-facts     |
  | preferences    | likes / dislikes / rules the user states | selective-prefs     |
  | summary        | one rolling summary of the conversation  | selective-summary   |
  | episodes       | notable events (bookings, cancellations) | selective-episodes  |

The conversational agent never sees any of this — extraction runs OFF the
conversation path (after the turn), exactly like AgentCore's async pipeline.
The demo measures what that costs when you own it: extra LLM tokens per turn
and write-path latency, against AgentCore's managed version (mechanism C).

The extractor model is deliberately independent from the chat model — any
Strands-supported provider works; a small cheap model is the realistic choice.
"""

import json
import os
import time

# Using OpenAI-compatible interface via Strands SDK (not direct OpenAI usage)
from openai import OpenAI

import memory_stores as ms

EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "gpt-4o-mini")

# One S3 Vectors index per memory type — AgentCore's per-strategy partitioning, manual.
INDEXES = {
    "facts": "selective-facts",
    "preferences": "selective-prefs",
    "summary": "selective-summary",
    "episodes": "selective-episodes",
}

# The selection criteria live HERE, in four specialized prompts — this is the
# part AgentCore ships built-in and this mechanism lets you customize freely.
EXTRACTION_PROMPTS = {
    "facts": (
        "Extract durable FACTS about the user's world from this conversation turn: "
        "identity, home airport, allergies, memberships — objective attributes true "
        "beyond this trip. NOT preferences (wants/likes belong elsewhere), NOT opinions, "
        "NOT small talk or today's weather. "
        "Reply with a raw JSON list of short fact strings (no markdown fences), "
        "or exactly NOTHING if none: "
    ),
    "preferences": (
        "Extract the user's stated PREFERENCES from this conversation turn: what they "
        "want or avoid by choice (cabin, seats, layovers, budget rules). NOT objective "
        "facts like allergies or home airport, NOT events, NOT small talk. "
        "Reply with a raw JSON list of short preference strings (no markdown fences), "
        "or exactly NOTHING if none: "
    ),
    "summary": (
        "If this turn advances the CURRENT TRIP being planned, reply with ONE sentence "
        "summarizing the trip's state so far. If the turn is small talk or unrelated, "
        "reply exactly NOTHING: "
    ),
    "episodes": (
        "If this turn contains a notable EVENT (a booking made, a cancellation, a "
        "payment), describe it in one sentence. Otherwise reply exactly NOTHING: "
    ),
}

_client = None


def _llm():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def extract(turn_text: str) -> tuple[dict[str, list[str]], dict]:
    """Run the four extractors over one raw turn.

    Returns ({memory_type: [items]}, cost) where cost has total extractor
    tokens and wall milliseconds — the price of owning the pipeline.
    """
    kept: dict[str, list[str]] = {}
    tokens = 0
    start = time.perf_counter()
    for mem_type, prompt in EXTRACTION_PROMPTS.items():
        resp = _llm().chat.completions.create(
            model=EXTRACTOR_MODEL,
            messages=[{"role": "user", "content": prompt + "\n\nTURN: " + turn_text}],
            temperature=0,
        )
        tokens += resp.usage.total_tokens
        answer = resp.choices[0].message.content.strip()
        # Models often wrap JSON in markdown fences despite instructions — strip them.
        if answer.startswith("```"):
            answer = answer.strip("`").removeprefix("json").strip()
        if answer == "NOTHING" or not answer:
            continue
        try:
            items = json.loads(answer)
            if isinstance(items, str):
                items = [items]
        except json.JSONDecodeError:
            items = [answer]  # summary/episodes reply with a bare sentence
        items = [i for i in items if i and i != "NOTHING"]
        if items:
            kept[mem_type] = items
    elapsed_ms = (time.perf_counter() - start) * 1000
    return kept, {"tokens": tokens, "ms": elapsed_ms}


class TypedVectorMemory:
    """Four S3 Vectors indexes, one per memory type — self-provisioned."""

    def __init__(self):
        self.stores = {mem_type: ms.S3VectorStore(index=index_name)
                       for mem_type, index_name in INDEXES.items()}

    def write(self, mem_type: str, items: list[str], turn_id: str) -> float:
        """Embed + store extracted items in the type's index. Returns write ms."""
        start = time.perf_counter()
        for n, item in enumerate(items):
            self.stores[mem_type].put(f"{turn_id}-{mem_type}-{n}", item, ms.embed(item))
        return (time.perf_counter() - start) * 1000

    def query(self, mem_type: str, question: str, top_k: int = 3) -> list[tuple[str, float]]:
        return self.stores[mem_type].query(ms.embed(question), top_k)

    def counts(self) -> dict[str, int]:
        return {t: s.count() for t, s in self.stores.items()}

    def clear(self) -> int:
        return sum(s.clear() for s in self.stores.values())
