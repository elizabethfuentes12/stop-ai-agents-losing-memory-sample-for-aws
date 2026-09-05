"""Tools for the vector-memory demo, remember by meaning, not by key.

A returning traveler has accumulated memories over past conversations (notes,
preferences, episodes). This demo's tools let a Strands agent write those
memories and retrieve them THREE ways, so the tests can measure the difference:

  - recall_by_key     , Demo 01's move: exact key lookup (needs to know the key)
  - recall_semantic   , vector search (needs only the MEANING)

The semantic tool queries whichever vector backend the test wires in (FAISS
in-process or Amazon S3 Vectors), same embeddings, same memories, so the
backends are interchangeable from the agent's point of view.

Tool docstrings follow the research-backed pattern (ToolLLM/AgentTuning):
first sentence says WHEN to use the tool; the system prompt never re-describes them.
"""

import json

from strands import tool

import memory_stores as ms

# Wired by the test/notebook at startup: which stores the tools operate on.
KV = None          # memory_stores.KeyValueStore
VECTOR = None      # FaissStore or S3VectorStore (same query interface)


def init_stores(kv, vector) -> None:
    """Wire the tools to the stores built by the test, call once at startup."""
    global KV, VECTOR
    KV, VECTOR = kv, vector


def store_note(key: str, note: str) -> None:
    """Write one note into BOTH stores (embedded once). Plain function shared by
    the remember tool and the tests, so seeding and live writes take the same path."""
    KV.put(key, note)
    vector = ms.embed(note)
    if isinstance(VECTOR, ms.S3VectorStore):
        VECTOR.put(key, note, vector)
    else:
        VECTOR.put(note, vector)


@tool
def remember_note(key: str, note: str) -> str:
    """Store a durable note about the user in long-term memory.

    Use this tool when the user:
    - shares a fact worth keeping ("I'm vegetarian", "my passport expires in May")
    - asks you to remember something for later trips

    Args:
        key: Short snake_case identifier, e.g. "dietary_notes".
        note: The full note text to remember.

    Returns:
        Confirmation naming the key. The note lands in BOTH the key-value store
        and the vector index (embedded once at write time).
    """
    store_note(key, note)
    return f"Remembered under '{key}'."


@tool
def recall_by_key(key: str) -> str:
    """Look up one memory when you KNOW its exact key.

    Use this tool when the question maps to a known identifier -
    "what's my preferred cabin?" -> key "preferred_cabin".

    Args:
        key: The exact key used at write time, e.g. "dietary_notes".

    Returns:
        The stored note, or a not-found message (keys must match exactly).
    """
    text = KV.get(key)
    return text or f"No memory stored under '{key}'."


@tool
def recall_semantic(question: str, top_k: int = 3) -> str:
    """Retrieve the memories most relevant to a question BY MEANING.

    Use this tool when the user asks something their history could answer but
    no key is obvious, "what should I avoid eating on this trip?",
    "anything I said about long layovers?". The stored notes do not need to
    share any words with the question.

    Args:
        question: The user's question, as asked.
        top_k: How many memories to return (default 3).

    Returns:
        The top-k matching notes with similarity scores, best first.
    """
    hits = VECTOR.query(ms.embed(question), top_k)
    if not hits:
        return "No relevant memories found."
    return json.dumps([{"note": text, "score": round(score, 3)} for text, score in hits], indent=1)
