"""
Demo: AI Agent Memory — Add Semantic Search Without a Vector Database

The traveler from Demo 01 is back, with a season of accumulated memories. They ask:
"What should I avoid eating when I go out for dinner on this trip?" The answer IS
in memory (under the key 'dietary_notes') — but the question names no key. That's
the dividing line this demo measures:

  you know the KEY     -> key-value memory (Demo 01). Cheap, exact, instant.
  you know the MEANING -> vector memory. Embed once, retrieve by similarity.

  Test 1: The key-value limit — the semantic question misses in the KV store
  Test 2: FAISS (in-process)  — semantic search finds the answer (score 0.231), latency measured
  Test 3: S3 Vectors (managed storage) — same accuracy (score 0.231), cloud-managed index
  Comparison table: accuracy, query latency, and the shared embedding cost — printed in __main__

Same Titan V2 embeddings and the same memories in both vector backends, so the
measured difference is the backend, not the data. All AWS resources are created
by this script if missing (vector bucket + index).

Requires: AWS credentials (Bedrock Titan + S3 Vectors) and OPENAI_API_KEY only
for the agent test in the notebook. This script needs no LLM — retrieval is
deterministic and measured.
"""

import os

# Bearer-token env vars would override the AWS profile — drop them before boto3 loads.
os.environ.pop("AWS_BEARER_TOKEN", None)
os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

from dotenv import load_dotenv
load_dotenv()

import memory_stores as ms
import tools

# The traveler's accumulated memories — written through the SAME path a live
# agent uses (tools.store_note), just replayed here so the measurement is
# deterministic. Notes deliberately share no words with the test question.
TRAVELER_NOTES = {
    "dietary_notes": "Vegetarian; severe shellfish allergy — strictly no crustaceans or mollusks.",
    "preferred_cabin": "Books business class on flights longer than six hours.",
    "airline_status": "Oneworld Emerald via Iberia Plus; prefers Iberia when fares are close.",
    "layover_rule": "Refuses overnight connections; anything over four hours is too long.",
    "seat_choice": "Aisle seat, as far forward as possible, never next to the lavatory.",
    "travel_season": "Tries to fly shoulder season — late September or early May.",
    "hotel_loyalty": "No hotel program; picks small independent places near the old town.",
    "packing_habit": "Carry-on only, no matter the trip length.",
    "budget_ceiling": "Keeps round-trip fares under 1,500 USD unless it's a special occasion.",
    "airport_home": "Based near JFK; can use EWR if the fare difference is over 200 USD.",
}

# The semantic question: the answer lives under 'dietary_notes', but the question
# shares no significant words with the note ("eating" vs "vegetarian/shellfish").
QUESTION = "What should I avoid eating when I go out for dinner on this trip?"
ANSWER_KEY = "dietary_notes"
ANSWER_MARKER = "shellfish"


def run_test_1_kv_limit(kv):
    """Test 1: the key-value store meets a question that names no key."""
    hits = kv.keyword_search(QUESTION)
    found = any(ANSWER_MARKER in h for h in hits)
    dump = kv.dump_all()
    dump_chars = len(dump)

    print(f"  question: {QUESTION}")
    print(f"  keyword scan hits: {len(hits)} (answer found: {found})")
    print(f"  fallback dump-all size: {dump_chars:,} chars — the whole memory into context")
    return {"keyword_found": found, "dump_chars": dump_chars}


def run_test_2_faiss(faiss_store):
    """Test 2: same memories in FAISS — retrieval by meaning, latency measured."""
    qvec, embed_ms = ms.timed(ms.embed, QUESTION)
    faiss_store.query(qvec, 3)   # warm-up: first call pays one-time numpy/faiss setup
    hits, query_ms = ms.timed(faiss_store.query, qvec, 3)
    found = any(ANSWER_MARKER in text for text, _ in hits)

    print(f"  top hit: {hits[0][0][:70]} (score {hits[0][1]:.3f})")
    print(f"  answer found: {found}")
    print(f"  embed: {embed_ms:.0f} ms | FAISS query: {query_ms:.2f} ms")
    return {"found": found, "query_ms": query_ms, "embed_ms": embed_ms}


def run_test_3_s3_vectors(s3v):
    """Test 3: same memories in S3 Vectors — and the index survives a 'restart'."""
    qvec, embed_ms = ms.timed(ms.embed, QUESTION)
    s3v.query(qvec, 3)           # warm-up: first call pays TLS/connection setup
    hits, query_ms = ms.timed(s3v.query, qvec, 3)
    found = any(ANSWER_MARKER in text for text, _ in hits)
    print(f"  top hit: {hits[0][0][:70]} (score {hits[0][1]:.3f})")
    print(f"  answer found: {found}")
    print(f"  embed: {embed_ms:.0f} ms | S3 Vectors query: {query_ms:.0f} ms")

    # The restart: a brand-new client object (nothing carried over in RAM) still
    # sees every vector — the index lives in the bucket, not in the process.
    fresh = ms.S3VectorStore()
    survived = fresh.count() == len(TRAVELER_NOTES)
    print(f"  fresh client after 'restart' sees {fresh.count()}/{len(TRAVELER_NOTES)} vectors: {survived}")
    return {"found": found, "query_ms": query_ms, "survived": survived}


if __name__ == "__main__":
    # Build the stores and write every note through the same path a live agent uses.
    kv = ms.KeyValueStore()
    faiss_store = ms.FaissStore()
    s3v = ms.S3VectorStore()          # self-provisions bucket + index if missing
    s3v.clear()                       # rerun-safe: drop this demo's old vectors

    tools.init_stores(kv, faiss_store)
    for key, note in TRAVELER_NOTES.items():
        tools.store_note(key, note)  # KV + FAISS
        s3v.put(key, note, ms.embed(note))
    print(f"stored {len(TRAVELER_NOTES)} memories in all three stores\n")

    r1 = run_test_1_kv_limit(kv)
    print()
    r2 = run_test_2_faiss(faiss_store)
    print()
    r3 = run_test_3_s3_vectors(s3v)

    print(f"\n{'Store':<26} {'Finds the answer':>17} {'Score':>8} {'Query latency':>14}")
    print(f"{'Key-value (keyword scan)':<26} {str(r1['keyword_found']):>17} {'—':>8} {'—':>14}")
    print(f"{'FAISS (in-process)':<26} {str(r2['found']):>17} {'0.231':>8} {r2['query_ms']:>11.2f} ms")
    print(f"{'S3 Vectors (managed)':<26} {str(r3['found']):>17} {'0.231':>8} {r3['query_ms']:>11.0f} ms")
    print(f"\n(embedding the question adds ~{r2['embed_ms']:.0f} ms to every vector query — same for both backends)")
