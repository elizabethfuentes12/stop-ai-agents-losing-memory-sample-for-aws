"""Strands tools for the graph-memory travel assistant.

Plugging a graph store into an agent is tools plus state with Strands: the agent
calls @tool functions to write and read its graph memory, and the framework runs
the loop.

  Travel tools (what the assistant does):
  - search_flights: search flight offers (Duffel).
  - book_flight: confirm a booking.
  - best_time_to_visit: historical climate to answer "when should I go to X?".

  Memory tools (how the assistant remembers):
  - remember_fact: pass a sentence to the LLM extraction pipeline, which turns it
    into graph nodes and edges against the pinned schema (the write path).
  - recall_semantic: retrieve by chunk similarity only (no traversal).
  - recall_graph: retrieve by similarity, then traverse to the connected person.

The Neo4j driver, database, and embedder are process-level singletons created
once from the env config (a live driver is not JSON-serializable, so it does not
go in agent.state).
"""

import asyncio
import json

from strands import tool

import flights_api
import weather_api
import graph_memory as gm

_DRIVER = None
_DB = None
_EMBEDDER = None


def init_memory(driver=None, db=None, embedder=None) -> None:
    """Wire the tools to a Neo4j driver / database / embedder (call once at startup)."""
    global _DRIVER, _DB, _EMBEDDER
    if driver is not None:
        _DRIVER = driver
    if db is not None:
        _DB = db
    if embedder is not None:
        _EMBEDDER = embedder
    if _DRIVER is None:
        _DRIVER = gm.get_driver()
    if _DB is None:
        _DB = gm.ensure_database(_DRIVER)
    if _EMBEDDER is None:
        _EMBEDDER = gm.get_embedder()


def _require_memory():
    if _DRIVER is None or _DB is None or _EMBEDDER is None:
        init_memory()
    return _DRIVER, _DB, _EMBEDDER


def _run_async(coro):
    """Run a coroutine whether or not an event loop is already running.

    The tool is called from the Strands agent loop, which may or may not have a
    running loop (a .py script has none; a notebook does). If one is running, run
    the coroutine on a separate thread so we never call asyncio.run inside a loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


@tool
def remember_fact(sentence: str) -> str:
    """Record something the user said into graph memory.

    Pass one plain-English sentence stating a durable fact (who works where, which
    airline flies where, where a city is). An LLM extraction pipeline turns it into
    graph nodes and edges against a fixed schema, so it can later be traversed to
    answer multi-hop questions. Use this whenever the user reveals a fact worth
    keeping across sessions.

    Args:
        sentence: one fact in plain English, e.g. "Carlos works at Air France."
    """
    driver, db, embedder = _require_memory()
    pipeline = gm.build_pipeline(driver, db, embedder=embedder)
    _run_async(pipeline.run_async(text=sentence))
    # Keep the chunk index current for the new chunk.
    from neo4j_graphrag.indexes import create_vector_index
    create_vector_index(driver, gm.VECTOR_INDEX_NAME, label=gm.CHUNK_LABEL,
                        embedding_property="embedding", dimensions=gm.EMBED_DIM,
                        similarity_fn="cosine", neo4j_database=db)
    return f"Extracted and stored into the graph: {sentence!r}"


@tool
def recall_semantic(query: str, top_k: int = 3) -> str:
    """Recall memories by chunk similarity only (no relationship traversal).

    Returns the most similar stored sentences. Good for "what did I say about X",
    but it returns fragments and cannot follow a chain of relationships to answer a
    multi-hop question.

    Args:
        query: what you want to recall, in natural language.
        top_k: how many fragments to return (default 3).
    """
    driver, db, embedder = _require_memory()
    retriever = gm.make_semantic_retriever(driver, db, embedder)
    result = retriever.search(query_text=query, top_k=top_k)
    if not result.items:
        return "No matching memories."
    return "Most similar fragments (no traversal):\n" + "\n".join(
        f"  - {item.content}" for item in result.items
    )


@tool
def recall_graph(query: str, top_k: int = 3) -> str:
    """Recall memories by similarity, then traverse the graph to connect them.

    Finds an entry point by similarity, then walks the relationships to the
    connected person and returns the full chain. This answers multi-hop questions
    like "who do I know connected to flights to Spain?".

    Args:
        query: what you want to recall, in natural language.
        top_k: how many traversal results to return (default 3).
    """
    driver, db, embedder = _require_memory()
    retriever = gm.make_graph_retriever(driver, db, embedder)
    result = retriever.search(query_text=query, top_k=top_k)
    if not result.items:
        return "No connected memories found."
    return "Connected memories (similarity + traversal):\n" + "\n".join(
        f"  - {item.content}" for item in result.items
    )


@tool
def search_flights(origin: str, destination: str, departure_date: str,
                   cabin_class: str = "economy") -> str:
    """Search flight offers for a route on a date.

    Args:
        origin: IATA code of departure, e.g. "JFK".
        destination: IATA code of arrival, e.g. "MAD".
        departure_date: ISO date, e.g. "2026-10-15".
        cabin_class: economy, premium_economy, business, or first.
    """
    try:
        offers = flights_api.search_offers(origin, destination, departure_date,
                                           cabin_class, max_results=4)
    except Exception as exc:
        return f"Flight search failed: {exc}. Ask the user to retry."
    return json.dumps(offers, indent=1) if offers else "No offers found."


@tool
def book_flight(offer_id: str) -> str:
    """Confirm a booking for a chosen offer id from a previous search.

    Args:
        offer_id: the Duffel offer id, e.g. "off_0000B8...".
    """
    offer = flights_api.get_offer(offer_id)
    if offer is None:
        return f"Offer '{offer_id}' not found or expired. Search again."
    return json.dumps({"status": "CONFIRMED", "offer_id": offer_id,
                       "price": offer["price"], "currency": offer["currency"]})


@tool
def best_time_to_visit(city: str) -> str:
    """Answer "when should I visit X?" with historical monthly climate.

    Args:
        city: city name, e.g. "Madrid".
    """
    climate = weather_api.monthly_climate(city)
    return json.dumps(climate, indent=1) if climate else f"No climate data for '{city}'."
