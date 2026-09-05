"""Strands tools for the graph-memory travel assistant.

Plugging an external graph store into an agent is *just tools + state* with Strands:
the agent calls ``@tool`` functions to write and read its graph memory, and the
framework runs the loop.

Six tools split into two groups:

  Travel tools (what the assistant does):
  - ``search_flights``  , search live flight offers.
  - ``book_flight``     , confirm a booking and store what was learned in the graph.
  - ``best_time_to_visit``, historical climate to answer "when should I go to X?".

  Memory tools (how the assistant remembers):
  - ``remember_fact``   , record a new fact as a graph edge (write path).
  - ``recall_semantic`` , retrieve by vector similarity only.
  - ``recall_graph``    , retrieve by similarity + graph traversal (multi-hop).

The Neo4j driver, database name, and embedder are process-level singletons created
once from the env config (a live driver is not JSON-serializable, so it does not go
in ``agent.state``).
"""

import json

from strands import tool, ToolContext

import flights_api
import weather_api
import graph_memory as gm

# ── Process-level connections (created once, reused across tool calls) ───────
_DRIVER = None
_DB = None
_EMBEDDER = None


def init_memory(driver=None, db=None, embedder=None) -> None:
    """Wire the tools to a Neo4j driver / database / embedder.

    Call this once at startup. If any argument is omitted, it is created from the
    env config via ``graph_memory``. The test script and notebook pass in the objects
    they already built so everything shares one connection.
    """
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
    """Return (driver, db, embedder), initializing from env if needed."""
    if _DRIVER is None or _DB is None or _EMBEDDER is None:
        init_memory()
    return _DRIVER, _DB, _EMBEDDER


@tool(context=True)
def remember_fact(subject: str, relation: str, obj: str, tool_context: ToolContext) -> str:
    """Record a fact in graph memory as a relationship between two entities.

    Use this whenever the user reveals a durable fact worth remembering, who they
    met, where a place is, what style something is. Stored as a graph edge so it can
    later be traversed to answer multi-hop questions.

    Args:
        subject: The entity the fact starts from (e.g. "Maya Torres").
        relation: The relationship type. One of: WORKS_AT, MEMBER_OF, FLIES_TO, IN_COUNTRY.
        obj: The entity the fact points to (e.g. "Iberia").
    """
    driver, db, _ = _require_memory()

    relation = relation.strip().upper().replace(" ", "_")
    if relation not in gm.ALLOWED_RELATIONS:
        return (
            f"Unknown relation '{relation}'. Allowed relations: "
            f"{', '.join(sorted(gm.ALLOWED_RELATIONS))}."
        )

    # Relationship type is validated above, so this interpolation is safe. Node
    # names are passed as parameters (never interpolated).
    with driver.session(database=db) as session:
        session.run(
            f"MERGE (a:{gm.NODE_LABEL} {{name: $s}}) "
            f"MERGE (b:{gm.NODE_LABEL} {{name: $o}}) "
            f"MERGE (a)-[:`{relation}`]->(b)",
            s=subject, o=obj,
        )

    # Also log the write into agent.state, a serializable trace of what the agent
    # believes it has stored (the graph itself is the source of truth).
    log = tool_context.agent.state.get("remembered_facts") or []
    log.append({"subject": subject, "relation": relation, "object": obj})
    tool_context.agent.state.set("remembered_facts", log)

    return f"Remembered: {subject} -[{relation}]-> {obj}"


@tool
def recall_semantic(query: str, top_k: int = 3) -> str:
    """Recall memories by semantic similarity only (no relationship traversal).

    Returns the individually most-similar memory entries. Good for "what do I know
    about X" but blind to how memories connect, it cannot follow a chain of
    relationships to answer a multi-hop question.

    Args:
        query: What you want to recall, in natural language.
        top_k: How many memory entries to return (default 3).
    """
    driver, db, embedder = _require_memory()
    retriever = gm.make_semantic_retriever(driver, db, embedder)
    result = retriever.search(query_text=query, top_k=top_k)
    if not result.items:
        return "No matching memories."
    lines = [f"  - {item.content}" for item in result.items]
    return "Most similar memories (no traversal):\n" + "\n".join(lines)


@tool
def recall_graph(query: str, top_k: int = 3) -> str:
    """Recall memories by similarity, then traverse the graph to connect them.

    Finds an entry point by vector similarity, then walks the relationships to the
    connected person and returns the full chain. This is what answers multi-hop
    questions like "who do I know connected to flights to Spain?".

    Args:
        query: What you want to recall, in natural language.
        top_k: How many traversal results to return (default 3).
    """
    driver, db, embedder = _require_memory()
    retriever = gm.make_graph_retriever(driver, db, embedder)
    result = retriever.search(query_text=query, top_k=top_k)
    if not result.items:
        return "No connected memories found."
    lines = [f"  - {item.content}" for item in result.items]
    return "Connected memories (similarity + graph traversal):\n" + "\n".join(lines)


# ── Travel tools ──────────────────────────────────────────────────────────────

@tool
def search_flights(origin: str, destination: str, departure_date: str,
                   cabin_class: str = "economy") -> str:
    """Search live flight offers when the user wants to fly somewhere on a date.

    Use this when the user:
    - asks for flights between two airports ("find me a flight to Madrid")
    - wants options or prices for a route on a date
    - asks what's available for a destination

    Args:
        origin: IATA airport code of departure, e.g. "JFK".
        destination: IATA airport code of arrival, e.g. "MAD".
        departure_date: ISO date, e.g. "2026-10-15".
        cabin_class: One of: economy, premium_economy, business, first.

    Returns:
        JSON list of offers sorted by price (offer_id, price, currency, cabin,
        slices with carrier and stops).
    """
    offers = flights_api.search_offers(origin, destination, departure_date,
                                       cabin_class, max_results=4)
    return json.dumps(offers, indent=1)


@tool(context=True)
def book_flight(offer_id: str, tool_context: ToolContext) -> str:
    """Confirm a booking and store what was learned in graph memory.

    Use this when the user picks a specific offer and wants to book it. The booking
    records the chosen airline and destination as graph edges so the knowledge graph
    grows with the user's travel history.

    Args:
        offer_id: The offer id from a previous search result.

    Returns:
        Booking confirmation with the edges written to the graph.
    """
    offer = flights_api.get_offer(offer_id)
    if offer is None:
        return f"Offer '{offer_id}' not found or expired. Search again for a fresh offer."

    carriers = sorted({
        seg["carrier"]
        for sl in offer["slices"]
        for seg in sl["segments"]
        if seg["carrier"]
    })
    destinations = sorted({sl["destination"] for sl in offer["slices"]})

    driver, db, _ = _require_memory()
    edges_written = []

    with driver.session(database=db) as session:
        for carrier in carriers:
            session.run(
                f"MERGE (a:{gm.NODE_LABEL} {{name: $s}}) "
                f"MERGE (b:{gm.NODE_LABEL} {{name: $o}}) "
                f"MERGE (a)-[:BOOKED_WITH]->(b)",
                s="User", o=carrier,
            )
            edges_written.append(f"User -[BOOKED_WITH]-> {carrier}")

        for dest in destinations:
            session.run(
                f"MERGE (a:{gm.NODE_LABEL} {{name: $s}}) "
                f"MERGE (b:{gm.NODE_LABEL} {{name: $o}}) "
                f"MERGE (a)-[:TRAVELED_TO]->(b)",
                s="User", o=dest,
            )
            edges_written.append(f"User -[TRAVELED_TO]-> {dest}")

    log = tool_context.agent.state.get("remembered_facts") or []
    for edge in edges_written:
        parts = edge.split(" -[")
        subj, rest = parts[0], parts[1]
        rel, obj = rest.rstrip("]->").split("]-> ")
        log.append({"subject": subj, "relation": rel, "object": obj})
    tool_context.agent.state.set("remembered_facts", log)

    return json.dumps({
        "status": "CONFIRMED",
        "offer_id": offer_id,
        "cabin": offer["cabin"],
        "price": offer["price"],
        "currency": offer["currency"],
        "edges_written_to_graph": edges_written,
    })


@tool
def best_time_to_visit(city: str) -> str:
    """Answer 'when should I visit X?' with historical climate data by month.

    Use this when the user asks about the best season to visit a city, whether it
    will be hot, rainy, or cold on a planned trip, or for general travel timing advice.

    Args:
        city: City name, e.g. "Madrid", "Tokyo", "New York".
    """
    return weather_api.best_time_to_visit(city)
