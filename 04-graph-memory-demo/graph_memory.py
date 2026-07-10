"""Graph memory layer for the graph-memory demo.

Semantic memory (Demo 03) retrieves by *similarity* but cannot reason over
*relationships*. A multi-hop question — "Who did I meet that's connected to
boutique hotels in Japan?" — needs a different move: find an entry point by
vector similarity, then **traverse the graph** to the answer.

This module builds a small knowledge graph of what the agent has learned across
sessions and exposes the two retrieval strategies the demo contrasts:

  - Before: ``VectorRetriever`` — pure vector similarity. Returns the individually
    most-similar memory nodes. It surfaces "boutique", "Kyoto", "Japan" as separate
    pieces but never connects them to the person, because similarity has no notion
    of a relationship.
  - After: ``VectorCypherRetriever`` — vector similarity to find an entry node, then
    a Cypher traversal that walks the relationships back to the person. It answers
    "Sarah Chen" and returns the path Sarah -> Vista Hotels -> boutique -> Kyoto -> Japan.

Both strategies receive the SAME facts. The graph wins because it stores them as
connected nodes, not because it is given the answer — the advantage is structural.

The graph is a *known*, seeded graph (explicit MERGE statements), so the demo is
reproducible run to run. A production system would extract entities from natural
language with an LLM (see ``neo4j_graphrag.experimental.pipeline.SimpleKGPipeline``);
that is powerful but non-deterministic, which is why the teaching demo seeds a fixed
graph and the README shows the pipeline as an optional upgrade.

Research on graph-structured agent memory:
  https://arxiv.org/abs/2603.27910 (GAAMA — Graph Augmented Associative Memory for Agents)
  https://arxiv.org/abs/2601.03236 (MAGMA — Multi-Graph based Agentic Memory Architecture)
  https://arxiv.org/abs/2605.01688 (GRAVITY — structured anchoring for long-horizon memory)

Neo4j facts used here are from the official Neo4j docs: native vector index
(``CREATE VECTOR INDEX``, cosine, dims 1-4096) and the documented "vector search then
expand through the graph" pattern implemented by ``VectorCypherRetriever``.
"""

import os
import re
import time

from dotenv import load_dotenv

# Load .env before reading any config below, so env vars are available regardless of
# which module imports this one first.
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.indexes import create_vector_index, drop_index_if_exists
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_graphrag.utils.version_utils import supports_search_clause

# ── Connection config (from env / .env) ──────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# The demo uses its OWN isolated database so it never touches other graphs on the
# same server. On Neo4j Community (single database) this falls back to the default
# database automatically — see ensure_database().
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "memorydemo")
DEFAULT_DATABASE = os.getenv("NEO4J_DEFAULT_DATABASE", "neo4j")

# ── Vector index / embedding config ──────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536              # dimensions of text-embedding-3-small
VECTOR_INDEX_NAME = "memory_embeddings"
NODE_LABEL = "Memory"         # every memory node carries this label; the index is on it

# ── The seeded memory graph (a KNOWN graph → reproducible) ───────────────────
# What the agent learned across sessions, as (subject, relation, object) triples.
# These are the SAME facts the flat/semantic memory in the "before" case receives.
FACTS = [
    ("Sarah Chen",   "WORKS_AT",   "Vista Hotels"),
    ("Vista Hotels", "HAS_STYLE",  "boutique"),
    ("Vista Hotels", "LOCATED_IN", "Kyoto"),
    ("Kyoto",        "IN_COUNTRY", "Japan"),
]

# A secondary type label per node, so a traversal can ask for "a Person connected
# to the matched entry" instead of hard-coding a relationship name.
NODE_TYPES = {
    "Sarah Chen":   "Person",
    "Vista Hotels": "Hotel",
    "boutique":     "Style",
    "Kyoto":        "Location",
    "Japan":        "Country",
}

# The text we embed for each node. Deliberately NEUTRAL: it does not echo the
# query wording ("who did I meet"), so pure vector similarity genuinely cannot
# identify the person — only the graph traversal can. This keeps the contrast honest.
NODE_TEXT = {
    "Sarah Chen":   "Sarah Chen. A contact name.",
    "Vista Hotels": "Vista Hotels. A hotel brand.",
    "boutique":     "boutique. A hotel category.",
    "Kyoto":        "Kyoto. A city.",
    "Japan":        "Japan. A country.",
}

# Allow-lists: relation types and node-type labels are interpolated into Cypher
# (Cypher cannot parameterize a relationship type or a label), so we validate them
# against these fixed sets to keep the interpolation safe.
ALLOWED_RELATIONS = {"WORKS_AT", "HAS_STYLE", "LOCATED_IN", "IN_COUNTRY"}
ALLOWED_TYPES = {"Person", "Hotel", "Style", "Location", "Country"}

# The multi-hop question the whole demo is built around.
MULTIHOP_QUESTION = "Who did I meet that's connected to boutique hotels in Japan?"

# The traversal that makes the "after" case work: the vector index hands us an
# entry node (via `node` + `score`); we then find a Person and the shortest path
# from that Person to the entry, and return the person plus the full chain.
RETRIEVAL_QUERY = """
WITH node AS entry, score
MATCH (person:Person)
WHERE person <> entry
MATCH path = shortestPath((person)-[*1..5]-(entry))
RETURN person.name AS who,
       [n IN nodes(path) | n.name] AS chain,
       max(score) AS score
ORDER BY score DESC
"""


def _valid_identifier(name: str) -> bool:
    """A conservative check for names we interpolate into Cypher (DB names)."""
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name))


def get_driver() -> "GraphDatabase.driver":
    """Create a Neo4j driver from the env config and verify connectivity."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def get_embedder() -> OpenAIEmbeddings:
    """Real OpenAI embeddings (text-embedding-3-small), same model as Demo 03.

    To use Amazon Bedrock (Amazon Titan) embeddings instead, see the commented
    block in the README — swap this for a Bedrock embedder in Demo 07.
    """
    return OpenAIEmbeddings(model=EMBED_MODEL)


def ensure_database(driver) -> str:
    """Create the demo's isolated database in the right Cypher language, and return its name.

    Two things this demo needs on the target server:

    1. An isolated database, so it never touches other graphs on the same server.

    2. The right Cypher language. On the current Neo4j (2026.x) the server defaults to
       Cypher 5, but the ``neo4j-graphrag`` retriever classes emit the newer
       ``SEARCH ... IN (VECTOR INDEX ...)`` clause, which only parses under **Cypher 25**.
       So the retrievers fail with ``Invalid input 'SEARCH'`` on a Cypher-5 database.

       The correct, supported fix is to give the database Cypher 25 as its own default
       language — set atomically when the database is created:

           CREATE DATABASE memorydemo IF NOT EXISTS DEFAULT LANGUAGE CYPHER 25

       This is the mechanism Neo4j documents for per-database language (no ``neo4j.conf``
       edit, no server restart). We gate the language clause on the library's own
       ``supports_search_clause`` so it stays in sync with what the retrievers actually
       emit — on Neo4j 5.x they use the classic ``db.index.vector.queryNodes`` procedure,
       which needs no language change, so we create a plain database there.

    On Neo4j Community (single database, no ``CREATE``/``ALTER DATABASE``) this falls back
    to the default database; see the printed guidance if the SEARCH clause is needed there.
    """
    if not _valid_identifier(NEO4J_DATABASE):
        raise ValueError(f"Invalid NEO4J_DATABASE name: {NEO4J_DATABASE!r}")

    # Server-version decision (independent of any database's language). "system" always exists.
    needs_cypher_25 = supports_search_clause(driver, "system")
    language_clause = " DEFAULT LANGUAGE CYPHER 25" if needs_cypher_25 else ""

    db = NEO4J_DATABASE
    try:
        with driver.session(database="system") as session:
            # Born in the right language, in one atomic statement.
            session.run(f"CREATE DATABASE {db} IF NOT EXISTS{language_clause}")
            # IF NOT EXISTS leaves a pre-existing database untouched, so also align the
            # language on a database created by an earlier run (idempotent no-op otherwise).
            if needs_cypher_25:
                session.run(f"ALTER DATABASE {db} SET DEFAULT LANGUAGE CYPHER 25")
        _wait_for_database_online(driver, db)
        if needs_cypher_25:
            print(f"  ✅ Database '{db}' uses Cypher 25 (required by the vector retrievers on this server).")
        return db
    except Exception as exc:
        # Community edition (or restricted permissions): can't create/alter databases.
        # Fall back to the default database — the demo still runs, but it shares a database.
        db = DEFAULT_DATABASE
        print(
            f"  ⚠️  Could not create database '{NEO4J_DATABASE}' ({str(exc).splitlines()[0][:80]}).\n"
            f"      Falling back to the default database '{db}'. On Neo4j Community this is expected."
        )
        if needs_cypher_25:
            print(
                f"      This server needs Cypher 25 for the vector retrievers, but Community can't set it\n"
                f"      per database. If they fail with \"Invalid input 'SEARCH'\", set\n"
                f"      db.query.default_language=CYPHER_25 in neo4j.conf and restart the server."
            )
        return db


def _wait_for_database_online(driver, db: str, timeout_s: float = 30.0) -> None:
    """Block until the given database reports currentStatus 'online'."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with driver.session(database="system") as session:
            row = session.run(
                "SHOW DATABASE $db YIELD currentStatus RETURN currentStatus", db=db
            ).single()
        if row and row["currentStatus"] == "online":
            return
        time.sleep(0.5)


def reset_graph(driver, db: str) -> None:
    """Remove this demo's nodes and vector index so a rerun starts clean.

    Scoped to :Memory nodes and this demo's index only — it never runs a blanket
    'delete everything', so it is safe even if the demo shares a database.
    """
    drop_index_if_exists(driver, VECTOR_INDEX_NAME, neo4j_database=db)
    with driver.session(database=db) as session:
        session.run(f"MATCH (n:{NODE_LABEL}) DETACH DELETE n")


def seed_graph(driver, db: str, embedder: OpenAIEmbeddings) -> dict:
    """Seed the known memory graph: nodes + real embeddings, relationships, vector index.

    Returns a small summary dict (node/relationship/index counts) for the test output.
    """
    # 1) Nodes: each memory node gets its neutral text, a type label, and a real embedding.
    #    The embedding is computed once here (write time) — the production pattern, and the
    #    reason the query path only has to embed the query itself.
    with driver.session(database=db) as session:
        for name, node_type in NODE_TYPES.items():
            if node_type not in ALLOWED_TYPES:
                raise ValueError(f"Unexpected node type: {node_type!r}")
            vector = embedder.embed_query(NODE_TEXT[name])
            session.run(
                f"MERGE (m:{NODE_LABEL} {{name: $name}}) "
                f"SET m.text = $text, m.type = $type, m:`{node_type}`, m.embedding = $vector",
                name=name, text=NODE_TEXT[name], type=node_type, vector=vector,
            )

        # 2) Relationships: the edges that make multi-hop traversal possible.
        for subject, relation, obj in FACTS:
            if relation not in ALLOWED_RELATIONS:
                raise ValueError(f"Unexpected relation: {relation!r}")
            session.run(
                f"MATCH (a:{NODE_LABEL} {{name: $s}}), (b:{NODE_LABEL} {{name: $o}}) "
                f"MERGE (a)-[:`{relation}`]->(b)",
                s=subject, o=obj,
            )

    # 3) Native Neo4j vector index over the node embeddings (cosine similarity).
    create_vector_index(
        driver,
        VECTOR_INDEX_NAME,
        label=NODE_LABEL,
        embedding_property="embedding",
        dimensions=EMBED_DIM,
        similarity_fn="cosine",
        neo4j_database=db,
    )
    _wait_for_index_online(driver, db, VECTOR_INDEX_NAME)

    with driver.session(database=db) as session:
        nodes = session.run(f"MATCH (n:{NODE_LABEL}) RETURN count(n) AS c").single()["c"]
        rels = session.run(
            f"MATCH (:{NODE_LABEL})-[r]->(:{NODE_LABEL}) RETURN count(r) AS c"
        ).single()["c"]
    return {"nodes": nodes, "relationships": rels, "index": VECTOR_INDEX_NAME}


def _wait_for_index_online(driver, db: str, name: str, timeout_s: float = 20.0) -> None:
    """Block until the named index reports state 'ONLINE'."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with driver.session(database=db) as session:
            row = session.run(
                "SHOW INDEXES YIELD name, state WHERE name = $name RETURN state", name=name
            ).single()
        if row and row["state"] == "ONLINE":
            return
        time.sleep(0.5)


def make_before_retriever(driver, db: str, embedder: OpenAIEmbeddings) -> VectorRetriever:
    """The 'before' retriever: pure vector similarity, no traversal.

    Returns the individually most-similar memory nodes. On a multi-hop question it
    surfaces related concepts as separate pieces but cannot connect them to a person.
    """
    return VectorRetriever(
        driver,
        index_name=VECTOR_INDEX_NAME,
        embedder=embedder,
        return_properties=["name", "text", "type"],
        neo4j_database=db,
    )


def make_after_retriever(driver, db: str, embedder: OpenAIEmbeddings) -> VectorCypherRetriever:
    """The 'after' retriever: vector similarity to find an entry node, then graph traversal.

    Uses RETRIEVAL_QUERY to walk from a Person to the vector-matched entry node and
    return the person plus the full relationship chain — the multi-hop answer.
    """
    return VectorCypherRetriever(
        driver,
        index_name=VECTOR_INDEX_NAME,
        retrieval_query=RETRIEVAL_QUERY,
        embedder=embedder,
        neo4j_database=db,
    )


def build() -> tuple:
    """Convenience: connect, ensure DB + language, reset, seed. Returns (driver, db, embedder).

    The caller owns the driver and should close it when done.
    """
    driver = get_driver()
    db = ensure_database(driver)
    embedder = get_embedder()
    reset_graph(driver, db)
    summary = seed_graph(driver, db, embedder)
    print(
        f"  Seeded graph in database '{db}': "
        f"{summary['nodes']} nodes, {summary['relationships']} relationships, "
        f"vector index '{summary['index']}'."
    )
    return driver, db, embedder
