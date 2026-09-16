"""Graph memory built by an LLM, not by hand.

Semantic memory (Demo 02) retrieves by similarity but cannot reason over
relationships. A multi-hop question — "Who do I know connected to flights to
Spain?" — needs to find an entry point by similarity, then traverse the graph.

This module builds that graph the way Neo4j builds knowledge graphs in
production: `SimpleKGPipeline` reads text and an LLM extracts entities and
relationships against a pinned schema, then merges duplicates (entity
resolution). No hand-written MERGE statements, no regex triple extractor. The
same pipeline the neo4j-graphrag docs and the AWS GraphRAG workshop use.

The pipeline writes a lexical graph (Document -> Chunk -> extracted entities)
and embeds the chunks. Retrieval matches a chunk by vector similarity, then
walks FROM_CHUNK into the extracted entities and across their relationships.

Two retrieval strategies the demo contrasts:
  - VectorRetriever: similarity over chunks only. Surfaces text, no traversal.
  - VectorCypherRetriever: similarity to an entry chunk, then a Cypher traversal
    to the connected Person and the chain that links them.

Both receive the same text and share the same vector index. The graph wins
because it stores connected entities, not because it is handed the answer.

Neo4j facts used here (verified against neo4j-graphrag 1.18.0): SimpleKGPipeline
embeds Chunk nodes (not entities) and does not create the vector index itself,
so the index is created explicitly over Chunk.embedding.
"""

import os
import re
import threading
import time

from dotenv import load_dotenv

load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.indexes import create_vector_index, drop_index_if_exists
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever
from neo4j_graphrag.utils.version_utils import supports_search_clause
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)

# ── Connection config (from env / .env) ──────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "memorydemo")
DEFAULT_DATABASE = os.getenv("NEO4J_DEFAULT_DATABASE", "neo4j")

# ── Model / index config ─────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536
CHAT_MODEL = os.getenv("GRAPH_CHAT_MODEL", "gpt-4o-mini")
VECTOR_INDEX_NAME = "chunk_embeddings"
CHUNK_LABEL = "Chunk"

# ── The pinned extraction schema ─────────────────────────────────────────────
# Without a pinned schema, SimpleKGPipeline lets the LLM invent labels per chunk
# and the traversal below can't rely on them. additional_*: False refuses
# anything outside this contract, the same discipline the AWS GraphRAG workshop
# uses for its hotel schema.
GRAPH_SCHEMA = {
    "node_types": [
        {"label": "Person", "description": "A person the traveler knows.",
         "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "Airline", "description": "An airline.",
         "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "Alliance", "description": "An airline alliance, e.g. Oneworld.",
         "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "City", "description": "A city.",
         "properties": [{"name": "name", "type": "STRING"}]},
        {"label": "Country", "description": "A country.",
         "properties": [{"name": "name", "type": "STRING"}]},
    ],
    "relationship_types": [
        {"label": "WORKS_AT"}, {"label": "MEMBER_OF"},
        {"label": "FLIES_TO"}, {"label": "IN_COUNTRY"},
    ],
    "patterns": [
        ("Person", "WORKS_AT", "Airline"),
        ("Airline", "MEMBER_OF", "Alliance"),
        ("Airline", "FLIES_TO", "City"),
        ("City", "IN_COUNTRY", "Country"),
    ],
    "additional_node_types": False,
    "additional_relationship_types": False,
    "additional_patterns": False,
}

# The seed facts live in SEED_FACTS (below), one fact per sentence, so each is
# extracted into its own chunk. That separation is what makes the contrast real:
# similarity over the chunks finds a single related fact, but only a traversal
# follows the chain Person -> Airline -> Alliance/City -> Country to the person.

# Traversal for the "after" case: from the vector-matched Chunk, step into the
# entities extracted from it (FROM_CHUNK), find a Person, and return the shortest
# path from that Person to any entity on the matched chunk.
RETRIEVAL_QUERY = """
WITH node AS chunk, score
MATCH (chunk)<-[:FROM_CHUNK]-(entity)
MATCH (person:Person)
WHERE person <> entity
MATCH path = shortestPath((person)-[:WORKS_AT|MEMBER_OF|FLIES_TO|IN_COUNTRY*1..5]-(entity))
RETURN DISTINCT person.name AS who,
       [n IN nodes(path) | coalesce(n.name, head(labels(n)))] AS chain,
       max(score) AS score
ORDER BY score DESC
LIMIT 5
"""


def _valid_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name))


def get_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def get_embedder() -> OpenAIEmbeddings:
    """OpenAI embeddings (text-embedding-3-small), the same model as Demo 02."""
    return OpenAIEmbeddings(model=EMBED_MODEL)


def get_llm() -> OpenAILLM:
    """The LLM the pipeline uses to extract entities and relationships.

    temperature=0 keeps extraction stable run to run. To use Amazon Bedrock,
    swap this for neo4j_graphrag.llm.BedrockLLM.
    """
    return OpenAILLM(model_name=CHAT_MODEL, model_params={"temperature": 0})


def ensure_database(driver) -> str:
    """Create the demo's isolated database in the right Cypher language; return its name.

    On Neo4j 2026.x the vector retrievers emit the Cypher 25 `SEARCH` clause, so
    the database is created with `DEFAULT LANGUAGE CYPHER 25`. On Community
    (single database) this falls back to the default database.
    """
    if not _valid_identifier(NEO4J_DATABASE):
        raise ValueError(f"Invalid NEO4J_DATABASE name: {NEO4J_DATABASE!r}")

    needs_cypher_25 = supports_search_clause(driver, "system")
    language_clause = " DEFAULT LANGUAGE CYPHER 25" if needs_cypher_25 else ""

    db = NEO4J_DATABASE
    try:
        with driver.session(database="system") as session:
            session.run(f"CREATE DATABASE {db} IF NOT EXISTS{language_clause}")
            if needs_cypher_25:
                session.run(f"ALTER DATABASE {db} SET DEFAULT LANGUAGE CYPHER 25")
        _wait_for_database_online(driver, db)
        if needs_cypher_25:
            print(f"  Database '{db}' created in Cypher 25 (required by the vector retrievers on this server).")
        return db
    except Exception as exc:
        db = DEFAULT_DATABASE
        print(
            f"  Could not create database '{NEO4J_DATABASE}' ({str(exc).splitlines()[0][:80]}).\n"
            f"      Falling back to the default database '{db}'. On Neo4j Community this is expected."
        )
        return db


def _wait_for_database_online(driver, db: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with driver.session(database="system") as session:
            row = session.run(
                "SHOW DATABASE $db YIELD currentStatus RETURN currentStatus", db=db
            ).single()
        if row and row["currentStatus"] == "online":
            return
        threading.Event().wait(0.5)


def _wait_for_index_online(driver, db: str, name: str, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with driver.session(database=db) as session:
            row = session.run(
                "SHOW INDEXES YIELD name, state WHERE name = $name RETURN state", name=name
            ).single()
        if row and row["state"] == "ONLINE":
            return
        threading.Event().wait(0.5)


# Everything SimpleKGPipeline writes carries this label, so the wipe is scoped to
# this demo's own output instead of a blanket delete.
KG_LABEL = "__KGBuilder__"


def triples_for_sentence(driver, db: str, sentence: str) -> list[dict]:
    """Return the (subject, relation, object) triples the pipeline extracted from one
    sentence, by matching the chunk whose text is that sentence and reading the
    relationships between the entities extracted from it. Lets a tool report what it
    just wrote to the graph as structured triples, e.g. to log them in agent.state.
    """
    query = (
        f"MATCH (c:{CHUNK_LABEL})<-[:FROM_CHUNK]-(a:__Entity__)-[r]->(b:__Entity__) "
        "WHERE c.text = $sentence "
        "RETURN DISTINCT a.name AS subject, type(r) AS relation, b.name AS object"
    )
    with driver.session(database=db) as session:
        rows = session.run(query, sentence=sentence)
        return [{"subject": r["subject"], "relation": r["relation"], "object": r["object"]}
                for r in rows]


def reset_graph(driver, db: str) -> None:
    """Drop this demo's vector index and the nodes the pipeline created."""
    drop_index_if_exists(driver, VECTOR_INDEX_NAME, neo4j_database=db)
    with driver.session(database=db) as session:
        session.run(f"MATCH (n:`{KG_LABEL}`) DETACH DELETE n")
        session.run(f"MATCH (c:{CHUNK_LABEL}) DETACH DELETE c")
        session.run("MATCH (d:Document) DETACH DELETE d")


def build_pipeline(driver, db: str, llm=None, embedder=None) -> SimpleKGPipeline:
    """Construct the LLM extraction pipeline with the schema pinned.

    from_pdf=False so we can pass text directly. perform_entity_resolution=True
    merges duplicate entities (one Iberia, not one per chunk). A small chunk size
    keeps each fact in its OWN chunk, so similarity search retrieves a single fact
    and cannot see the person on the other end of the chain: that is what forces the
    multi-hop, and what makes the semantic-vs-graph contrast real.
    """
    return SimpleKGPipeline(
        llm=llm or get_llm(),
        driver=driver,
        embedder=embedder or get_embedder(),
        schema=GRAPH_SCHEMA,
        from_pdf=False,
        perform_entity_resolution=True,
        neo4j_database=db,
        text_splitter=FixedSizeSplitter(chunk_size=60, chunk_overlap=0),
    )


# The seed facts, one per sentence. Kept as a list (not one blob) so each fact is
# extracted into its OWN chunk: similarity then returns a single fact, and only a
# graph traversal connects it back to the person.
SEED_FACTS = [
    "Maya Torres works at Iberia.",
    "Iberia is a member of the Oneworld alliance.",
    "Iberia flies to Madrid.",
    "Madrid is in Spain.",
    "Diego Fuentes works at Lufthansa.",
    "Lufthansa is a member of the Star Alliance.",
    "Lufthansa flies to Munich.",
    "Munich is in Germany.",
    "Priya Nair works at Qatar Airways.",
    "Qatar Airways is a member of the Oneworld alliance.",
    "Qatar Airways flies to Doha.",
    "Doha is in Qatar.",
    "Sofia Rossi works at ITA Airways.",
    "ITA Airways flies to Rome.",
    "Rome is in Italy.",
]


async def seed_graph(driver, db: str, facts: list = None,
                     llm=None, embedder=None) -> dict:
    """Extract the graph from the seed facts with the LLM, then build the chunk vector
    index. Each fact is run separately so it lands in its own chunk.

    Returns a summary (entity + relationship + index) for the notebook output.
    """
    facts = facts if facts is not None else SEED_FACTS
    pipeline = build_pipeline(driver, db, llm=llm, embedder=embedder)
    for fact in facts:
        await pipeline.run_async(text=fact)

    # SimpleKGPipeline embeds chunks but does not create the index; create it here.
    create_vector_index(
        driver, VECTOR_INDEX_NAME, label=CHUNK_LABEL, embedding_property="embedding",
        dimensions=EMBED_DIM, similarity_fn="cosine", neo4j_database=db,
    )
    _wait_for_index_online(driver, db, VECTOR_INDEX_NAME)

    with driver.session(database=db) as session:
        entities = session.run(
            "MATCH (e:__Entity__) RETURN count(e) AS c"
        ).single()["c"]
        rels = session.run(
            "MATCH (:__Entity__)-[r]->(:__Entity__) RETURN count(r) AS c"
        ).single()["c"]
    return {"entities": entities, "relationships": rels, "index": VECTOR_INDEX_NAME}


def make_semantic_retriever(driver, db: str, embedder=None) -> VectorRetriever:
    """Similarity over chunks only, no traversal.

    Returns the most similar chunk text. It contains the facts as prose but the
    retriever cannot follow the relationships between the entities in them.
    """
    return VectorRetriever(
        driver, index_name=VECTOR_INDEX_NAME,
        embedder=embedder or get_embedder(),
        return_properties=["text"], neo4j_database=db,
    )


def make_graph_retriever(driver, db: str, embedder=None) -> VectorCypherRetriever:
    """Similarity to an entry chunk, then a Cypher traversal to the connected Person."""
    return VectorCypherRetriever(
        driver, index_name=VECTOR_INDEX_NAME,
        retrieval_query=RETRIEVAL_QUERY,
        embedder=embedder or get_embedder(), neo4j_database=db,
    )


def teardown_graph(driver, db: str) -> None:
    """Full teardown: clear this demo's nodes/index, then drop the isolated database."""
    reset_graph(driver, db)
    if db == DEFAULT_DATABASE:
        print(f"  running on the default database '{db}'; cleared this demo's nodes only.")
        return
    with driver.session(database="system") as session:
        session.run(f"DROP DATABASE {db} IF EXISTS")
    print(f"  dropped database '{db}' (full teardown).")


async def build() -> tuple:
    """Connect, ensure DB + language, reset, extract the graph. Returns (driver, db, embedder)."""
    driver = get_driver()
    db = ensure_database(driver)
    embedder = get_embedder()
    reset_graph(driver, db)
    summary = await seed_graph(driver, db, embedder=embedder)
    print(
        f"  Extracted graph in '{db}': {summary['entities']} entities, "
        f"{summary['relationships']} relationships, vector index '{summary['index']}'."
    )
    return driver, db, embedder
