"""Decision traces over a graph store (Neo4j), provenance you can traverse.

This is the graph counterpart to trace_kv.py. It stores the SAME decision traces, but as
connected nodes instead of flat blobs:

    (:Decision)-[:HAS_STEP]->(:Step)-[:NEXT]->(:Step)      the reasoning chain
    (:Step)-[:USED]->(:Evidence)                            what each step relied on
    (:Evidence)-[:DERIVED_FROM]->(:Evidence)                evidence built on other evidence
    (:Evidence)-[:FROM_SOURCE]->(:Source)                   external origin of the evidence

Both stores answer "why did I decide X?" fine. The question that separates them is the
REVERSE audit: "evidence source S turned out to be false, which of my decisions depended
on it?" In the flat store that is a linear scan of each trace's own blob, so it only finds
decisions that cite S *directly*. In the graph it is one traversal:

    MATCH (d:Decision)-[:HAS_STEP]->(:Step)-[:USED]->(:Evidence)
          -[:DERIVED_FROM*0..]->(:Evidence)-[:FROM_SOURCE]->(s:Source {name: $source})
    RETURN DISTINCT d

The variable-length ``DERIVED_FROM*0..`` hop is what the flat store cannot express: it
follows provenance through decisions that depended on S only via *other decisions'
outputs*, at any depth.

Uses an isolated database (default ``reasoningdemo``) created in Cypher 25, matching
demos 03 and 05, this demo's replay/audit queries are plain Cypher (no vector index
needed), but a consistent database language means Demo 03's vector retrievers can be
pointed at this graph later without surprises. Never touches other databases.
"""

import os
import re
import threading
import time

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.utils.version_utils import supports_search_clause

from trace_kv import COMPROMISED_SOURCE, EXTERNAL_SOURCES, SEED_TRACES

# ── Connection config (from env / .env) ──────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("REASONING_NEO4J_DATABASE", "reasoningdemo")
DEFAULT_DATABASE = os.getenv("NEO4J_DEFAULT_DATABASE", "neo4j")

DEMO_LABELS = ("Decision", "Step", "Evidence", "Source")


def _valid_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name))


def get_driver():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    return driver


def ensure_database(driver) -> str:
    """Create the demo's isolated database (Cypher 25 where supported); return its name.

    Same approach as demos 03 and 05: atomic ``CREATE DATABASE ... DEFAULT LANGUAGE
    CYPHER 25`` on servers that support it, no ``neo4j.conf`` edit, no restart. Falls
    back to the default database on Neo4j Community.
    """
    if not _valid_identifier(NEO4J_DATABASE):
        raise ValueError(f"Invalid database name: {NEO4J_DATABASE!r}")

    use_cypher_25 = supports_search_clause(driver, "system")
    language_clause = " DEFAULT LANGUAGE CYPHER 25" if use_cypher_25 else ""

    db = NEO4J_DATABASE
    try:
        with driver.session(database="system") as session:
            session.run(f"CREATE DATABASE {db} IF NOT EXISTS{language_clause}")
            if use_cypher_25:
                session.run(f"ALTER DATABASE {db} SET DEFAULT LANGUAGE CYPHER 25")
        _wait_for_database_online(driver, db)
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


def reset_graph(driver, db: str) -> None:
    """Remove this demo's nodes so a rerun starts clean (scoped to the demo's labels)."""
    labels = "|".join(DEMO_LABELS)
    with driver.session(database=db) as session:
        session.run(f"MATCH (n:{labels}) DETACH DELETE n")


def teardown_graph(driver, db: str) -> None:
    """Full teardown: clear the demo's nodes, then DROP the isolated database entirely.

    reset_graph() only clears the demo's labels (for rerun-safety). This removes the
    whole `reasoningdemo` database so nothing this demo created is left behind.
    Guarded: never drops the shared default database (the Community fallback).
    """
    reset_graph(driver, db)
    if db == DEFAULT_DATABASE:
        print(f"  running on the default database '{db}'; leaving it in place (only cleared this demo's nodes).")
        return
    with driver.session(database="system") as session:
        session.run(f"DROP DATABASE {db} IF EXISTS")
    print(f"  dropped database '{db}' (full teardown).")



# ── Writing traces as graph chains ────────────────────────────────────────────
def write_trace(driver, db: str, trace: dict) -> None:
    """Store one decision trace as a node chain with evidence provenance.

    Every evidence record's ``source`` either names an external :Source (origin) or
    another :Evidence record (derivation), the same field the flat store keeps, but
    here it becomes a traversable edge instead of a string inside a blob.
    """
    with driver.session(database=db) as session:
        session.run(
            "MERGE (d:Decision {id: $id}) SET d.question = $question, d.outcome = $outcome",
            id=trace["id"], question=trace["question"], outcome=trace["outcome"],
        )
        previous_step = None
        for step in trace["steps"]:
            step_id = f"{trace['id']}-step-{step['n']}"
            session.run(
                "MATCH (d:Decision {id: $did}) "
                "MERGE (s:Step {id: $sid}) SET s.n = $n, s.tool = $tool, s.input = $input "
                "MERGE (d)-[:HAS_STEP]->(s)",
                did=trace["id"], sid=step_id, n=step["n"], tool=step["tool"],
                input=str(step["input"]),
            )
            if previous_step:
                session.run(
                    "MATCH (a:Step {id: $prev}), (b:Step {id: $curr}) MERGE (a)-[:NEXT]->(b)",
                    prev=previous_step, curr=step_id,
                )
            previous_step = step_id

            evidence = step["evidence"]
            session.run(
                "MATCH (s:Step {id: $sid}) "
                "MERGE (e:Evidence {name: $name}) SET e.text = $text "
                "MERGE (s)-[:USED]->(e)",
                sid=step_id, name=evidence["name"], text=evidence["text"],
            )
            if evidence["source"] in EXTERNAL_SOURCES:
                session.run(
                    "MATCH (e:Evidence {name: $name}) "
                    "MERGE (src:Source {name: $source}) "
                    "MERGE (e)-[:FROM_SOURCE]->(src)",
                    name=evidence["name"], source=evidence["source"],
                )
            else:
                session.run(
                    "MATCH (e:Evidence {name: $name}) "
                    "MERGE (parent:Evidence {name: $source}) "
                    "MERGE (e)-[:DERIVED_FROM]->(parent)",
                    name=evidence["name"], source=evidence["source"],
                )


def seed_graph(driver, db: str) -> dict:
    """Write the shared seeded decision history into the graph. Returns a summary."""
    for trace in SEED_TRACES:
        write_trace(driver, db, trace)
    with driver.session(database=db) as session:
        counts = session.run(
            "MATCH (d:Decision) WITH count(d) AS decisions "
            "MATCH (e:Evidence) WITH decisions, count(e) AS evidence "
            "MATCH (s:Source) RETURN decisions, evidence, count(s) AS sources"
        ).single()
    return dict(counts)


# ── The queries (deterministic, plain Cypher, no LLM judge) ─────────────────
def replay_why_graph(driver, db: str, topic: str) -> dict | None:
    """Answer "why did I decide X?" by walking the decision's step chain.

    Several decisions may mention the topic (a budget that includes the flight, the
    decision that chose it, ...). Rank matches by where the topic first appears in the
    outcome, the decision *about* X mentions it earliest, so the pick is deterministic.
    """
    with driver.session(database=db) as session:
        row = session.run(
            "MATCH (d:Decision) "
            "WHERE toLower(d.outcome) CONTAINS toLower($topic) "
            "   OR toLower(d.question) CONTAINS toLower($topic) "
            "MATCH (d)-[:HAS_STEP]->(s:Step)-[:USED]->(e:Evidence) "
            "RETURN d.id AS id, d.question AS question, d.outcome AS outcome, "
            "       collect({n: s.n, tool: s.tool, input: s.input, evidence: e.text}) AS steps "
            # Text before the first occurrence of the topic; absent → full length (ranks last).
            "ORDER BY size(split(toLower(d.outcome), toLower($topic))[0]), id LIMIT 1",
            topic=topic,
        ).single()
    if row is None:
        return None
    trace = dict(row)
    trace["steps"] = sorted(trace["steps"], key=lambda s: s["n"])
    return trace


def find_affected_decisions_graph(driver, db: str, source_name: str = COMPROMISED_SOURCE) -> list:
    """The reverse audit as ONE traversal: every decision whose evidence chain reaches
    the compromised source, directly or through any depth of derived evidence.

    ``DERIVED_FROM*0..`` is the part a flat scan cannot express: length 0 covers
    evidence taken straight from the source, and longer paths follow provenance through
    decisions that only depended on the source via other decisions' outputs.
    """
    with driver.session(database=db) as session:
        rows = session.run(
            "MATCH (d:Decision)-[:HAS_STEP]->(:Step)-[:USED]->(:Evidence)"
            "      -[:DERIVED_FROM*0..]->(:Evidence)-[:FROM_SOURCE]->(src:Source {name: $source}) "
            "RETURN DISTINCT d.id AS id ORDER BY id",
            source=source_name,
        )
        return [row["id"] for row in rows]


def provenance_path(driver, db: str, decision_id: str, source_name: str = COMPROMISED_SOURCE) -> list:
    """Return one shortest evidence path from a decision to the source, as readable hops.

    This is the audit's receipt: not just "this decision is affected" but the exact
    chain of evidence that connects it to the compromised source.
    """
    with driver.session(database=db) as session:
        row = session.run(
            "MATCH (d:Decision {id: $id}), (src:Source {name: $source}) "
            "MATCH path = shortestPath("
            "  (d)-[:HAS_STEP|USED|DERIVED_FROM|FROM_SOURCE*..10]->(src)) "
            "RETURN [n IN nodes(path) | coalesce(n.id, n.name)] AS hops",
            id=decision_id, source=source_name,
        ).single()
    return row["hops"] if row else []
