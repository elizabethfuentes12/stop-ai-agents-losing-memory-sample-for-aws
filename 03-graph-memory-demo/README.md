# Graph Memory for AI Agents: Reasoning Over Relationships, Not Just Similarity

**Problem:** Semantic memory retrieves by *similarity*, so it finds related pieces but can't connect them. A multi-hop question — "Who do I know that's connected to flights to Spain?" — needs the *relationships* between memories, not just their vectors.

**Solution:** Store memories as a knowledge graph. Find an entry point by vector similarity, then **traverse the graph** to the answer.

> **Assumed familiarity:** This is the most advanced demo in the series. It assumes you're comfortable with the earlier demos (agent state, semantic retrieval) and have a Neo4j instance you can connect to. If you're new here, start with [Demo 01](../01-key-value-memory-demo/).

Based on research:
- [GAAMA: Graph Augmented Associative Memory for Agents](https://arxiv.org/abs/2603.27910) — Paul et al., 2026
- [MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](https://arxiv.org/abs/2601.03236) — Jiang et al., 2026
- [GRAVITY: Architecture-Agnostic Structured Anchoring for Long-Horizon Conversational Memory](https://arxiv.org/abs/2605.01688) — Sun et al., 2026

This demo uses [Strands Agents](https://github.com/strands-agents/sdk-python) for the agent harness and [Neo4j](https://neo4j.com/) with the official [`neo4j-graphrag`](https://neo4j.com/docs/neo4j-graphrag-python/) package for graph memory. The memory patterns are framework-agnostic and carry over to other agent frameworks; this demo also shows how little wiring Strands needs to plug in an external graph store.

---

## What This Demo Shows

### The scenario: what the agent learned across sessions

Over several conversations, a travel assistant picked up four facts. Stored as a graph, they form a chain:

```
(Maya Torres) ──WORKS_AT──▶ (Iberia) ──MEMBER_OF──▶ (Oneworld)
                                  │
                             FLIES_TO
                                  ▼
                              (Madrid) ──IN_COUNTRY──▶ (Spain)
```

**The question:** *"Who do I know that's connected to flights to Spain?"*

The answer — **Maya Torres** — is never stated directly. You can only reach it by following the relationships. That's a multi-hop question, and it's exactly where similarity-only memory falls short.

![Multi-hop questions answered: semantic vs graph memory](images/graph-memory-multihop.png)

### Before vs after — the same graph, two retrievers

| Retriever | Strategy | Result on the multi-hop question |
|-----------|----------|----------------------------------|
| **Before** — `VectorRetriever` | Pure vector similarity | Surfaces `Oneworld`, `Madrid`, `Spain` as separate pieces. **Never connects them to Maya.** |
| **After** — `VectorCypherRetriever` | Similarity → graph traversal | Finds an entry node, walks the edges back: **Maya Torres → Iberia → Madrid → Spain.** |

Both retrievers receive the **same facts** and share the **same vector index**. The graph wins because it stores memories as *connected nodes*, not because it's handed the answer. The advantage is structural.

---

## Scenarios Demonstrated

| Test | What it does | Recovers the multi-hop answer? |
|------|--------------|-------------------------------|
| **1. Semantic recall (before)** | `VectorRetriever` — pure similarity | No |
| **2. Graph recall (after)** | `VectorCypherRetriever` — similarity + traversal | Yes |
| **3. Full Strands agent** | Agent recalls via graph memory **and writes a new fact back** into the graph | Yes |
| **4. Deterministic scorecard** | 4 multi-hop questions, before vs after, checked against the known graph | before **1/4**, after **4/4** |

The scorecard is a deterministic check against the known graph — **not** an LLM judge — so the numbers are reproducible. (On question 3 pure similarity happens to surface the person by luck; similarity isn't always wrong on multi-hop, it's *unreliable*, while traversal is consistent.)

---

## The Strands angle (the harness)

Plugging an external graph store into an agent is *just tools + state* with Strands. You don't hand-roll the tool-calling loop:

```python
@tool(context=True)
def remember_fact(subject: str, relation: str, obj: str, tool_context: ToolContext) -> str:
    """The agent records a fact as a graph edge, and logs it to agent.state."""
    driver, db, _ = _require_memory()
    with driver.session(database=db) as session:
        session.run(
            f"MERGE (a:Memory {{name: $s}}) MERGE (b:Memory {{name: $o}}) "
            f"MERGE (a)-[:`{relation}`]->(b)", s=subject, o=obj)
    log = tool_context.agent.state.get("remembered_facts") or []
    log.append({"subject": subject, "relation": relation, "object": obj})
    tool_context.agent.state.set("remembered_facts", log)
    return f"Remembered: {subject} -[{relation}]-> {obj}"
```

The recall tools are thin wrappers over the two `neo4j-graphrag` retriever classes — no custom retrieval code.

---

## Quick Start

### Prerequisites

```bash
python --version   # Python 3.9+
export OPENAI_API_KEY="your-key-here"
```

You also need a running **Neo4j** (5.18.1+ or Aura). Options:

- **Neo4j Desktop** — create a local instance and start it (default bolt port `7687`).
- **Docker** — `docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your-password neo4j:latest`
- **Neo4j Aura** — free tier; use the `neo4j+s://...` URI it gives you.

**AI Model Provider** — This demo uses OpenAI by default. You can also use [Amazon Bedrock](https://aws.amazon.com/bedrock/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el), Anthropic, or Ollama. See [supported model providers](https://strandsagents.com/docs/user-guide/concepts/model-providers/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el). Change the model in `test_graph_memory.py`.

### Installation

```bash
uv venv && uv pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY and your NEO4J_* values
```

### Run Demo

```bash
uv run python test_graph_memory.py
```

---

## Files

| File | Purpose |
|------|---------|
| `test_graph_memory.py` | Main demo — 4 tests + comparison table |
| `graph_memory.py` | Graph memory layer: seed graph, vector index, the two retrievers |
| `travel_tools.py` | Strands `@tool`s: `remember_fact`, `recall_semantic`, `recall_graph` |
| `test_graph_memory.ipynb` | Interactive notebook walkthrough |
| `images/generate_chart.py` | Generates the scorecard chart |
| `requirements.txt` | Dependencies |

---

## How It Works

### 1. Seed a known graph with real embeddings

Facts are written as `MERGE` statements (idempotent — rerunning doesn't duplicate). Each node gets a real OpenAI embedding (`text-embedding-3-small`), computed once at write time — the production pattern.

### 2. Create a native Neo4j vector index

```python
from neo4j_graphrag.indexes import create_vector_index
create_vector_index(driver, "memory_embeddings", label="Memory",
                    embedding_property="embedding", dimensions=1536,
                    similarity_fn="cosine", neo4j_database=db)
```

### 3. Contrast the two official retrievers

```python
from neo4j_graphrag.retrievers import VectorRetriever, VectorCypherRetriever

# Before: pure similarity
before = VectorRetriever(driver, "memory_embeddings", embedder=embedder, ...)

# After: similarity → traversal. The retrieval_query receives `node` + `score`
# from the vector index, then walks the graph back to the person.
RETRIEVAL_QUERY = """
WITH node AS entry, score
MATCH (person:Person) WHERE person <> entry
MATCH path = shortestPath((person)-[*1..5]-(entry))
RETURN person.name AS who, [n IN nodes(path) | n.name] AS chain, max(score) AS score
ORDER BY score DESC
"""
after = VectorCypherRetriever(driver, "memory_embeddings", RETRIEVAL_QUERY, embedder=embedder, ...)
```

This "vector search then expand through the graph" is the documented graph-RAG pattern.

---

## Key Concepts

### Why multi-hop needs a graph

| Memory type | Stores | Answers "what do I know about X?" | Answers "who is connected to X through Y?" |
|-------------|--------|-----------------------------------|--------------------------------------------|
| Flat / key-value | Blobs | Yes (if you dump it) | No |
| Semantic (Demo 02) | Blobs + vectors | Yes (top-k) | No — no notion of relationships |
| **Graph (this demo)** | Nodes + edges + vectors | Yes | **Yes — traverse the edges** |

### This demo vs a production graph

| Component | Demo | Production |
|-----------|------|------------|
| Graph construction | Seeded known facts (`MERGE`) — reproducible | LLM entity extraction (`SimpleKGPipeline`) from raw text |
| Embeddings | OpenAI `text-embedding-3-small` (real) | Same, or Amazon Titan via Bedrock (production demo) |
| Index | Neo4j native vector index | Same, plus fulltext / hybrid retrievers |
| Traversal | Fixed shortest-path Cypher | Learned or templated multi-hop queries |

`SimpleKGPipeline` (LLM-extracted graphs) is powerful but non-deterministic, so this teaching demo seeds a fixed graph to keep the before/after scores reproducible.

---

## Learning Objectives

1. Understand why similarity-only memory fails on multi-hop questions
2. Model agent memory as a knowledge graph (nodes, typed edges, embeddings)
3. Contrast `VectorRetriever` vs `VectorCypherRetriever` on the same graph
4. Plug an external graph store into a Strands agent with just tools + state
5. Connect to production patterns (LLM graph construction, hybrid retrieval)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ServiceUnavailable` / connection refused | Neo4j isn't running or `NEO4J_URI` is wrong. Start the instance; check the bolt port (`7687`). |
| `AuthError: unauthorized` | Wrong `NEO4J_USER` / `NEO4J_PASSWORD` in `.env`. |
| `Invalid input 'SEARCH'` (CypherSyntaxError) | Version churn — see below. The demo creates its database already in Cypher 25 on Neo4j 2026.01+, so this shouldn't happen. It only appears on Neo4j **Community** (which can't set a per-database language): set `db.query.default_language=CYPHER_25` in `neo4j.conf` and restart. |
| Can't create database `memorydemo` | Expected on Neo4j **Community** (single database). The demo falls back to the default database automatically. |
| `OPENAI_API_KEY` errors | Both the model and the embeddings need it — set it in `.env`, or switch to the Bedrock/Titan block. |

### ⚠️ A real API-churn note (why the `SEARCH` clause matters)

`neo4j-graphrag` 1.18.0 emits the newer `SEARCH ... IN (VECTOR INDEX ...)` Cypher clause on Neo4j **2026.01+**. That clause is only parsed under **Cypher 25**, but the current Neo4j servers still default to **Cypher 5** — so the retriever classes fail with `Invalid input 'SEARCH'` on a Cypher-5 database. This demo handles it the supported way: it creates its database *already in Cypher 25*, atomically, when it creates it —

```cypher
CREATE DATABASE memorydemo IF NOT EXISTS DEFAULT LANGUAGE CYPHER 25
```

— gated on the library's own `supports_search_clause`, so no `neo4j.conf` edit and no server restart are needed. On Neo4j **5.x**, the same retriever classes use the classic `db.index.vector.queryNodes` procedure and no language change is needed, so the demo creates a plain database there.

**Tested versions:** Strands 1.46.0, `neo4j-graphrag` 1.18.0, `neo4j` driver 6.2.0, Neo4j server 2026.01.3 Enterprise. Behavior differs across versions — check your own.

---

## References

- [GAAMA: Graph Augmented Associative Memory for Agents](https://arxiv.org/abs/2603.27910) — Paul et al., 2026
- [MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](https://arxiv.org/abs/2601.03236) — Jiang et al., 2026
- [GRAVITY: Architecture-Agnostic Structured Anchoring for Long-Horizon Conversational Memory](https://arxiv.org/abs/2605.01688) — Sun et al., 2026

We reproduce the *mechanism* these papers describe (graph-structured memory + relationship traversal), not their specific benchmark numbers.

### Framework Documentation

- [Strands Agents SDK](https://github.com/strands-agents/sdk-python)
- [neo4j-graphrag (Python)](https://neo4j.com/docs/neo4j-graphrag-python/)
- [Neo4j vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/)

---

## Next Steps

1. [Demo 02: Vector Memory](../02-vector-memory-demo/) — similarity retrieval (what this demo extends)
2. [Demo 04: Selective Memory](../04-selective-memory-demo/) — the agent decides what to remember

---

## Contributing

Contributions are welcome! See [CONTRIBUTING](../CONTRIBUTING.md) for more information.

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file for details.
