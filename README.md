# Stop AI Agents from Losing Memory: 6 Research-Backed Fixes

[![License](https://img.shields.io/badge/License-MIT--0-blue.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg?style=for-the-badge&logo=python)](https://python.org)
[![Strands](https://img.shields.io/badge/Strands_Agents-framework-blue.svg?style=for-the-badge)](https://github.com/strands-agents/sdk-python)
[![AWS](https://img.shields.io/badge/AWS-Bedrock-orange.svg?style=for-the-badge&logo=amazon-aws)](https://aws.amazon.com/bedrock/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)

Research-backed techniques to stop AI agents from losing memory: key-value state for preference retention, vector memory for semantic retrieval, graph memory for multi-hop reasoning, selective memory for deciding what to keep, memory hygiene against poisoning, and decision traces for auditability.

These demos use Strands Agents for implementation. The memory patterns demonstrated are framework-agnostic and carry over to other agent frameworks.

---

## Demos

| Demo | Description | Stack |
|------|-------------|-------|
| [01 - Key-Value Memory](01-key-value-memory-demo/) | Stop your agent from forgetting user preferences: the same 3-turn conversation climbing the durability ladder — no memory → `agent.state` → local disk → Amazon S3. Real flight data (Duffel); the only variable is where memory lives. | ![Python](https://img.shields.io/badge/Python-3.9+-green) ![Strands](https://img.shields.io/badge/Strands-agent.state-blue) |
| [02 - Vector Memory](02-vector-memory-demo/) | Do you need a vector database for agent memory? FAISS (in-process, <0.1 ms, dies with the process) vs Amazon S3 Vectors (managed, ~200 ms, survives restarts) — same Titan embeddings, measured. | ![Python](https://img.shields.io/badge/Python-3.9+-green) ![AWS](https://img.shields.io/badge/AWS-S3_Vectors-orange) ![Strands](https://img.shields.io/badge/Strands-memory-blue) |
| [03 - Graph Memory](03-graph-memory-demo/) | Vector memory can't reason over relationships. Store memories as a Neo4j knowledge graph and traverse it to answer multi-hop questions: before 1/4, after 4/4. | ![Python](https://img.shields.io/badge/Python-3.9+-green) ![Neo4j](https://img.shields.io/badge/Neo4j-graph_memory-blue) ![Strands](https://img.shields.io/badge/Strands-tools+state-blue) |
| [04 - Selective Memory](04-selective-memory-demo/) | What should your agent actually remember? Three selection mechanisms measured: agent tools (inline), your own 4-prompt extractor to S3 Vectors, and AgentCore's built-in strategies — keep/discard quality, turn overhead, and the ~53-85 s managed extraction lag nobody publishes. | ![Python](https://img.shields.io/badge/Python-3.9+-green) ![Strands](https://img.shields.io/badge/Strands-core_memory-blue) |
| [05 - Memory Hygiene](05-memory-hygiene-demo/) | What an agent should NOT remember. A write-gate blocks poisoned/injected content; forget removes it. One poisoned fact contaminates 1 answer in key-value memory but 4/4 in a graph. | ![Python](https://img.shields.io/badge/Python-3.9+-green) ![Neo4j](https://img.shields.io/badge/Neo4j-graph_memory-blue) ![Strands](https://img.shields.io/badge/Strands-write_gate-blue) |
| [06 - Reasoning Memory](06-reasoning-memory-demo/) | Remember WHY the agent decided, not just what it knows. A HookProvider records decision traces automatically; the reverse audit finds 2/4 affected decisions with a flat scan vs 4/4 with a graph traversal. | ![Python](https://img.shields.io/badge/Python-3.9+-green) ![Neo4j](https://img.shields.io/badge/Neo4j-provenance-blue) ![Strands](https://img.shields.io/badge/Strands-hooks-blue) |
| 07 - Hybrid Memory | *In design.* Two memories, one agent: vector + graph combined (GAAMA pattern), with S3 Vectors-built-by-hand vs AgentCore-managed at full parity. | ![AWS](https://img.shields.io/badge/AWS-S3_Vectors-orange) ![AgentCore](https://img.shields.io/badge/Bedrock-AgentCore_Memory-orange) ![Neo4j](https://img.shields.io/badge/Neo4j-graph-blue) |
| 08 - Production Deploy | *Pending.* One deploy per memory type — pick the memory the use case needs, don't ship a monolithic all-in-one stack. | — |

---

## How Each Demo Works

### Demo 01: Key-Value Memory (Agent State) — Stop Your Agent from Forgetting

**Research:** [MemoryOS of AI Agent](https://arxiv.org/abs/2506.06326) (Kang et al., 2025) — +49% F1 (accuracy metric), +46% BLEU-1 (text quality metric) with hierarchical memory

Without `agent.state`, the agent has no mechanism to learn from user actions: tools return results but never store preferences, so the only "memory" is the conversation transcript. Within a session the transcript papers over it — but nothing structured is learned, and one restart (every new process or request in production) erases everything. A user books a business-class flight, the process restarts, and "based on what you know about me" gets a generic answer.

| Test | Approach | Structured profile | Cross-session |
|------|----------|--------------------|---------------|
| 1 | Stateless tools (transcript only) | No | No |
| 2 | `agent.state` tools | Yes | No |
| 3 | `FileSessionManager` | Yes | Yes (local disk) |
| 4 | `S3SessionManager` | Yes | Yes (Amazon S3) |

```python
@tool(context=True)
def book_flight(offer_id: str, tool_context: ToolContext) -> str:
    offer = flights_api.get_offer(offer_id)              # the REAL chosen offer
    prefs = tool_context.agent.state.get("user_preferences") or {}
    prefs["preferred_cabin"] = offer["cabin"]            # learned from the action
    tool_context.agent.state.set("user_preferences", prefs)
```

---

### Demo 02: Vector Memory — FAISS vs Amazon S3 Vectors

**Research:** [Zep: Temporal Knowledge Graph](https://arxiv.org/abs/2501.13956) (Rasmussen et al., 2025) · [Amazon S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) · [Bedrock AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/built-in-strategies.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)

The traveler asks *"what should I avoid eating on this trip?"* — the answer is stored under `dietary_notes`, but the question names no key and shares no words with the note. Key-value memory misses; vector memory retrieves by meaning. Then the real decision: in-process index or managed storage? Same Titan V2 embeddings, same memories, measured:

| Store | Finds the answer | Query latency | Survives restart |
|-------|------------------|---------------|------------------|
| Key-value (keyword scan) | No | — | with a session manager |
| FAISS (in-process) | Yes | <0.1 ms | No |
| Amazon S3 Vectors (managed) | Yes | ~170-200 ms | Yes (verified) |

The honest footnote: embedding the question (~0.5 s with Titan V2) dominates and costs the same for both backends.

| Backend | What you manage | Scope |
|---------|-----------------|-------|
| **FAISS** (in-process) | You build and hold the index in memory | Manual, per-process |
| **Amazon S3 Vectors** | Managed storage: you create the bucket + index and call `put_vectors` / `query_vectors` | One index per tenant / memory type |

---

### Demo 03: Graph Memory — Reasoning Over Relationships

**Research:** [MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](https://arxiv.org/abs/2601.03236) (Jiang et al., 2026)

Semantic search finds *similar* memories but can't connect them. A question like *"Who do I know that's connected to flights to Spain?"* needs the relationships between memories. Store them as a Neo4j knowledge graph, then find an entry point by vector similarity and **traverse the edges** to the answer.

| Retriever | Strategy | Multi-hop questions correct |
|-----------|----------|-----------------------------|
| `VectorRetriever` (before) | Pure similarity | 1/4 |
| `VectorCypherRetriever` (after) | Similarity + traversal | 4/4 |

```python
# Plugging a graph store into a Strands agent is just tools + state
agent = Agent(
    model=MODEL,
    tools=[recall_graph, recall_semantic, remember_fact],
)
```

---

### Demo 04: Selective Memory — 3 Ways to Decide What to Remember

**Research:** [MIRIX: Multi-Agent Memory System](https://arxiv.org/abs/2507.07957) (Wang & Chen, 2025) — 6 memory types, +35% accuracy, state-of-the-art (SOTA) 85.4% on LOCOMO

A planted conversation carries 5 items worth keeping (facts, preferences, an episode) and 3 decoys (small talk, a passing opinion, ephemeral weather). Three selection mechanisms, same input, deterministic scoring:

| Mechanism | Kept | Decoys leaked | Turn overhead | Available after |
|-----------|------|---------------|---------------|-----------------|
| A — agent tools (inline, `agent.state`) | 4/5 | 0 | ~1.5-2.5 s/turn | immediately |
| B — own 4-prompt extractor → S3 Vectors | **5/5** | 0 | 0 (off-path) | ~4 s/turn |
| C — AgentCore built-in strategies | 5/5* | 0-2* | ~0.4 s/turn | **~53-85 s (measured)** |

\* C is nondeterministic run to run — its criteria aren't yours to tune; that's the trade-off. B is "AgentCore built by hand": same pipeline, same per-type partitioning, but you own the prompts.

| Memory type | What it holds | AgentCore strategy equivalent |
|-------------|---------------|-------------------------------|
| `facts` | durable facts about the user's world | semantic |
| `preferences` | likes/dislikes the user reveals | userPreference |
| `trip_summary` | rolling summary of the current plan | summary |
| `episodes` | notable events, one entry each | episodic |

---

### Demo 05: Memory Hygiene — What an Agent Should NOT Remember

**Research:** [AgentPoison](https://arxiv.org/abs/2407.12784) (2024) · [PoisonedRAG](https://arxiv.org/abs/2402.07867) (USENIX Security 2025)

Poisoned or injected content that reaches long-term memory persists across sessions and corrupts future answers. Defend at the **write path** (screen before storing) and **forget** what already got in. The same attack has a very different blast radius depending on the store:

| Backend | Poisoned | Gated (write-gate) | Cleaned (forget) |
|---------|----------|--------------------|------------------|
| Key-value (`agent.state`) | 1/4 | 0/4 | 0/4 |
| Graph (Neo4j) | 4/4 | 0/4 | 0/4 |

One poisoned fact contaminates every multi-hop answer that traverses it — so graph memory is more powerful *and* more sensitive to poisoning.

---

### Demo 06: Reasoning Memory — Remember WHY You Decided

**Research:** [MemWeaver](https://arxiv.org/abs/2601.18204) (2026) · [Less Context, More Accuracy (the Engram system)](https://arxiv.org/abs/2606.09900) (2026, preprint) — the traceability/provenance theme. *"Reasoning memory" itself is an engineering pattern, not an established academic category.*

Agent memory stores *what* the agent knows — not *why it decided*. A `DecisionTraceRecorder` (a Strands `HookProvider`) captures each decision's question → tool steps → evidence → outcome automatically, with **zero changes to the tools**. Both a flat store and a graph replay "why did you recommend X?"; the graph also answers the **reverse audit**:

| Store | "Why did I decide X?" | "Source S was wrong — which decisions relied on it?" |
|-------|-----------------------|------------------------------------------------------|
| Key-value (flat scan) | ✅ | 2/4 — direct citations only |
| Graph (Neo4j traversal) | ✅ | 4/4 — follows provenance at any depth, with receipts |

```python
agent = Agent(
    model=MODEL,
    tools=[search_flights, check_fare_alert],   # unchanged tools
    hooks=[DecisionTraceRecorder()],            # the only addition
)
```

---

## Quick Start

```bash
cd 01-key-value-memory-demo
uv venv && uv pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
uv run python test_key_value_memory.py
```

You can use different AI model providers (like Amazon Bedrock or Anthropic Claude) instead of OpenAI. See [supported model providers](https://strandsagents.com/docs/user-guide/concepts/model-providers/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) for details — each demo's README shows the one-line model swap.

---

## Research Papers Referenced

| Paper | Key Finding | Demo |
|-------|------------|------|
| [MemoryOS of AI Agent](https://arxiv.org/abs/2506.06326) | Hierarchical memory: +49% F1 on LoCoMo | 01 |
| [Cognitive Memory in LLMs](https://arxiv.org/abs/2504.02441) | Survey: sensory, short-term, long-term memory | 01 |
| [MemGPT](https://arxiv.org/abs/2310.08560) | Core memory concept, virtual context management | 01, 04 |
| [Zep Temporal Knowledge Graph](https://arxiv.org/abs/2501.13956) | 94.8% DMR, 18.5% accuracy improvement | 02 |
| [HippoRAG 2](https://arxiv.org/abs/2502.14802) | +7% associative memory via Personalized PageRank | 02 |
| [MAGMA: A Multi-Graph based Agentic Memory Architecture](https://arxiv.org/abs/2601.03236) | Semantic/temporal/causal/entity graphs + policy-guided traversal | 03 |
| [GAAMA: Graph Augmented Associative Memory for Agents](https://arxiv.org/abs/2603.27910) | Concept-mediated knowledge graph + kNN/PageRank retrieval | 03 |
| [GRAVITY: Structured Anchoring for Long-Horizon Memory](https://arxiv.org/abs/2605.01688) | Entity profiles grounded in relational graphs | 03 |
| [MIRIX: Multi-Agent Memory](https://arxiv.org/abs/2507.07957) | 6 memory types, SOTA 85.4% on LOCOMO | 04 |
| [AgentPoison](https://arxiv.org/abs/2407.12784) | >80% attack success poisoning <0.1% of agent memory | 05 |
| [PoisonedRAG](https://arxiv.org/abs/2402.07867) | ~90% attack success with 5 malicious texts (USENIX Security 2025) | 05 |
| [MINJA: Memory Injection Attack](https://arxiv.org/abs/2503.03704) | Memory injection through query-only interaction (preprint) | 05 |
| [MemWeaver: Traceable Long-Horizon Agentic Reasoning](https://arxiv.org/abs/2601.18204) | Hybrid memory with dual-channel retrieval; traceability theme | 06 |
| [Less Context, More Accuracy (the Engram system)](https://arxiv.org/abs/2606.09900) | Every stored fact keeps provenance + a supersession chain (preprint) | 06 |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING](CONTRIBUTING.md) for more information.

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file for details.
