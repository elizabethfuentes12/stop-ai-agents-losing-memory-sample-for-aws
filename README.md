# Stop AI Agents from Losing Memory: 3 Essential Fixes

[![License](https://img.shields.io/badge/License-MIT--0-blue.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg?style=for-the-badge&logo=python)](https://python.org)
[![Strands](https://img.shields.io/badge/Strands_Agents-framework-blue.svg?style=for-the-badge)](https://github.com/strands-agents/sdk-python)
[![AWS](https://img.shields.io/badge/AWS-Bedrock-orange.svg?style=for-the-badge&logo=amazon-aws)](https://aws.amazon.com/bedrock/)

Research-backed techniques to stop AI agents from losing memory: persistent state for preference retention, core memory pattern for structured user profiles, and semantic retrieval for efficient memory access at scale.

These demos use Strands Agents for implementation. The memory patterns demonstrated are framework-agnostic and can be applied in LangGraph, AutoGen, CrewAI, or other agent frameworks.

---

## Demos

| Demo | Description | Stack |
|------|-------------|-------|
| [01 - Memory Decay](01-memory-decay-demo/) | Compare stateless vs stateful agents on a 3-turn travel conversation. Stateful agents retain preferences, stateless agents forget. | ![Python](https://img.shields.io/badge/Python-3.9+-green) ![Strands](https://img.shields.io/badge/Strands-agent.state-blue) |
| [02 - Core Memory Pattern](02-core-memory-demo/) | Give the agent explicit tools to manage its own memory: read, write, update, list. Agent decides what to remember (MIRIX/MemGPT pattern). | ![Python](https://img.shields.io/badge/Python-3.9+-green) ![Strands](https://img.shields.io/badge/Strands-core_memory-blue) |
| [03 - Memory Retrieval](03-memory-retrieval-demo/) | When memory grows large, compare dump-all vs keyword vs semantic retrieval. Semantic search uses 60-98% fewer tokens. | ![Python](https://img.shields.io/badge/Python-3.9+-green) ![Strands](https://img.shields.io/badge/Strands-semantic_search-blue) |
| [04 - Graph Memory](04-graph-memory-demo/) | Semantic memory can't reason over relationships. Store memories as a Neo4j knowledge graph and traverse it to answer multi-hop questions: before 1/4, after 4/4. | ![Python](https://img.shields.io/badge/Python-3.9+-green) ![Neo4j](https://img.shields.io/badge/Neo4j-graph_memory-blue) ![Strands](https://img.shields.io/badge/Strands-tools+state-blue) |

---

## How Each Demo Works

### Demo 01: Memory Decay — Why Agents Forget

**Research:** [MemoryOS of AI Agent](https://arxiv.org/abs/2506.06326) (Kang et al., 2025) — +49% F1 (accuracy metric), +46% BLEU-1 (text quality metric) with hierarchical memory

Without `agent.state`, the agent has no mechanism to learn from user actions. Tools return results but never store preferences. The user books a luxury ryokan in Tokyo, then searches Zurich — the agent recommends budget hostels.

![Stateless vs Stateful Agent comparison](images/memory-decay-comparison.png)

| Test | Approach | Remembers | Cross-session |
|------|----------|-----------|---------------|
| 1 | Stateless tools | No | No |
| 2 | `agent.state` tools | Yes | No |
| 3 | `FileSessionManager` | Yes | Yes |

```python
@tool(context=True)
def book_hotel(hotel_name: str, tool_context: ToolContext) -> str:
    prefs = tool_context.agent.state.get("user_preferences") or {}
    prefs["preferred_style"] = hotel["style"]       # Learned from action
    tool_context.agent.state.set("user_preferences", prefs)
```

---

### Demo 02: Core Memory Pattern — Enable Agent Memory Management

**Research:** [MIRIX: Multi-Agent Memory System](https://arxiv.org/abs/2507.07957) (Wang & Chen, 2025) — 6 memory types, +35% accuracy, state-of-the-art (SOTA) 85.4% on LOCOMO

Core Memory gives the agent explicit tools to manage its own memory — read, write, update, list sections. The agent decides WHEN to store information, WHAT to remember, and HOW to update its knowledge.

| Memory Section | Content | Example |
|----------------|---------|---------|
| `persona` | Identity | `{"name": "Alex", "role": "traveler"}` (Alex is one example persona; real applications would have diverse user profiles) |
| `preferences` | Learned preferences | `{"style": "traditional", "stars": 4}` |
| `history` | Past events | `[{"hotel": "Zen Garden", "rating": 5}]` |
| `instructions` | User rules | `"Always suggest pet-friendly"` |

![Core memory sections evolving over time](images/core-memory-evolution.png)

```python
agent = Agent(
    model=MODEL,
    tools=[core_memory_read, core_memory_write, core_memory_update, core_memory_list,
           search_hotels, book_hotel],
    session_manager=FileSessionManager(session_id="user-42"),
)
```

---

### Demo 03: Memory Retrieval — Find the Right Memories

**Research:** [Zep: Temporal Knowledge Graph](https://arxiv.org/abs/2501.13956) (Rasmussen et al., 2025) — 94.8% on DMR, 90% less latency

When memory grows to dozens of sections, dumping everything into context wastes tokens and degrades quality. Semantic search retrieves only the most relevant memories per query.

| Strategy | Tokens | Precision |
|----------|--------|-----------|
| Dump all | High (~2,000) | Low |
| Keyword | Medium | Medium |
| Semantic top-3 | Low (~800) | High |

---

### Demo 04: Graph Memory — Reasoning Over Relationships

**Research:** [MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](https://arxiv.org/abs/2601.03236) (Jiang et al., 2026)

Semantic search finds *similar* memories but can't connect them. A question like *"Who did I meet that's connected to boutique hotels in Japan?"* needs the relationships between memories. Store them as a Neo4j knowledge graph, then find an entry point by vector similarity and **traverse the edges** to the answer.

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

## Quick Start

```bash
cd 01-memory-decay-demo
uv venv && uv pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
uv run python test_memory_decay.py
```

You can use different AI model providers (like Amazon Bedrock or Anthropic Claude) instead of OpenAI. See [supported model providers](https://strandsagents.com/docs/user-guide/concepts/model-providers/) for details. Change the model in the setup cell below.

---

## Research Papers Validated

| Paper | Key Finding | Demo |
|-------|------------|------|
| [MemoryOS of AI Agent](https://arxiv.org/abs/2506.06326) | Hierarchical memory: +49% F1 on LoCoMo | 01 |
| [Cognitive Memory in LLMs](https://arxiv.org/abs/2504.02441) | Survey: sensory, short-term, long-term memory | 01 |
| [MIRIX: Multi-Agent Memory](https://arxiv.org/abs/2507.07957) | 6 memory types, SOTA 85.4% on LOCOMO | 02 |
| [MemGPT](https://arxiv.org/abs/2310.08560) | Core memory concept, virtual context management | 02 |
| [Enabling Personalized Interactions](https://arxiv.org/abs/2510.07925) | Persistent memory + evolving user profiles | 02 |
| [Zep Temporal Knowledge Graph](https://arxiv.org/abs/2501.13956) | 94.8% DMR, 18.5% accuracy improvement | 03 |
| [PersonaAgent with GraphRAG](https://arxiv.org/abs/2511.17467) | +56.1% F1 on movie tagging via graph personas | 03 |
| [HippoRAG 2](https://arxiv.org/abs/2502.14802) | +7% associative memory via Personalized PageRank | 03 |
| [GAAMA: Graph Augmented Associative Memory for Agents](https://arxiv.org/abs/2603.27910) | Concept-mediated knowledge graph + kNN/PageRank retrieval | 04 |
| [MAGMA: A Multi-Graph based Agentic Memory Architecture](https://arxiv.org/abs/2601.03236) | Semantic/temporal/causal/entity graphs + policy-guided traversal | 04 |
| [GRAVITY: Structured Anchoring for Long-Horizon Memory](https://arxiv.org/abs/2605.01688) | Entity profiles grounded in relational graphs | 04 |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING](CONTRIBUTING.md) for more information.

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file for details.
