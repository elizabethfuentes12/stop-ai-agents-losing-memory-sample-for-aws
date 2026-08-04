# What Should Your AI Agent Actually Remember? 3 Ways to Build Selective Memory

**Problem:** A real conversation mixes durable facts, throwaway small talk, preferences, and events. Store everything and memory becomes expensive and dirty ([Demo 05](../05-memory-hygiene-demo/) shows it's also dangerous); store nothing and the agent forgets its user (Demo 01, Test 1).

**Solution:** **Selection** — deciding what deserves to persist, in which memory *type*, and what to ignore. This demo builds it three ways and measures them against the same planted conversation with deterministic ground truth.

This demo uses [Strands Agents](https://github.com/strands-agents/sdk-python), [Amazon S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el), and [Amazon Bedrock AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/built-in-strategies.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el). The patterns are framework-agnostic and carry over to other agent frameworks.

---

## The three mechanisms

| Mechanism | Who selects | Where it stores | Selection runs |
|-----------|-------------|-----------------|----------------|
| **A — agent tools** | the conversational agent itself | `agent.state` (key-value) | inline, inside the turn |
| **B — own extractor** | 4 specialized LLM prompts, one per memory type | Amazon S3 Vectors — one index per type | off the conversation path |
| **C — AgentCore Memory** | the 4 built-in managed strategies | AgentCore's managed store | async, inside AWS |

**B is "AgentCore built by hand"** — the same pipeline (extract per type → embed → index per type) with the same partitioning (one index per memory type), except you own the extraction prompts and pay their tokens. C is the managed version: send raw turns via `create_event`, AWS extracts.

The four memory types are the same everywhere — the thread that runs through this series:

| Type | This demo (A and B) | AgentCore built-in (C) |
|------|---------------------|------------------------|
| facts | `facts` section / `selective-facts` index | `semanticMemoryStrategy` |
| preferences | `preferences` / `selective-prefs` | `userPreferenceMemoryStrategy` |
| summary | `trip_summary` / `selective-summary` | `summaryMemoryStrategy` |
| episodes | `episodes` / `selective-episodes` | `episodicMemoryStrategy` |

---

## The measured result (from a real run — yours will vary some)

Planted ground truth: **5 keepers** (2 facts, 2 preferences, 1 episode) and **3 decoys** (small talk, passing opinion, ephemeral weather). Deterministic scoring, no LLM judge.

| Mechanism | Kept | Decoys leaked | Turn overhead | Available after | Extra cost |
|-----------|------|---------------|---------------|-----------------|------------|
| A — agent tools | 4/5 | 0 | ~1.4-1.7 s/turn (selection inside the turn) | immediately | none |
| B — own extractor + S3V | **5/5** | 0 | **0 — off-path** | ~4 s/turn | ~2k tokens / 6 turns |
| C — AgentCore managed | 5/5* | 2* | ~0.4 s/turn (`create_event` only) | **~53 s** (measured) | managed pricing |

\* C's extraction is nondeterministic — across runs it kept 4-5/5 and leaked 0-2 decoys. The built-in criteria aren't yours to tune; that's part of the trade-off.

**What the numbers teach:**
- **A** is free but couples selection to conversation latency, and filing quality rides on the chat model's attention.
- **B** won on selection quality — four single-purpose prompts beat both a multitasking agent and a generic managed extractor — and keeps the conversation path untouched. The price: you own the prompts and pay their tokens.
- **C** has the cheapest write path and zero pipeline to maintain, but extraction lands ~a minute later — a number AWS doesn't publish; this demo measures it.

## The decision table

| Situation | Pick |
|-----------|------|
| Prototype; want memory decisions visible in the conversation | **A** — tools inline |
| You need control of the keep/discard criteria (regulated domain, custom taxonomy) or your own storage | **B** — own extractor |
| Production multi-user; ~1 min extraction lag is fine; no pipeline to maintain | **C** — AgentCore strategies |

**Orthogonal to backends:** selection decides *what* enters memory; the backend decides *where* it lives. A writes to key-value, B to S3 Vectors — either output could target the graph of [Demo 03](../03-graph-memory-demo/) (facts as triples). C bundles pipeline + storage — part of its trade-off.

---

## Quick Start

### Prerequisites

- Python 3.10+
- **AWS credentials** (`aws configure`) — Titan embeddings, S3 Vectors, AgentCore Memory. **All resources are created automatically if missing** (4 vector indexes + 1 AgentCore memory with 4 strategies).
- **`OPENAI_API_KEY`** — the agent (A) and extractor (B). Both swappable to Amazon Bedrock via the commented blocks.

```bash
cp .env.example .env   # set OPENAI_API_KEY
```

### Installation

```bash
uv venv && uv pip install -r requirements.txt
```

### Run Demo

```bash
uv run python test_selective_memory.py

# Interactive notebook
# Open test_selective_memory.ipynb in Jupyter, JupyterLab, or VS Code
```

**Cost/cleanup note:** mechanism C creates one AgentCore memory (name `SelectiveMemoryDemo`) and writes ~6 events per run; events expire after 7 days. The memory resource stays for reruns — delete it with `aws bedrock-agentcore-control delete-memory --memory-id <id>` if you want zero residue.

---

## Files

| File | Purpose |
|------|---------|
| `test_selective_memory.py` | Main demo — the 3 mechanisms + measured comparison |
| `test_selective_memory.ipynb` | Interactive walkthrough |
| `tools.py` | Mechanism A: the four generic memory tools (+ flight tools for reuse) |
| `extractor.py` | Mechanism B: 4 extraction prompts + typed S3 Vectors store |
| `agentcore_memory.py` | Mechanism C: ensure_memory + create_event + retrieve + lag measurement |
| `memory_stores.py` | Shared S3 Vectors store + Titan embedder (from Demo 02) |
| `requirements.txt` | Dependencies |

---

## How It Works

### B — the extraction prompt IS the selection policy

```python
EXTRACTION_PROMPTS = {
    "facts": "Extract durable FACTS about the user's world... NOT preferences, "
             "NOT opinions, NOT small talk. Reply with a raw JSON list, or NOTHING.",
    "preferences": "Extract the user's stated PREFERENCES... NOT objective facts...",
    "summary": "If this turn advances the CURRENT TRIP... ONE sentence, or NOTHING.",
    "episodes": "If this turn contains a notable EVENT (a booking...)... or NOTHING.",
}
```

Each prompt runs against each raw turn, off the conversation path. `NOTHING` is a first-class answer — that's the discard half of selection.

### C — AgentCore facts verified by running (not in the docs)

- `episodicMemoryStrategy` **requires** `reflectionConfiguration.namespaces` — a bare `{"name": ...}` fails with a blank `ValidationException`.
- A memory in `CREATING` status can't be deleted — wait for `ACTIVE`.
- Write path: `create_event(payload=[{"conversational": {"content": {"text": ...}, "role": "USER"}}])`.
- Retrieval namespaces: `/strategies/{strategyId}/actors/{actorId}/` — summary and episodic records live under `/sessions/{sessionId}/`.
- Extraction lag measured at ~53 s for this conversation.

---

## Learning Objectives

1. Frame memory selection as its own capability: what to keep, in which type, what to ignore
2. Compare inline (agent tools) vs off-path (extractor) vs managed (AgentCore) selection
3. Build AgentCore's strategy pipeline by hand: typed extraction prompts → Titan embeddings → per-type S3 Vectors indexes
4. Use AgentCore Memory end-to-end: create_memory with 4 strategies, create_event, retrieve_memory_records
5. Measure what matters: keep/discard quality, turn overhead, availability lag, token cost

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ValidationException` creating the memory | The episodic strategy needs `reflectionConfiguration.namespaces` — use the demo's `ensure_memory()` |
| `ValidationException` deleting a memory | It's still `CREATING` — wait until `ACTIVE` |
| Extraction never appears (C) | Lag is ~1 min for short conversations; the demo polls up to 7 min. Check the memory is `ACTIVE` |
| Wrong AWS account picked up | `AWS_BEARER_TOKEN*` env vars override profiles; the demo drops them |
| B keeps a decoy / misses a keeper | The prompts are the policy — tune them; that's the point of owning the extractor |

**Tested versions:** Strands 1.46.0, boto3 1.43.x, Titan Text Embeddings V2, AgentCore Memory API (July 2026).

---

## References

- [Amazon Bedrock AgentCore Memory — built-in strategies](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/built-in-strategies.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)
- [Amazon S3 Vectors — User Guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)
- [MIRIX: Multi-Agent Memory System](https://arxiv.org/abs/2507.07957) — Wang & Chen, 2025 (typed memory)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) — Packer et al., 2023
- [Strands Agents SDK](https://github.com/strands-agents/sdk-python)

---

## Next Steps

1. [Demo 05: Memory Hygiene](../05-memory-hygiene-demo/) — the flip side: what an agent must NOT remember
2. [Demo 03: Graph Memory](../03-graph-memory-demo/) — a third destination for extracted facts (triples)

---

## Contributing

Contributions are welcome! See [CONTRIBUTING](../CONTRIBUTING.md) for more information.

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file for details.
