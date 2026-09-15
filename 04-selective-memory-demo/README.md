# AI Agent Memory: What to Store and What to Throw Away

![AI agent memory, what to store and what to throw away: a robot files useful facts, preferences, and events by type while throwing small talk, weather, and opinions into the trash](images/ai-agent-selective-memory-cover.png)

**Problem:** Everyone races to make agents remember *more*, but the agent that wins keeps the right things and throws the rest away. Store everything and memory becomes expensive, slow, and dirty ([Demo 05](../05-memory-hygiene-demo/) shows it's also dangerous); store nothing and the agent forgets its user (Demo 01, Test 1).

**Solution:** **Selection** (memory extraction) decides what to keep, in which memory *type*, and what to throw away. This demo builds it three ways and measures them against the same planted conversation with deterministic ground truth.

The first two mechanisms run entirely on **Strands Agents' native memory framework** with no hand-rolled memory tools, no memory logic in the chat agent's system prompt. The third is fully managed by AWS ([Amazon Bedrock AgentCore Memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/built-in-strategies.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)). Storage is [Amazon S3 Vectors](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) or [Amazon DynamoDB Vector Search](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/vector-search.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el).

![Selective memory architecture: native MemoryManager with one store (A) or four typed stores (B), plus managed AgentCore Memory (C)](images/ai-agent-selective-memory-architecture.png)

![Meme: a robot overwhelmed by hoarding every message versus a calm robot with a few tidy typed memory trays, discarding the junk](images/ai-agent-selective-memory-meme.png)

---

## What this demo uses from Strands (native memory framework)

Everything memory-related is the SDK's job. The pieces (all from `strands.memory`):

| Native Strands piece | What it does (and why it's notable) | Official docs |
|----------------------|-------------------------------------|---------------|
| [`MemoryManager`](https://strandsagents.com/docs/api/python/strands.memory.memory_manager/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) | One plugin attached with `Agent(memory_manager=...)` that wires three behaviors at once: registers a `search_memory` tool, runs background extraction, and injects retrieved memory into the model. `add()` writes to all writable stores concurrently and surfaces per-store failures. | [memory_manager](https://strandsagents.com/docs/api/python/strands.memory.memory_manager/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) |
| `MemoryStore` (contract) | Where memories live and how they're searched: a small `add` / `search` contract. `VectorMemoryStore` implements it over Titan V2 + a vector index, so the backend (S3 Vectors / DynamoDB) is a swappable detail. | [strands.memory.types](https://strandsagents.com/docs/api/python/strands.memory.types/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) |
| [`ModelExtractor`](https://strandsagents.com/docs/api/python/strands.memory.extraction.model_extractor/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) | Client-side extraction: a **separate** model call (can be a cheaper model than the chat) whose `system_prompt` IS the keep/discard policy. Parses tolerantly: a bad reply means "nothing kept," never a crash; `[]` is a first-class discard. | [model_extractor](https://strandsagents.com/docs/api/python/strands.memory.extraction.model_extractor/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) |
| `ExtractionConfig` | Binds *when* (trigger) + *how* (extractor) + *what to feed* (message filter) to a store. Smart default: `add`-only stores extract client-side via `ModelExtractor`; `add_messages` stores extract server-side (no model call). Default filter strips tool traffic so tool JSON never enters memory. | [strands.memory.types](https://strandsagents.com/docs/api/python/strands.memory.types/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) |
| `IntervalTrigger` / `InvocationTrigger` | Control *when* extraction fires (every N turns / every turn), **off the conversation turn**. A per-store high-water mark delivers each message at most once, writes are serialized per store, and a repeatedly-failing store backs off automatically. | [strands.memory.types](https://strandsagents.com/docs/api/python/strands.memory.types/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) |
| Memory **injection** (default on) | Before each model call, folds retrieved memory into the model input **without touching durable history**. Adaptive query from the latest user ask, source-attributed rendering, and **fails open** (skips injection rather than breaking the call). | [memory_manager](https://strandsagents.com/docs/api/python/strands.memory.memory_manager/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) |

You own two things only: the extractor's **selection prompt** and the **store**. The chat agent's system prompt stays about the agent's job.

```python
from strands import Agent
from strands.memory import MemoryManager, ModelExtractor, ExtractionConfig, IntervalTrigger

store = VectorMemoryStore(
    name="traveler_memory",
    extraction=ExtractionConfig(
        trigger=[IntervalTrigger(turns=1)],                 # when (off the turn)
        extractor=ModelExtractor(model=extractor_model,     # how
                                 system_prompt=SELECTION_PROMPT),  # the policy you own
    ),
)
agent = Agent(model=MODEL, system_prompt="You are a flight assistant.",  # persona only
              memory_manager=MemoryManager(stores=[store]))
```

> **Store choice.** Strands ships a zero-setup [`TestMemoryStore`](https://strandsagents.com/docs/api/python/strands.vended_memory_stores.test_memory_store.store/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) (persists to a local JSON file, lexical recall), the canonical starting point in the [Memory overview](https://strandsagents.com/docs/user-guide/concepts/memory/overview/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el). This demo instead implements a custom `MemoryStore` (`VectorMemoryStore`) so recall is **semantic** (Titan V2 + a vector index), which is the whole point of Demo 02.

> **You don't always have to build the store.** The [Strands integrations directory](https://strandsagents.com/integrations/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) lists ready-made `MemoryStore` backends you can drop in instead of writing one:
> - **Amazon Bedrock Knowledge Base** and **AgentCore Memory Store** (AWS, built-in / featured): managed semantic recall and server-side extraction.
> - **`s3-vectors-memory`** (aws-samples) and **`strands-dynamodb-storage`** (aws): the same S3 Vectors / DynamoDB backends this demo builds by hand, packaged.
> - Partner stores: **Mem0**, **Zep** (temporal knowledge graph), **Vectorize Hindsight**, **Neo4j** (graph memory), **Dakera**.
>
> We implement the store ourselves here to *teach the `MemoryStore` contract* (`add` + `search`); in production you'd often reach for one of the above.

---

## The three mechanisms

| Mechanism | What it is | Who owns the policy | Partitions |
|-----------|------------|---------------------|------------|
| **A: native, one store** | `MemoryManager` + one `VectorMemoryStore` | you (one general prompt) | one |
| **B: native, four typed stores** | `MemoryManager` + four `VectorMemoryStore`s | you (one prompt per type) | four (facts / prefs / trip_summary / episodes) |
| **C: Amazon Bedrock AgentCore Memory** | fully managed by AWS | AWS (managed, or override) | managed |

**A vs B is granularity, not backend.** A is the simplest native setup (one store, one prompt); B reproduces AgentCore's per-type partitioning with the native SDK, so the criteria AgentCore ships built-in become text you own. **The vector backend is a separate lever** (`VECTOR_BACKEND=s3|dynamodb`) that applies to both A and B. C is the fully managed counterpart: send raw turns via the official session manager, AWS extracts.

The four memory types run through the whole series:

| Type | This demo (A & B) | AgentCore built-in (C) |
|------|-------------------|------------------------|
| facts | `facts` store / `selective-facts` | `semanticMemoryStrategy` |
| preferences | `preferences` / `selective-prefs` | `userPreferenceMemoryStrategy` |
| trip_summary | `trip_summary` / `selective-summary` | `summaryMemoryStrategy` |
| episodes | `episodes` / `selective-episodes` | `episodicMemoryStrategy` |

---

## The measured result (your numbers will vary)

![Storing everything is not memory quality: a jar that stores everything reaches perfect recall but keeps the junk, while a selective jar keeps recall high and drops the noise](images/ai-agent-selective-memory-store-everything-vs-selective.png)

The test conversation mixes **5 keepers** (2 facts, 2 preferences, 1 episode) with **3 decoys** to throw away (small talk, a passing opinion, ephemeral weather). The score is **selection recall**: how many of the 5 keepers a mechanism stored, checked deterministically (no LLM judge). The decoys are there so a mechanism cannot win by hoarding.

All three run on the native `MemoryManager`; what changes is who writes the keep/throw-away policy. From 20 runs each for A and B, repeated runs for C (gpt-4o-mini; recall varies run to run, your numbers will differ):

| Mechanism | Selection recall | Who owns the policy | When queryable |
|-----------|:---------------:|---------------------|----------------|
| A: native, one store | ~3.9/5 | you (one prompt) | when the turn returns (~3.2 s) |
| B: native, four typed stores | **~5/5** | you (one prompt per type) | when the turn returns (~3.5 s) |
| C: Amazon Bedrock AgentCore Memory | **5/5** | AWS (managed, or override) | ~20-55 s later (async) |

**What the numbers teach:**
- **All three recall the keepers well.** The difference is how much of the selection policy you hold, not which one is "better".
- **B is the sharpest when you own every criterion.** One non-overlapping prompt per type keeps each store to its own kind of memory, so recall lands ~5/5 every run. Choose B when the keep/throw-away rules are yours to define and tune per type.
- **A is the same idea with one prompt.** You own the policy at a coarser grain; recall runs a touch lower and noisier (~3.9/5). Choose A when one flat memory is enough.
- **C lets AWS run the pipeline.** You send raw turns and the managed strategies extract, embed, and index server-side, with nothing to maintain. Extraction is asynchronous, so the memory lands ~20-55 s after the turn. If you want to shape what it keeps, use [custom strategies with prompt overrides](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/long-term-configuring-custom-strategies.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el).
- **B reproduces AgentCore's per-type partitioning with the native SDK**, so the built-in criteria become text you own.

## The decision table

| Situation | Pick |
|-----------|------|
| Least setup; one selection prompt is enough | **A**: native, one store |
| Per-type control of the keep/discard criteria | **B**: native, four typed stores |
| Production multi-user; ~1 min lag is fine; no pipeline to maintain | **C**: AgentCore Memory |

**Backend is orthogonal:** `VECTOR_BACKEND=s3` (Amazon S3 Vectors) or `dynamodb` (Amazon DynamoDB Vector Search), the same two backends from [Demo 02](../02-vector-memory-demo/). Selection decides *what*; the backend decides *where*.

---

## Quick Start

### Prerequisites

- Python 3.10+
- **AWS credentials** (`aws configure`) for Titan embeddings, the vector backend (S3 Vectors or DynamoDB), AgentCore Memory. **All resources are created automatically if missing.**
- **`OPENAI_API_KEY`**: the chat agent and the native `ModelExtractor`. Both swappable to Amazon Bedrock via the commented blocks.

```bash
cp .env.example .env   # set OPENAI_API_KEY
```

### Installation

```bash
uv venv && uv pip install -r requirements.txt
```

### Deterministic vs model-based

The control lives in the agent's harness: a native `MemoryManager` over the stores. The selection-recall scoring and the stores are deterministic. The `ModelExtractor` deciding what to keep (its prompt is your policy) and the embeddings are model inference, with no reproducibility guarantee across runs ([research](https://arxiv.org/abs/2601.17768)). The selection policy is text you own; the decision it drives is a model call, so recall varies run to run. The scoring that grades it is deterministic.

## Run Demo

```bash
uv run python test_selective_memory.py

# Interactive notebook
# Open test_selective_memory.ipynb in Jupyter, JupyterLab, or VS Code
```

### Chat with each mechanism (interactive)

One chat script per mechanism. Talk to the agent, then watch memory accumulate turn by turn (small talk gets discarded, keepers get stored). Commands: `/memory` and `/quit`.

```bash
uv run python chat_single_store.py    # A: one store, one selection prompt
uv run python chat_typed_stores.py    # B: four typed stores, memory shown per type
uv run python chat_agentcore.py       # C: AgentCore managed (watch the async lag with /memory then /wait)
```

The single-store and typed chats start from an empty store so you see memory fill from zero (in production you would not clear). The AgentCore chat keeps its managed memory.

### Choose the vector backend (mechanisms A & B, and their chats)

```bash
# .env
VECTOR_BACKEND=s3          # Amazon S3 Vectors (default), one index per store partition
# VECTOR_BACKEND=dynamodb  # Amazon DynamoDB Vector Search, one table per partition (needs boto3>=1.43.72)
```

The backend is orthogonal to the mechanism: A, B, and their chats all honor `VECTOR_BACKEND`. C (AgentCore) has its own managed store and ignores it. The notebook's **Step 5b** runs the typed-store setup on DynamoDB in one extra cell (it flips the backend, runs, and restores it), so you can see both backends without editing `.env`.

**Which backend, and why.** Same Titan V2 embeddings either way, so recall quality is identical; the choice is where the vectors live:
- **S3 Vectors** ([docs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)): a dedicated vector bucket, separate from operational data. AWS positions it for cost-optimized storage at massive scale with infrequent access; query latency is sub-second (~100 ms or less for frequent queries, higher when cold). Use it when memory is a standalone concern, the corpus is large or archival, and sub-second (not sub-10 ms) latency is fine.
- **DynamoDB Vector Search** ([docs](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/VectorSearch.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)): the vector index lives inside a DynamoDB table via the `VectorIndexes` parameter on `CreateTable`/`UpdateTable`, queried with the `SearchVectors` API. AWS states single-digit-ms latency at 99%+ recall. Use it for real-time retrieval or to keep data and memory collocated in one service. Needs a recent boto3 with the `SearchVectors` API.

The embedding call dominates end-to-end latency for both, so neither is "faster memory" the user feels; pick by architecture, not speed.

**Cost/cleanup note:** mechanism C creates one AgentCore memory (name `SelectiveMemoryDemo`) and writes ~6 events per run; events expire after 7 days. Delete it with `aws bedrock-agentcore-control delete-memory --memory-id <id>`. On `VECTOR_BACKEND=dynamodb`, A & B create on-demand DynamoDB tables (`selective-memory-*`); drop them with `aws dynamodb delete-table --table-name <name>`. The notebook's last cell also cleans up.

---

## Files

| File | Purpose |
|------|---------|
| `test_selective_memory.py` | Main demo: the 3 mechanisms + measured comparison |
| `test_selective_memory.ipynb` | Interactive walkthrough |
| `chat_single_store.py` | Chat with Mechanism A (one store) |
| `chat_typed_stores.py` | Chat with Mechanism B (four typed stores, memory shown per type) |
| `chat_agentcore.py` | Chat with Mechanism C (AgentCore managed; shows the async extraction lag) |
| `vector_memory_store.py` | `VectorMemoryStore`: the native `MemoryStore` contract over Titan V2 + S3 Vectors / DynamoDB. The `MemoryManager` + `ModelExtractor` selection prompts are wired inline in `test_selective_memory.py` and each `chat_*.py`. |
| `memory_stores.py` | The synchronous S3 Vectors + DynamoDB Vector Search backends + Titan embedder (from Demo 02) |
| `tools.py` | The agent's domain tools only (flights, climate): memory is not handled here |
| `agentcore_memory.py` | Mechanism C: ensure_memory + create_event + retrieve + lag measurement |
| `requirements.txt` | Dependencies (`strands-agents[openai]`, `boto3>=1.43.72`) |

---

## How It Works

### A: native MemoryManager, one store

One `VectorMemoryStore` + one general `ModelExtractor` prompt. The `MemoryManager` runs extraction off the turn (`IntervalTrigger`), stores survivors, and injects recalled memory into the model. The selection prompt is the whole keep/discard policy:

```python
SELECTION_PROMPT = (
    "You extract durable memories worth keeping... KEEP durable facts, stated "
    "preferences, notable events. DISCARD small talk, weather, passing opinions. "
    'Return ONLY a JSON array of {"content": string}, or [] if nothing.'
)
```

### B: native MemoryManager, four typed stores

Four `VectorMemoryStore`s, each with its own `ModelExtractor` prompt and its own vector partition: AgentCore's per-strategy partitioning, native SDK. Each prompt keeps one type and returns `[]` for the rest.

### When is a memory saved? (`flush`)

Extraction runs in the background, so the last turn's memory may not be persisted when the agent finishes responding. `await manager.flush()` forces every store to save its buffered messages and waits for those writes (the synchronization point for a graceful shutdown. **This demo uses the synchronous `agent("...")` path, where the framework flushes after each invocation, so we never call `flush()` manually.** With the async APIs (`invoke_async` / `stream_async`) you'd `await memory_manager.flush()` yourself at shutdown. (Don't flush every turn alongside a periodic trigger) it defeats the trigger's schedule.)

### Practical notes for the managed path (C)

Wiring up Amazon Bedrock AgentCore Memory through the [official Strands session manager](https://strandsagents.com/docs/integrations/session-managers/agentcore-memory/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) (`AgentCoreMemorySessionManager`):

- **Give each strategy an explicit namespace at creation** (`/facts/{actorId}/`, `/preferences/{actorId}/`, `/summaries/{actorId}/{sessionId}/`, `/episodes/{actorId}/{sessionId}/`). The namespace you set on the strategy is the one you reference in `RetrievalConfig`.
- **Use `RetrievalConfig(relevance_score=...)`** to keep only records above a relevance threshold per namespace.
- **Extraction is asynchronous.** The memory became queryable ~20-55 s after the turn (measured, polling until it settled). Plan for eventual consistency.
- **A memory in `CREATING` status isn't ready.** Wait for `ACTIVE` before sending events.
- **To shape what the managed strategies keep,** use [custom strategies with prompt overrides](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/long-term-configuring-custom-strategies.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el): your own prompt and model on top of the managed pipeline.

---

## Learning Objectives

1. Frame memory selection as its own capability: what to keep, in which type, what to throw away
2. Use Strands' **native** memory framework end to end: `MemoryManager`, a `MemoryStore`, a `ModelExtractor`, triggers, and injection
3. Own the selection policy (the extractor's system prompt) while the SDK orchestrates extraction and retrieval
4. Compare framework-managed selection (A, B) against fully managed AWS selection (C: AgentCore)
5. Measure selection recall, who owns the keep/throw-away policy, turn latency, and when the memory is queryable

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ValidationException` creating the AgentCore memory | The episodic strategy needs `reflectionConfiguration.namespaces`: use the demo's `ensure_memory()` |
| `ValidationException` deleting a memory | It's still `CREATING`: wait until `ACTIVE` |
| Extraction never appears (C) | Lag is ~1 min for short conversations; the demo polls up to 7 min. Check the memory is `ACTIVE` |
| `SearchVectors` not found (DynamoDB backend) | Upgrade to `boto3>=1.43.72`, the release that added the DynamoDB `SearchVectors` API |
| Wrong AWS account picked up | `AWS_BEARER_TOKEN*` env vars override profiles; the demo drops them |
| A or B keeps a decoy / misses a keeper | The `ModelExtractor` system prompt is the policy: tune the selection prompts in `test_selective_memory.py` (or the matching `chat_*.py`) |

**Tested versions:** Strands Agents 1.55.1, boto3 1.43.94 (the DynamoDB `SearchVectors` API needs boto3 ≥ 1.43.72), Titan Text Embeddings V2, AgentCore Memory API (2026).

---

## References

- [Strands Agents: Memory (concepts overview)](https://strandsagents.com/docs/user-guide/concepts/memory/overview/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)
- [Strands Agents: `MemoryManager` API](https://strandsagents.com/docs/api/python/strands.memory.memory_manager/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)
- [Strands Agents: memory types (`MemoryStore`, `MemoryEntry`)](https://strandsagents.com/docs/api/python/strands.memory.types/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)
- [Strands Agents: `ModelExtractor`](https://strandsagents.com/docs/api/python/strands.memory.extraction.model_extractor/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)
- [Amazon Bedrock AgentCore Memory: built-in strategies](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/built-in-strategies.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)
- [Amazon S3 Vectors: User Guide](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)
- [Amazon DynamoDB Vector Search: Developer Guide](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/vector-search.html?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el)
- [MIRIX: Multi-Agent Memory System](https://arxiv.org/abs/2507.07957): Wang & Chen, 2025 (typed memory)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560): Packer et al., 2023

---

## Pricing

The three mechanisms use different services, so cost depends on which you run. Mechanism A (agent tools) adds no service beyond the model. Mechanism B stores vectors (S3 Vectors or DynamoDB, your choice via `VECTOR_BACKEND`) and embeds with Titan. Mechanism C uses the managed Bedrock AgentCore Memory. Check the current rates:

| Service | Used for | Pricing |
|---------|----------|---------|
| Amazon Bedrock AgentCore Memory | Mechanism C: managed extraction strategies | [Bedrock AgentCore pricing](https://aws.amazon.com/bedrock/agentcore/pricing/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) |
| Amazon S3 Vectors | Mechanism B: vector store (`VECTOR_BACKEND=s3`) | [S3 pricing](https://aws.amazon.com/s3/pricing/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) |
| Amazon DynamoDB | Mechanism B: vector store (`VECTOR_BACKEND=dynamodb`) | [DynamoDB pricing](https://aws.amazon.com/dynamodb/pricing/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) |
| Amazon Bedrock (Titan Embeddings V2) | Mechanism B: embedding memories and queries | [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el) |
| OpenAI | Default chat model provider | [OpenAI pricing](https://openai.com/api/pricing/) |

For an estimate before you commit spend, use the [AWS Pricing Calculator](https://calculator.aws/?trk=87c4c426-cddf-4799-a299-273337552ad8&sc_channel=el).

---

## Next Steps

1. [Demo 05: Memory Hygiene](../05-memory-hygiene-demo/): the flip side: what an agent must NOT remember
2. [Demo 03: Graph Memory](../03-graph-memory-demo/): a third destination for extracted facts (triples)

---

## Contributing

Contributions are welcome! See [CONTRIBUTING](../CONTRIBUTING.md) for more information.

---

## Security

If you discover a potential security issue in this project, notify AWS/Amazon Security via the [vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

---

## License

This library is licensed under the MIT-0 License. See the [LICENSE](../LICENSE) file for details.
