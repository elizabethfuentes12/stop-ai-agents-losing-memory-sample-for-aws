"""A native Strands ``MemoryStore`` backed by a vector index (Amazon S3 Vectors
or Amazon DynamoDB Vector Search), with Amazon Titan Text Embeddings V2.

This is the piece that lets Demo 04 run entirely on Strands' native memory
framework. The `MemoryManager` orchestrates *when* to extract (triggers), *how*
to extract (a `ModelExtractor` whose system prompt is the selection policy), and
injects retrieved memories into the model, but it needs somewhere to persist and
search. That "somewhere" is a `MemoryStore`, and this class implements that
contract against a real vector backend so recall is semantic, not keyword.

Native contract implemented (see the installed source and the official docs):
  - `search(query, options) -> list[MemoryEntry]`   (semantic recall)
  - `add(content, metadata) -> Any`                 (the write sink the
    `ModelExtractor` calls with each distilled fact; at-least-once, so we
    de-duplicate)
  plus the declarative attributes a store exposes: `name`, `description`,
  `max_search_results`, `writable`, `extraction`.

Docs:
  https://strandsagents.com/docs/api/python/strands.memory.memory_manager/
  https://strandsagents.com/docs/api/python/strands.memory.types/   (MemoryStore, MemoryEntry)

The vector I/O (embed + put + query) reuses the same boto3 stores Demo 02 measures
(`memory_stores.py`), so the only new thing here is the async `MemoryStore`
adapter around them. boto3 is synchronous; we run each call in a worker thread with
`asyncio.to_thread` so we honor the async contract without blocking the event loop.
"""

from __future__ import annotations

import asyncio

from strands.memory import MemoryEntry, MemoryStore, SearchOptions
from strands.memory.types import Metadata

import memory_stores as ms

DEFAULT_MAX_SEARCH_RESULTS = 10


class VectorMemoryStore(MemoryStore):
    """A Strands ``MemoryStore`` whose recall is semantic (Titan V2 + vector search).

    One instance == one vector partition (an S3 Vectors index or a DynamoDB table).
    Use one store for a single flat memory, or several, one per memory type, to
    reproduce AgentCore's per-strategy partitioning with the native SDK.

    Args:
        name: Unique store name (the `MemoryManager` targets stores by name; it is
            also surfaced on retrieved `MemoryEntry.store_name`).
        partition: The vector partition this store writes to (S3 Vectors index
            name, or DynamoDB table suffix). Defaults to ``name``.
        description: Human-readable description, folded into tool descriptions.
        extraction: The store's automatic-extraction config (an ``ExtractionConfig``
            or ``True``/``False``). A store implementing ``add`` (this one does) uses
            a client-side ``ModelExtractor``; the selection prompt lives there.
        max_search_results: Default recall size when a caller passes none.
    """

    def __init__(self, name: str, partition: str | None = None, description: str | None = None,
                 extraction=None, max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS):
        # Declarative MemoryStore attributes (the Protocol re-declares these).
        self.name = name
        self.description = description
        self.max_search_results = max_search_results
        self.writable = True
        self.extraction = extraction

        # The synchronous vector backend (S3 Vectors or DynamoDB) from Demo 02.
        self._backend = ms.make_store(partition or name)
        self._seen: set[str] = set()  # de-dup guard: extraction writes are at-least-once

    async def initialize(self) -> None:
        """Async setup hook the `MemoryManager` calls during ``init_agent``.

        The backend self-provisions its bucket/index or table/index on construction,
        so there is nothing to resolve remotely here; kept for contract completeness.
        """
        return None

    async def add(self, content: str, metadata: Metadata | None = None) -> dict:
        """Write one distilled memory: embed it (Titan V2) and store it in the vector index.

        Called by the `ModelExtractor` for each fact it keeps. Extraction is
        at-least-once, so identical content is de-duplicated to avoid piling up
        repeats across retries.
        """
        text = content.strip()
        if not text or text in self._seen:
            return {"stored": False}
        self._seen.add(text)

        def _write() -> None:
            vector = ms.embed(text)
            # A stable key from the content so a repeat write overwrites in place.
            key = f"{self.name}-{abs(hash(text))}"
            self._backend.put(key, text, vector)

        await asyncio.to_thread(_write)
        return {"stored": True}

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Semantic recall: embed the query and return the nearest stored memories.

        Returns `MemoryEntry` objects (the native result type); the cosine score is
        attached under `metadata["score"]`. The `MemoryManager` sets `store_name`.
        """
        limit = (options or {}).get("max_search_results") or self.max_search_results

        def _query() -> list[tuple[str, float]]:
            return self._backend.query(ms.embed(query), top_k=limit)

        hits = await asyncio.to_thread(_query)
        return [MemoryEntry(content=text, metadata={"score": score}) for text, score in hits]

    # ── demo helpers (not part of the MemoryStore contract) ──────────────────
    def clear(self) -> int:
        """Delete this partition's vectors so reruns start clean."""
        self._seen.clear()
        return self._backend.clear()

    def count(self) -> int:
        return self._backend.count()
