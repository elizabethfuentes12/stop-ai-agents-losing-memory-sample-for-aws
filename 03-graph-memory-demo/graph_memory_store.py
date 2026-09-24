"""A native Strands ``MemoryStore`` backed by a Neo4j knowledge graph.

This lets Demo 03 run on Strands' native memory framework, the same way Demo 04
does for vectors: the ``MemoryManager`` orchestrates *when* to extract, *how* to
extract, and injects retrieved memories into the model, but it needs somewhere to
persist and search. That "somewhere" is a ``MemoryStore``. This class implements
that contract against a real graph so recall is by traversal, not keyword.

Native contract implemented (see the installed source and the official docs):
  - ``search(query, options) -> list[MemoryEntry]``  (similarity entry point,
    then a Cypher traversal across relationships to the connected answer)
  - ``add(content, metadata) -> Any``                (the write sink: an LLM
    extraction pipeline turns a sentence into typed nodes and edges against a
    pinned schema)
  plus the declarative attributes a store exposes: ``name``, ``description``,
  ``max_search_results``, ``writable``, ``extraction``.

Docs:
  https://strandsagents.com/docs/api/python/strands.memory.memory_manager/
  https://strandsagents.com/docs/api/python/strands.memory.types/   (MemoryStore, MemoryEntry)

The graph I/O (build pipeline + traverse) reuses ``graph_memory.py``. The neo4j
``SimpleKGPipeline`` is already async; the ``VectorCypherRetriever`` is synchronous,
so ``search`` runs it in a worker thread with ``asyncio.to_thread`` to honor the
async contract without blocking the event loop.
"""

from __future__ import annotations

import asyncio

from strands.memory import MemoryEntry, MemoryStore, SearchOptions
from strands.memory.types import Metadata

import graph_memory as gm

DEFAULT_MAX_SEARCH_RESULTS = 3


class GraphMemoryStore(MemoryStore):
    """A Strands ``MemoryStore`` whose recall is graph traversal (Neo4j).

    One instance wraps one Neo4j database. ``add`` runs the LLM extraction pipeline
    (sentence -> typed nodes/edges against a pinned schema); ``search`` finds an entry
    chunk by similarity, then traverses the relationships to the connected answer.

    Args:
        name: Unique store name (the ``MemoryManager`` targets stores by name; it is
            also surfaced on retrieved ``MemoryEntry.store_name``).
        driver: An open Neo4j driver (from ``graph_memory.get_driver()``).
        db: The Neo4j database name (from ``graph_memory.ensure_database()``).
        embedder: The embedder used for chunk similarity (defaults to the module's).
        description: Human-readable description, folded into tool descriptions.
        extraction: The store's automatic-extraction config (``ExtractionConfig``
            or ``True``/``False``). A store implementing ``add`` uses a client-side
            ``ModelExtractor``; the selection prompt lives there.
        max_search_results: Default recall size when a caller passes none.
    """

    def __init__(self, name, driver, db, embedder=None, description=None,
                 extraction=None, max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS,
                 mode: str = "graph"):
        # Declarative MemoryStore attributes (the Protocol re-declares these).
        self.name = name
        self.description = description
        self.max_search_results = max_search_results
        self.writable = True
        self.extraction = extraction

        if mode not in ("graph", "semantic"):
            raise ValueError("mode must be 'graph' (traversal) or 'semantic' (similarity only)")
        self._mode = mode
        self._driver = driver
        self._db = db
        self._embedder = embedder or gm.get_embedder()
        self._seen: set[str] = set()  # de-dup guard: extraction writes are at-least-once

    async def initialize(self) -> None:
        """Async setup hook the ``MemoryManager`` calls during ``init_agent``.

        The database and vector index are provisioned by ``ensure_database`` /
        ``seed_graph`` before the store is constructed, so nothing to resolve here.
        """
        return None

    async def add(self, content: str, metadata: Metadata | None = None):
        """Write one fact into the graph: the LLM pipeline extracts typed nodes and edges.

        Called by the ``ModelExtractor`` for each fact it keeps. Extraction is
        at-least-once, so identical content is de-duplicated to avoid re-running the
        pipeline on the same sentence.
        """
        text = content.strip()
        if not text or text in self._seen:
            return {"stored": False}
        self._seen.add(text)

        pipeline = gm.build_pipeline(self._driver, self._db, embedder=self._embedder)
        await pipeline.run_async(text=text)

        # Keep the chunk vector index current for the newly written chunk.
        from neo4j_graphrag.indexes import create_vector_index
        create_vector_index(self._driver, gm.VECTOR_INDEX_NAME, label=gm.CHUNK_LABEL,
                            embedding_property="embedding", dimensions=gm.EMBED_DIM,
                            similarity_fn="cosine", neo4j_database=self._db)
        return {"stored": True}

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Recall: 'graph' mode traverses to the connected answer; 'semantic' mode
        returns similar chunks only (no traversal), for the semantic-vs-graph contrast.

        Returns ``MemoryEntry`` objects (the native result type). The
        ``MemoryManager`` sets ``store_name``.
        """
        limit = (options or {}).get("max_search_results") or self.max_search_results

        def _query():
            if self._mode == "graph":
                retriever = gm.make_graph_retriever(self._driver, self._db, self._embedder)
            else:
                retriever = gm.make_semantic_retriever(self._driver, self._db, self._embedder)
            result = retriever.search(query_text=query, top_k=limit)
            return [item.content for item in result.items]

        hits = await asyncio.to_thread(_query)
        return [MemoryEntry(content=text) for text in hits]
