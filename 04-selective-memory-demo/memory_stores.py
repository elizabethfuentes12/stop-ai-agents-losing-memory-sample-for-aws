"""The three memory stores this demo compares — key-value, FAISS, Amazon S3 Vectors.

The dividing line between Demo 01 and this demo:

  - You KNOW the key ("what's my preferred cabin?")  -> key-value memory (Demo 01).
  - You only know the MEANING ("what should I avoid eating on this trip?") ->
    vector memory: embed the memories once, embed the question, retrieve by
    similarity. The stored text never needs to share words with the question.

Two vector backends, same embeddings, same memories — so the measured difference
is the backend, not the data:

  - FAISS (in-process): the index lives in RAM. Microsecond queries, zero
    infrastructure — and it dies with the Python process.
  - Amazon S3 Vectors (managed storage): the index lives in a vector bucket.
    You create the bucket + index (this module self-provisions both if missing),
    write with put_vectors, query with query_vectors. It survives restarts and
    is reachable from any process with credentials.

Embeddings: Amazon Titan Text Embeddings V2 via Bedrock (1024 dims), the same
embedder for both backends. boto3 clients are built from an explicit profile
(AWS_PROFILE or default chain) so stray env tokens can't hijack the session.

Self-provisioning (series rule): every AWS resource the demo needs is created
by the demo itself if it doesn't exist — no console steps.
"""

import json
import os
import time

import boto3
import faiss
import numpy as np

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024

VECTOR_BUCKET = os.getenv("VECTOR_BUCKET", "agent-memory-demo-vectors")
VECTOR_INDEX = os.getenv("VECTOR_INDEX", "traveler-memories")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


# ── AWS clients (explicit session so env bearer tokens can't override) ───────
_session = None


def _aws():
    global _session
    if _session is None:
        profile = os.getenv("AWS_PROFILE")
        _session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return _session


def embed(text: str) -> list[float]:
    """Real Titan V2 embedding (1024 dims) — used by BOTH vector backends."""
    client = _aws().client("bedrock-runtime", region_name=AWS_REGION)
    resp = client.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": EMBED_DIM}),
    )
    return json.loads(resp["body"].read())["embedding"]


# ── Store 1: key-value (Demo 01's memory — the baseline) ─────────────────────
class KeyValueStore:
    """Plain dict store: perfect when you know the key, blind to meaning."""

    def __init__(self):
        self.data: dict[str, str] = {}

    def put(self, key: str, text: str) -> None:
        self.data[key] = text

    def get(self, key: str) -> str | None:
        return self.data.get(key)

    def keyword_search(self, query: str) -> list[str]:
        """The best a key-value store can do without a key: substring matching."""
        words = {w.lower().strip("?.,!") for w in query.split() if len(w) > 3}
        return [text for text in self.data.values()
                if any(w in text.lower() for w in words)]

    def dump_all(self) -> str:
        return "\n".join(self.data.values())


# ── Store 2: FAISS (in-process vector index) ─────────────────────────────────
class FaissStore:
    """Vector memory in RAM: cosine similarity via a normalized inner-product index."""

    def __init__(self):
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.texts: list[str] = []

    def put(self, text: str, vector: list[float]) -> None:
        v = np.array([vector], dtype="float32")
        faiss.normalize_L2(v)
        self.index.add(v)
        self.texts.append(text)

    def query(self, vector: list[float], top_k: int = 3) -> list[tuple[str, float]]:
        v = np.array([vector], dtype="float32")
        faiss.normalize_L2(v)
        scores, ids = self.index.search(v, top_k)
        return [(self.texts[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


# ── Store 3: Amazon S3 Vectors (managed vector storage) ─────────────────────
class S3VectorStore:
    """Vector memory in a vector bucket: persists across restarts, shared across processes."""

    def __init__(self, bucket: str = VECTOR_BUCKET, index: str = VECTOR_INDEX):
        self.bucket = bucket
        self.index = index
        self.client = _aws().client("s3vectors", region_name=AWS_REGION)
        self._ensure()

    def _ensure(self) -> None:
        """Create the vector bucket and index if they don't exist (idempotent)."""
        try:
            self.client.get_vector_bucket(vectorBucketName=self.bucket)
        except self.client.exceptions.NotFoundException:
            self.client.create_vector_bucket(vectorBucketName=self.bucket)
            print(f"  created vector bucket {self.bucket}")
        try:
            self.client.get_index(vectorBucketName=self.bucket, indexName=self.index)
        except self.client.exceptions.NotFoundException:
            self.client.create_index(
                vectorBucketName=self.bucket, indexName=self.index,
                dimension=EMBED_DIM, distanceMetric="cosine", dataType="float32",
            )
            print(f"  created vector index {self.index} ({EMBED_DIM} dims, cosine)")

    def put(self, key: str, text: str, vector: list[float]) -> None:
        self.client.put_vectors(
            vectorBucketName=self.bucket, indexName=self.index,
            vectors=[{"key": key, "data": {"float32": vector},
                      "metadata": {"text": text}}],
        )

    def query(self, vector: list[float], top_k: int = 3) -> list[tuple[str, float]]:
        resp = self.client.query_vectors(
            vectorBucketName=self.bucket, indexName=self.index,
            queryVector={"float32": vector}, topK=top_k,
            returnDistance=True, returnMetadata=True,
        )
        return [(v["metadata"]["text"], 1.0 - v["distance"]) for v in resp.get("vectors", [])]

    def count(self) -> int:
        resp = self.client.list_vectors(vectorBucketName=self.bucket, indexName=self.index)
        return len(resp.get("vectors", []))

    def clear(self) -> int:
        """Delete this demo's vectors so reruns start empty (bucket/index stay)."""
        resp = self.client.list_vectors(vectorBucketName=self.bucket, indexName=self.index)
        keys = [v["key"] for v in resp.get("vectors", [])]
        if keys:
            self.client.delete_vectors(vectorBucketName=self.bucket,
                                       indexName=self.index, keys=keys)
        return len(keys)


def timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_ms) — the demo measures, never guesses."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000
