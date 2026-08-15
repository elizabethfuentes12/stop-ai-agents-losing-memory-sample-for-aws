"""The four memory stores this demo compares — key-value, FAISS, Amazon S3 Vectors,
and Amazon DynamoDB Vector Search.

The dividing line between Demo 01 and this demo:

  - You KNOW the key ("what's my preferred cabin?")  -> key-value memory (Demo 01).
  - You only know the MEANING ("what should I avoid eating on this trip?") ->
    semantic search: embed once at write time, embed the question at query time,
    retrieve by cosine similarity. The stored text never needs to share words with the question.

Three vector backends, same embeddings, same memories — so the measured difference
is the backend, not the data:

  - FAISS (in-process): the index lives in RAM during the process. Microsecond queries,
    zero infrastructure. Not durable: index is gone when the process exits.
  - Amazon S3 Vectors (managed vector storage): dedicated vector bucket + index. Persists
    across restarts, reachable from any process with credentials. ~170-200 ms per query.
  - Amazon DynamoDB Vector Search (GA 2025): vector index added to a DynamoDB table.
    The same table can hold your operational data alongside embeddings — no separate
    vector store to provision. Uses the SearchVectors API (added in boto3 1.43.72).
    Single-digit millisecond latency, fully serverless, on-demand billing only.

Embeddings: Amazon Titan Text Embeddings V2 via Bedrock (1024 dims), the same
embedder for all vector backends. boto3 clients are built from an explicit profile
(AWS_PROFILE or default chain) so stray env tokens can't hijack the session.

Self-provisioning (series rule): every AWS resource the demo needs is created
by the demo itself if it doesn't exist — no console steps.

Requires boto3 >= 1.43.72 (SearchVectors was added in that release).
"""

import json
import os
import threading
import time

import boto3
import faiss
import numpy as np

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024

VECTOR_BUCKET = os.getenv("VECTOR_BUCKET", "agent-memory-demo-vectors")
VECTOR_INDEX = os.getenv("VECTOR_INDEX", "traveler-memories")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "agent-memory-demo-ddb")
DYNAMODB_VECTOR_INDEX = os.getenv("DYNAMODB_VECTOR_INDEX", "memory-vector-index")
DYNAMODB_VECTOR_ATTR = os.getenv("DYNAMODB_VECTOR_ATTR", "embedding")
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


# ── Store 4: Amazon DynamoDB Vector Search ───────────────────────────────────
class DynamoDBVectorStore:
    """Vector memory in a DynamoDB table with a native vector index.

    Vectors are stored as regular DynamoDB items (primary key + text + embedding
    List attribute). Queries use the SearchVectors API — approximate nearest neighbor
    at single-digit millisecond latency, fully serverless, on-demand billing.

    The key difference vs S3 Vectors: the vector index lives inside an existing
    DynamoDB table, so your operational data and embeddings share one service and
    one billing model. No separate vector bucket to manage.

    Score note: SearchVectors returns a COSINE distance (1 − cosine_similarity).
    For L2-normalized embeddings (Titan V2 is normalized), this is in [0, 1].
    This class returns 1.0 − score so the output is cosine_similarity, making
    it directly comparable to FAISS (inner product of normalized vectors) and
    S3 Vectors (which also returns 1.0 − distance).

    Requires boto3 >= 1.43.72 (SearchVectors was added in that release).
    """

    def __init__(self, table: str = DYNAMODB_TABLE, index: str = DYNAMODB_VECTOR_INDEX,
                 vector_attr: str = DYNAMODB_VECTOR_ATTR):
        self.table = table
        self.index = index
        self.vector_attr = vector_attr
        self.client = _aws().client("dynamodb", region_name=AWS_REGION)
        self._ensure()

    def _ensure(self) -> None:
        """Create the table and vector index if they don't exist (idempotent)."""
        try:
            desc = self.client.describe_table(TableName=self.table)["Table"]
        except self.client.exceptions.ResourceNotFoundException:
            self.client.create_table(
                TableName=self.table,
                # Vector indexes require on-demand capacity mode.
                BillingMode="PAY_PER_REQUEST",
                KeySchema=[{"AttributeName": "memory_key", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "memory_key", "AttributeType": "S"}],
                VectorIndexes=[{
                    "IndexName": self.index,
                    "VectorAttribute": {"AttributeName": self.vector_attr},
                    "Dimensions": EMBED_DIM,
                    "DistanceFunction": "COSINE",
                    "Projection": {"ProjectionType": "ALL"},
                }],
            )
            print(f"  created DynamoDB table {self.table!r} with vector index {self.index!r}")
            self._wait_table_active()
            return

        # Table exists — add the vector index if it's missing.
        existing = {vi["IndexName"] for vi in desc.get("VectorIndexes", [])}
        if self.index not in existing:
            self.client.update_table(
                TableName=self.table,
                VectorIndexUpdates=[{"Create": {
                    "IndexName": self.index,
                    "VectorAttribute": {"AttributeName": self.vector_attr},
                    "Dimensions": EMBED_DIM,
                    "DistanceFunction": "COSINE",
                    "Projection": {"ProjectionType": "ALL"},
                }}],
            )
            print(f"  added vector index {self.index!r} to existing table {self.table!r}")

        self._wait_index_active()

    def _wait_table_active(self, timeout_s: float = 60.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status = self.client.describe_table(
                TableName=self.table)["Table"]["TableStatus"]
            if status == "ACTIVE":
                self._wait_index_active()
                return
            threading.Event().wait(1.0)
        raise TimeoutError(f"DynamoDB table {self.table!r} not ACTIVE after {timeout_s:.0f}s")

    def _wait_index_active(self, timeout_s: float = 60.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            desc = self.client.describe_table(TableName=self.table)["Table"]
            for vi in desc.get("VectorIndexes", []):
                if vi["IndexName"] == self.index:
                    if vi["IndexStatus"] == "ACTIVE" and not vi.get("Backfilling", False):
                        return
                    break
            threading.Event().wait(1.0)
        raise TimeoutError(f"Vector index {self.index!r} not ACTIVE after {timeout_s:.0f}s")

    def put(self, key: str, text: str, vector: list[float]) -> None:
        """Write one item: primary key, text, and embedding as a DynamoDB List of N."""
        self.client.put_item(
            TableName=self.table,
            Item={
                "memory_key": {"S": key},
                "text": {"S": text},
                self.vector_attr: {"L": [{"N": str(float(f))} for f in vector]},
            },
        )

    def query(self, vector: list[float], top_k: int = 3) -> list[tuple[str, float]]:
        """Return (text, cosine_similarity) for the top-k most similar items.

        SearchVector uses DynamoDB's AttributeValue list format.
        Returned Score is COSINE distance (lower = more similar). Converting with
        1.0 − score gives cosine_similarity, comparable to FAISS and S3 Vectors.
        """
        resp = self.client.search_vectors(
            TableName=self.table,
            IndexName=self.index,
            SearchVector=[{"N": str(float(f))} for f in vector],
            TopK=top_k,
        )
        return [
            (r["Item"]["text"]["S"], 1.0 - r["Score"])
            for r in resp.get("SearchResults", [])
        ]

    def count(self) -> int:
        resp = self.client.scan(TableName=self.table, Select="COUNT")
        return resp["Count"]

    def clear(self) -> int:
        """Delete all items so reruns start empty (table and index stay)."""
        resp = self.client.scan(
            TableName=self.table,
            ProjectionExpression="memory_key",
        )
        items = resp.get("Items", [])
        for item in items:
            self.client.delete_item(
                TableName=self.table,
                Key={"memory_key": item["memory_key"]},
            )
        return len(items)


def timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_ms) — the demo measures, never guesses."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000
