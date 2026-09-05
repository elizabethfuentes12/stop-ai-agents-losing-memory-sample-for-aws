"""Vector backends for the extractor (mechanism B): Amazon S3 Vectors and
Amazon DynamoDB Vector Search, the same two managed backends compared in Demo 02.

Demo 04 is about SELECTION, deciding *what* to remember, not which backend to use.
But the survivors have to land somewhere, and the choice of vector backend is the
same trade-off Demo 02 measured. So mechanism B lets you pick:

  VECTOR_BACKEND=s3        -> Amazon S3 Vectors (dedicated vector bucket, one index per type)
  VECTOR_BACKEND=dynamodb  -> Amazon DynamoDB Vector Search (vector index inside a table)

Both keep the demo's per-memory-type partitioning (one index/table per type, the
same partitioning Amazon Bedrock AgentCore Memory gives you managed), and both use
the same Amazon Titan Text Embeddings V2 model, so the only thing that changes is
where the vectors live.

  - Amazon S3 Vectors (managed vector storage): dedicated vector bucket + index.
    Persists across restarts, reachable from any process with credentials.
  - Amazon DynamoDB Vector Search (GA 2025): vector index added to a DynamoDB table,
    so operational data and embeddings share one service. Uses the SearchVectors API
    (added in boto3 1.43.72). Single-digit millisecond latency, on-demand billing.

Self-provisioning (series rule): the bucket/index or table/index are created if
missing, no console steps needed to run the demo.

Requires boto3 >= 1.43.72 for the DynamoDB backend (SearchVectors).
"""

import json
import os
import threading
import time

import boto3

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024

VECTOR_BUCKET = os.getenv("VECTOR_BUCKET", "agent-memory-demo-vectors")
VECTOR_INDEX = os.getenv("VECTOR_INDEX", "traveler-memories")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Which backend mechanism B writes to. "s3" (default) or "dynamodb".
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "s3").lower()

# DynamoDB backend config (one table per memory type, mirrors the S3 per-index
# partitioning; the demo builds the full name from a prefix + the memory type).
DYNAMODB_TABLE_PREFIX = os.getenv("DYNAMODB_TABLE_PREFIX", "selective-memory")
DYNAMODB_VECTOR_INDEX = os.getenv("DYNAMODB_VECTOR_INDEX", "memory-vector-index")
DYNAMODB_VECTOR_ATTR = os.getenv("DYNAMODB_VECTOR_ATTR", "embedding")


# ── AWS clients (explicit session so env bearer tokens can't override) ───────
_session = None


def _aws():
    global _session
    if _session is None:
        profile = os.getenv("AWS_PROFILE")
        _session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return _session


def embed(text: str) -> list[float]:
    """Real Titan V2 embedding (1024 dims), used by both vector backends."""
    client = _aws().client("bedrock-runtime", region_name=AWS_REGION)
    resp = client.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": EMBED_DIM}),
    )
    return json.loads(resp["body"].read())["embedding"]


# ── Amazon S3 Vectors (managed vector storage) ───────────────────────────────
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


# ── Amazon DynamoDB Vector Search (vector index inside a table) ──────────────
class DynamoDBVectorStore:
    """Vector memory in a DynamoDB table with a native vector index.

    Vectors are stored as regular DynamoDB items (primary key + text + embedding
    List attribute). Queries use the SearchVectors API, approximate nearest
    neighbor at single-digit millisecond latency, serverless, on-demand billing.

    The key difference vs S3 Vectors: the vector index lives inside a DynamoDB
    table, so operational data and embeddings share one service and one billing
    model. No separate vector bucket to manage.

    Score note: SearchVectors returns a COSINE distance (1 − cosine_similarity).
    For L2-normalized embeddings (Titan V2 is normalized) this is in [0, 1]; this
    class returns 1.0 − score so the output is cosine_similarity, matching the
    S3 Vectors store above.

    Requires boto3 >= 1.43.72 (SearchVectors was added in that release).
    """

    def __init__(self, table: str, index: str = DYNAMODB_VECTOR_INDEX,
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

        # Table exists, add the vector index if it's missing.
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
        """Return (text, cosine_similarity) for the top-k most similar items."""
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
        resp = self.client.scan(TableName=self.table, ProjectionExpression="memory_key")
        items = resp.get("Items", [])
        for item in items:
            self.client.delete_item(
                TableName=self.table,
                Key={"memory_key": item["memory_key"]},
            )
        return len(items)


# ── Backend factory (mechanism B picks S3 Vectors or DynamoDB by env var) ────
def make_store(partition: str):
    """Return a vector store for one memory-type partition, on the configured backend.

    `partition` is the per-type name (e.g. "selective-facts"). For S3 Vectors it
    becomes the index name inside the shared bucket; for DynamoDB it becomes the
    table name (one table per memory type, mirroring the S3 per-index partitioning).
    """
    if VECTOR_BACKEND == "dynamodb":
        # DynamoDB table names allow [a-zA-Z0-9_.-]; the partition names already qualify.
        return DynamoDBVectorStore(table=f"{DYNAMODB_TABLE_PREFIX}-{partition}")
    if VECTOR_BACKEND == "s3":
        return S3VectorStore(index=partition)
    raise ValueError(f"Unknown VECTOR_BACKEND {VECTOR_BACKEND!r} (use 's3' or 'dynamodb')")


def timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_ms), the demo measures, never guesses."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000
