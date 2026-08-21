"""Amazon S3 Vectors store and Titan embedder used by this demo.

Demo 04 is about selection — deciding *what* to remember, not which backend to use.
The vector backend is Amazon S3 Vectors: the extractor writes memories there after
selecting what's worth keeping, and the agent reads them back by semantic similarity.

Self-provisioning (series rule): the bucket and index are created if missing — no
console steps needed to run the demo.
"""

import json
import os
import time

import boto3

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
    """Real Titan V2 embedding (1024 dims)."""
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


def timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_ms) — the demo measures, never guesses."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000
