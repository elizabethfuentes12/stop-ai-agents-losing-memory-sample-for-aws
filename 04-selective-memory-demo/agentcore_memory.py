"""Mechanism C, Amazon Bedrock AgentCore Memory: managed selective memory.

The fully managed version of what this demo builds by hand: you send RAW
conversation turns (create_event) and AgentCore's built-in strategies decide
what is worth keeping, extract it, embed it, and index it per memory type:

  | Strategy (this demo's name) | Type            | What it extracts            |
  |------------------------------|-----------------|------------------------------|
  | facts                        | SEMANTIC        | durable facts                |
  | preferences                  | USER_PREFERENCE | likes/dislikes               |
  | tripSummary                  | SUMMARIZATION   | rolling session summary      |
  | episodes                     | EPISODIC        | event episodes (reflection)  |

Extraction is asynchronous, records become retrievable seconds after the
event is written (measured ~1 min in this demo; the tests poll and report the
real number, because AWS publishes none).

Self-provisioning (series rule): ensure_memory() creates the memory with all
four strategies if it doesn't exist. Real API facts verified live 2026-07-16:
  - episodicMemoryStrategy REQUIRES reflectionConfiguration.namespaces -
    a bare {'name': ...} fails with a blank ValidationException.
  - DeleteMemory fails while status is CREATING, wait for ACTIVE first.
  - create_event payload = [{'conversational': {'content': {'text': ...},
    'role': 'USER'|'ASSISTANT'}}]; retrieval namespace =
    /strategies/{strategyId}/actors/{actorId}/ (session-scoped for
    summary/episodic).
"""

import os
import threading
import time
from datetime import datetime, timezone

import boto3

MEMORY_NAME = os.getenv("AGENTCORE_MEMORY_NAME", "SelectiveMemoryDemoV2")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

_session = None


def _aws():
    global _session
    if _session is None:
        profile = os.getenv("AWS_PROFILE")
        _session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return _session


def control():
    return _aws().client("bedrock-agentcore-control", region_name=AWS_REGION)


def data():
    return _aws().client("bedrock-agentcore", region_name=AWS_REGION)


def ensure_memory() -> dict:
    """Create the demo memory with the 4 built-in strategies if missing; return
    {'memory_id': ..., 'strategy_ids': {name: strategyId}} once ACTIVE."""
    ctrl = control()
    memory = None
    for m in ctrl.list_memories(maxResults=50).get("memories", []):
        if m["id"].startswith(MEMORY_NAME + "-"):
            memory = m
            break

    if memory is None:
        resp = ctrl.create_memory(
            name=MEMORY_NAME,
            description="Selective-memory demo, the 4 built-in strategies",
            eventExpiryDuration=7,
            memoryStrategies=[
                # Each strategy declares the SAME namespace you reference later in
                # RetrievalConfig (official Strands integration pattern). Without
                # explicit namespaces, retrieval can't scope by strategy.
                {"semanticMemoryStrategy": {
                    "name": "facts",
                    "namespaces": ["/facts/{actorId}/"]}},
                {"userPreferenceMemoryStrategy": {
                    "name": "preferences",
                    "namespaces": ["/preferences/{actorId}/"]}},
                {"summaryMemoryStrategy": {
                    "name": "tripSummary",
                    "namespaces": ["/summaries/{actorId}/{sessionId}/"]}},
                {"episodicMemoryStrategy": {
                    "name": "episodes",
                    "namespaces": ["/episodes/{actorId}/{sessionId}/"],
                    # Reflection namespace must be a hierarchical prefix of the
                    # episodic namespace (verified live: CreateMemory rejects
                    # otherwise). Keep it under the same /episodes/ root.
                    "reflectionConfiguration": {
                        "namespaces": ["/episodes/{actorId}/{sessionId}/"]},
                }},
            ],
        )
        memory = resp["memory"]
        print(f"  created AgentCore memory {memory['id']}")

    # Wait until ACTIVE (a memory in CREATING accepts no events and can't be deleted).
    for _ in range(60):
        detail = ctrl.get_memory(memoryId=memory["id"])["memory"]
        if detail["status"] == "ACTIVE":
            break
        threading.Event().wait(10)
    else:
        raise TimeoutError(f"memory {memory['id']} not ACTIVE after 10 min")

    strategy_ids = {s["name"]: s["strategyId"] for s in detail.get("strategies", [])}
    return {"memory_id": detail["id"], "strategy_ids": strategy_ids}


def send_turn(memory_id: str, actor_id: str, session_id: str, role: str, text: str) -> float:
    """Write one raw conversation turn (no selection on our side). Returns write ms."""
    start = time.perf_counter()
    data().create_event(
        memoryId=memory_id, actorId=actor_id, sessionId=session_id,
        eventTimestamp=datetime.now(timezone.utc),
        payload=[{"conversational": {"content": {"text": text}, "role": role}}],
    )
    return (time.perf_counter() - start) * 1000


def retrieve(memory_id: str, strategy_id: str, actor_id: str, query: str,
             session_id: str | None = None, top_k: int = 5) -> list[str]:
    """Retrieve extracted records for one strategy, best-first."""
    namespace = f"/strategies/{strategy_id}/actors/{actor_id}"
    if session_id:  # summary + episodic records live under the session namespace
        namespace += f"/sessions/{session_id}"
    resp = data().retrieve_memory_records(
        memoryId=memory_id, namespace=namespace + "/",
        searchCriteria={"searchQuery": query}, maxResults=top_k,
    )
    return [r.get("content", {}).get("text", "") for r in resp.get("memoryRecordSummaries", [])]


# Region alias for the official Strands session-manager integration.
REGION = AWS_REGION


def retrieve_by_namespace(memory_id: str, namespace: str, query: str,
                          top_k: int = 10, min_score: float = 0.0) -> list[str]:
    """Retrieve extracted records under a canonical strategy namespace
    (e.g. /facts/{actorId}/), keeping only records at or above min_score.

    This mirrors what the official AgentCoreMemorySessionManager does at recall
    time via RetrievalConfig.relevance_score: the API returns records ordered by
    relevance, and we drop the weakly-related ones instead of dumping everything.
    """
    resp = data().retrieve_memory_records(
        memoryId=memory_id, namespace=namespace,
        searchCriteria={"searchQuery": query}, maxResults=top_k,
    )
    out = []
    for r in resp.get("memoryRecordSummaries", []):
        score = r.get("score")
        if score is not None and score < min_score:
            continue
        out.append(r.get("content", {}).get("text", ""))
    return out


def wait_for_extraction(memory_id: str, strategy_id: str, actor_id: str, query: str,
                        expect_min: int = 1, timeout_s: int = 300) -> float | None:
    """Poll until at least expect_min records are retrievable under the facts
    namespace; return the lag in seconds (the number AWS doesn't publish), or None."""
    start = time.time()
    namespace = f"/facts/{actor_id}/"
    while time.time() - start < timeout_s:
        if len(retrieve_by_namespace(memory_id, namespace, query, top_k=5)) >= expect_min:
            return time.time() - start
        threading.Event().wait(10)
    return None
