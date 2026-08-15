"""Mechanism C — Amazon Bedrock AgentCore Memory: managed selective memory.

The fully managed version of what this demo builds by hand: you send RAW
conversation turns (create_event) and AgentCore's built-in strategies decide
what is worth keeping, extract it, embed it, and index it per memory type:

  | Strategy (this demo's name) | Type            | What it extracts            |
  |------------------------------|-----------------|------------------------------|
  | facts                        | SEMANTIC        | durable facts                |
  | preferences                  | USER_PREFERENCE | likes/dislikes               |
  | tripSummary                  | SUMMARIZATION   | rolling session summary      |
  | episodes                     | EPISODIC        | event episodes (reflection)  |

Extraction is asynchronous — records become retrievable seconds after the
event is written (measured ~1 min in this demo; the tests poll and report the
real number, because AWS publishes none).

Self-provisioning (series rule): ensure_memory() creates the memory with all
four strategies if it doesn't exist. Real API facts verified live 2026-07-16:
  - episodicMemoryStrategy REQUIRES reflectionConfiguration.namespaces —
    a bare {'name': ...} fails with a blank ValidationException.
  - DeleteMemory fails while status is CREATING — wait for ACTIVE first.
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

MEMORY_NAME = os.getenv("AGENTCORE_MEMORY_NAME", "SelectiveMemoryDemo")
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
            description="Selective-memory demo — the 4 built-in strategies",
            eventExpiryDuration=7,
            memoryStrategies=[
                {"semanticMemoryStrategy": {"name": "facts"}},
                {"userPreferenceMemoryStrategy": {"name": "preferences"}},
                {"summaryMemoryStrategy": {"name": "tripSummary"}},
                {"episodicMemoryStrategy": {
                    "name": "episodes",
                    # Required in practice (verified live): a bare episodic
                    # strategy fails validation without reflection namespaces.
                    "reflectionConfiguration": {
                        "namespaces": ["/strategies/{memoryStrategyId}/actors/{actorId}"]},
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


def wait_for_extraction(memory_id: str, strategy_id: str, actor_id: str, query: str,
                        expect_min: int = 1, timeout_s: int = 300) -> float | None:
    """Poll until at least expect_min records are retrievable; return the lag in
    seconds (the number AWS doesn't publish), or None on timeout."""
    start = time.time()
    while time.time() - start < timeout_s:
        if len(retrieve(memory_id, strategy_id, actor_id, query)) >= expect_min:
            return time.time() - start
        threading.Event().wait(10)
    return None
