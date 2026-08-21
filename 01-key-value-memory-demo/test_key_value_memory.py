"""
Demo: Stop Your AI Agent from Forgetting User Preferences — Key-Value Memory (Agent State)

(The research literature calls this problem "memory decay". Strands calls this
store "agent state": key-value storage that lives outside the conversation.)

A brand-new user arrives with EMPTY memory. The demo isolates ONE variable across
four tests: WHERE MEMORY LIVES. Same model, same tools' business logic, same
3-turn conversation — the only change is the memory wiring, climbing one rung of
durability per test. Everything is Strands built-ins.

  Test 1: No memory tools    — the "memory" is just the transcript: nothing
                               structured is learned, and a restart wipes it all
  Test 2: agent.state        — booking teaches it preferences (process memory)
  Test 3: FileSessionManager — the profile survives an agent restart (local disk)
  Test 4: S3SessionManager   — same, stored in Amazon S3 (production: no filesystem
                               to provision or mount, state shared across any compute)

(The flight data behind the tools is live — Duffel sandbox + Open-Meteo — so
nothing is hardcoded, but the APIs are scenery: the experiment is memory.)

Based on research:
  - MemoryOS of AI Agent (https://arxiv.org/abs/2506.06326) — Kang et al., 2025
  - Cognitive Memory in LLMs (https://arxiv.org/abs/2504.02441) — Shan et al., 2025

Requires: OPENAI_API_KEY (model) + DUFFEL_API_KEY (free sandbox token from
https://app.duffel.com). Climate data (Open-Meteo) needs no key. Test 4 also
needs AWS credentials + SESSIONS_BUCKET in .env (skipped gracefully if absent).
"""

import os
import json
import time
from dotenv import load_dotenv
from strands import Agent
# Using OpenAI-compatible interface via Strands SDK (not direct OpenAI usage)
from strands.models.openai import OpenAIModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.session import FileSessionManager, S3SessionManager
from tools import (
    search_flights_stateless, book_flight_stateless,
    search_flights, book_flight, get_user_profile, best_time_to_visit,
)

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not set. Get your API key from https://platform.openai.com/api-keys "
        "then either: 1) Add OPENAI_API_KEY=your-key to a .env file, or "
        "2) Run: export OPENAI_API_KEY=your-key"
    )
if not os.getenv("DUFFEL_API_KEY"):
    raise ValueError(
        "DUFFEL_API_KEY not set. Create a free sandbox token at https://app.duffel.com "
        "(More -> Developers -> Access tokens) and add DUFFEL_API_KEY=duffel_test_... "
        "to your .env file."
    )

MODEL = OpenAIModel(model_id="gpt-4o-mini")  # api_key read from the OPENAI_API_KEY env var

# To run on Amazon Bedrock instead (no OpenAI key; uses your AWS credentials),
# comment the MODEL line above (and the OpenAIModel import) and uncomment these:
# from strands.models import BedrockModel
# MODEL = BedrockModel(model_id="openai.gpt-oss-120b-1:0", region_name="us-west-2")

# Role and approach only — tool purposes live in the tools' own docstrings.
SYSTEM_PROMPT = (
    "You are a flight booking assistant for a returning traveler. "
    "Personalize recommendations using what you know about the user. "
    "Be concise — answer in 2-3 sentences maximum."
)

# The same 3-turn conversation is used in every test, so the ONLY variable is memory.
# A brand-new user: business-cabin taste revealed by their booking ACTION, not a form.
TURN_1 = "Find me flights from JFK to Paris CDG on 2026-09-15, business class."
TURN_2 = "Book the cheapest business option."
TURN_3 = ("Now I need Paris CDG to Tokyo Haneda on 2026-09-22 — "
          "what do you recommend based on what you know about me?")


def run_test_1_stateless():
    """Test 1: No memory tools — the only "memory" is the transcript itself.

    Within the session the agent DOES personalize turn 3: Strands keeps the full
    conversation history (agent.messages) between calls, so "business class" is
    still in the transcript when the model answers. But nothing structured is
    learned (user_preferences stays None), and a restart — a brand-new agent
    instance, which is every new process/request in production — asks turn 3
    with an empty transcript and gets a generic answer.
    """
    agent = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_flights_stateless, book_flight_stateless],
        callback_handler=None,
    )
    for turn in (TURN_1, TURN_2, TURN_3):
        resp = agent(turn)
        print(f"  User: {turn}\n  Agent: {str(resp).strip()[:200]}\n")

    prefs = agent.state.get("user_preferences")
    print(f"  user_preferences after 3 turns: {prefs}")
    print(f"  messages in transcript: {len(agent.messages)} — the booking lives ONLY here")

    # The restart: a new agent instance = new process/request in production.
    # No session manager, so the transcript is gone — turn 3 has nothing to use.
    agent_restarted = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_flights_stateless, book_flight_stateless],
        callback_handler=None,
    )
    resp = agent_restarted(TURN_3)
    print(f"\n  [after restart] User: {TURN_3}")
    print(f"  [after restart] Agent: {str(resp).strip()[:200]}")
    return {"prefs": prefs, "survived": False}


def run_test_2_stateful():
    """Test 2: agent.state — the booking ACTION teaches the agent its user.

    Same conversation. book_flight writes cabin/stops/price/carriers into
    agent.state; turn 3's search now ranks real offers by that profile.
    """
    agent = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        conversation_manager=SlidingWindowConversationManager(window_size=40),
        tools=[search_flights, book_flight, get_user_profile, best_time_to_visit],
        callback_handler=None,
    )
    for turn in (TURN_1, TURN_2, TURN_3):
        resp = agent(turn)
        print(f"  User: {turn}\n  Agent: {str(resp).strip()[:200]}\n")

    prefs = agent.state.get("user_preferences") or {}
    history = agent.state.get("booking_history") or []
    print(f"  user_preferences: {json.dumps(prefs)}")
    print(f"  bookings recorded: {len(history)}")
    return {"prefs": prefs, "bookings": len(history)}


def run_test_3_persistence():
    """Test 3: FileSessionManager — the profile survives an agent restart.

    Session A books (building the profile); Session B is a brand-new agent
    instance with the same session_id and starts already knowing the user.
    """
    session_id = "traveler-demo"
    storage_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions")

    def make_agent():
        return Agent(
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
            conversation_manager=SlidingWindowConversationManager(window_size=40),
            tools=[search_flights, book_flight, get_user_profile, best_time_to_visit],
            session_manager=FileSessionManager(session_id=session_id, storage_dir=storage_dir),
            callback_handler=None,
        )

    agent_a = make_agent()
    agent_a(TURN_1)
    agent_a(TURN_2)
    prefs_a = agent_a.state.get("user_preferences")
    print(f"  Session A learned: {json.dumps(prefs_a)}")

    agent_b = make_agent()  # new instance, same session_id = the "restart"
    prefs_b = agent_b.state.get("user_preferences")
    survived = prefs_a == prefs_b and prefs_b is not None
    print(f"  Session B restored: {json.dumps(prefs_b)}")
    print(f"  state survived restart: {survived}")

    resp = agent_b(TURN_3)
    print(f"  User: {TURN_3}\n  Agent: {str(resp).strip()[:250]}")

    import shutil
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)
    return {"survived": survived, "learned": prefs_a is not None}


def ensure_bucket(s3, bucket: str) -> None:
    """Create the sessions bucket if it doesn't exist (private, public access blocked).

    us-east-1 is the one region that must NOT send a LocationConstraint; every
    other region requires it. An existing bucket owned by you is fine (idempotent).
    """
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except s3.exceptions.ClientError:
        pass
    region = s3.meta.region_name or "us-east-1"
    params = {"Bucket": bucket}
    if region != "us-east-1":
        params["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**params)
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
        },
    )
    print(f"  created private bucket s3://{bucket} in {region}")


def run_test_4_s3_persistence():
    """Test 4: S3SessionManager — same restart survival, stored in Amazon S3.

    Same interface as FileSessionManager, but the session persists as plain JSON
    objects in an S3 bucket (regular S3 — no vectors, no embeddings). Why S3 over
    a filesystem in production: nothing to provision or mount (a durable filesystem
    on Lambda/Fargate means wiring up EFS: VPC, mount targets, security groups),
    and any compute instance can read the session without sharing a network drive.
    Skipped if AWS credentials or SESSIONS_BUCKET aren't configured.
    """
    bucket = os.getenv("SESSIONS_BUCKET")
    if not bucket:
        print("  skipped: set SESSIONS_BUCKET in .env to run this test (created if it doesn't exist)")
        return {"survived": None}

    import boto3
    ensure_bucket(boto3.client("s3"), bucket)
    session_id = "traveler-demo-s3"
    prefix = "kv-memory-demo"

    def make_agent():
        return Agent(
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
            conversation_manager=SlidingWindowConversationManager(window_size=40),
            tools=[search_flights, book_flight, get_user_profile, best_time_to_visit],
            session_manager=S3SessionManager(session_id=session_id, bucket=bucket, prefix=prefix),
            callback_handler=None,
        )

    agent_a = make_agent()
    agent_a(TURN_1)
    agent_a(TURN_2)
    prefs_a = agent_a.state.get("user_preferences")
    print(f"  Session A learned: {json.dumps(prefs_a)}")

    agent_b = make_agent()  # new instance, same session_id — restored FROM S3
    prefs_b = agent_b.state.get("user_preferences")
    survived = prefs_a == prefs_b and prefs_b is not None
    print(f"  Session B restored from s3://{bucket}/{prefix}: {json.dumps(prefs_b)}")
    print(f"  state survived restart: {survived}")

    # Clean up the demo's session objects so reruns start empty.
    s3 = boto3.client("s3")
    listed = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [{"Key": o["Key"]} for o in listed.get("Contents", [])]
    if keys:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})
    return {"survived": survived, "learned": prefs_a is not None}


if __name__ == "__main__":
    start = time.time()
    r1 = run_test_1_stateless()
    r2 = run_test_2_stateful()
    r3 = run_test_3_persistence()
    r4 = run_test_4_s3_persistence()

    # Comparison: measured results only — the narrative lives in the README.
    print(f"\n{'Test':<42} {'Learned prefs':>14} {'Survived restart':>17}")
    print(f"{'1 — no memory tools (transcript only)':<42} {str(r1['prefs'] is not None):>14} {str(r1['survived']):>17}")
    print(f"{'2 — agent.state':<42} {str(bool(r2['prefs'])):>14} {'—':>17}")
    print(f"{'3 — agent.state + FileSessionManager':<42} {str(r3['learned']):>14} {str(r3['survived']):>17}")
    print(f"{'4 — agent.state + S3SessionManager':<42} {str(r4.get('learned', '—')):>14} {str(r4['survived']):>17}")
    print(f"\nwall time: {time.time() - start:.0f}s (live Duffel + Open-Meteo calls)")
