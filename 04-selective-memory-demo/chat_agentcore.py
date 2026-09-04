"""
Interactive flight assistant — MANAGED MEMORY (Mechanism C: AgentCore Memory).

Chat with an agent whose long-term memory is fully managed by Amazon Bedrock
AgentCore Memory. You send raw turns (create_event); AWS's four built-in
strategies (semantic / userPreference / summary / episodic) extract, embed, and
index them for you. Nothing to tune, no pipeline to run.

The honest trade-off you will SEE here: extraction is asynchronous. Right after
you speak, `/memory` may show nothing, because AgentCore is still extracting
(measured ~1 minute in this demo). Wait and run `/memory` again to watch the
memories appear. That availability lag is the price of "managed".

Commands:
  /memory   retrieve what AgentCore has extracted so far, per strategy
  /wait     block until facts appear, and report the real extraction lag
  /quit
Note: this creates an AgentCore memory named SelectiveMemoryDemo if missing
(can take a few minutes the first time). It does NOT delete anything on exit;
use the notebook's cleanup cell for teardown.
"""

import os
import sys
import time

os.environ["OTEL_SDK_DISABLED"] = "true"
# Bearer-token env vars would override the AWS profile; drop them before boto3 loads.
os.environ.pop("AWS_BEARER_TOKEN", None)
os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not set. Add it to .env, or switch to Bedrock "
          "(uncomment the BedrockModel line below).")
    sys.exit(1)

from strands import Agent
# Using the OpenAI-compatible interface via the Strands SDK (not direct OpenAI usage).
from strands.models.openai import OpenAIModel

import agentcore_memory as acm
from tools import search_flights, book_flight, best_time_to_visit

MODEL = OpenAIModel(model_id="gpt-4o-mini")
# To use Amazon Bedrock instead of OpenAI (no OpenAI key needed), comment the line
# above and uncomment these:
# from strands.models import BedrockModel
# MODEL = BedrockModel(model_id="openai.gpt-oss-120b-1:0", region_name="us-west-2")

RECALL_QUERY = "traveler dietary restrictions travel preferences bookings and trip"


def show_memory(memory_id, sids, actor, session) -> None:
    """Retrieve what AgentCore has extracted so far, per strategy."""
    print("\n  🧠 AgentCore memory, per strategy (extraction is async, may lag ~1 min):")
    any_found = False
    for name, sid in sids.items():
        session_scoped = name in ("tripSummary", "episodes")
        records = acm.retrieve(memory_id, sid, actor, RECALL_QUERY,
                               session_id=session if session_scoped else None, top_k=10)
        if records:
            any_found = True
            for r in records:
                print(f"     [{name}] {r}")
        else:
            print(f"     [{name}] (nothing extracted yet)")
    if not any_found:
        print("  (empty so far — AgentCore is still extracting; try /memory again in ~1 min, or /wait)")
    print()


def main() -> None:
    print("Ensuring the AgentCore memory exists (first run can take a few minutes)...")
    info = acm.ensure_memory()
    memory_id, sids = info["memory_id"], info["strategy_ids"]
    actor = f"sam-{int(time.time())}"   # fresh actor per session -> clean namespaces
    session = "chat-session"

    agent = Agent(
        model=MODEL,
        system_prompt="You are a helpful flight assistant. Be concise, 2-3 sentences max.",
        callback_handler=None,
    )

    print("Flight assistant ready (Mechanism C: AgentCore managed memory).")
    print("Type a message, /memory, /wait, or /quit.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/memory":
            show_memory(memory_id, sids, actor, session)
            continue
        if user == "/wait":
            print("  waiting for AgentCore extraction (polling up to 7 min)...")
            lag = acm.wait_for_extraction(memory_id, sids["facts"], actor,
                                          "dietary restrictions travel preferences", timeout_s=420)
            print(f"  extraction lag: {lag:.0f}s (a number AWS does not publish)" if lag
                  else "  still nothing after 7 min")
            show_memory(memory_id, sids, actor, session)
            continue

        # The chat model answers now; AgentCore stores the raw turn for managed extraction.
        reply = agent(user)
        ms = acm.send_turn(memory_id, actor, session, "USER", user)
        print(f"\nassistant> {reply}")
        print(f"  (turn sent to AgentCore in {ms:.0f} ms; extraction happens async in the background)")

    print("Bye. AgentCore memory persists (delete it via the notebook cleanup cell).")


if __name__ == "__main__":
    main()
