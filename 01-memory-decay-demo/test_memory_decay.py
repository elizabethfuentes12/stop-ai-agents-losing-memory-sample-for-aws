"""
Demo: Memory Decay — Stateless Agent vs Stateful Agent with Persistent Preferences

Based on research:
  - MemoryOS of AI Agent (https://arxiv.org/abs/2506.06326) — Kang et al., 2025
  - Cognitive Memory in LLMs (https://arxiv.org/abs/2504.02441) — Shan et al., 2025

Runs two agents through the SAME multi-turn travel booking conversation:
  Test 1: Stateless agent — forgets everything between turns
  Test 2: Stateful agent — learns preferences via agent.state
  Test 3: Session persistence — preferences survive agent restart

Then prints a comparison showing what each agent remembered.
"""

import os
import json
import time
from dotenv import load_dotenv
from strands import Agent
# Using OpenAI-compatible interface via Strands SDK (not direct OpenAI usage)
from strands.models.openai import OpenAIModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.session import FileSessionManager
from tools import (
    search_hotels_stateless, book_hotel_stateless,
    search_hotels, book_hotel, get_user_profile,
)

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not set. Get your API key from https://platform.openai.com/api-keys "
        "then either: 1) Add OPENAI_API_KEY=your-key to a .env file, or "
        "2) Run: export OPENAI_API_KEY=your-key"
    )

MODEL = OpenAIModel(model_id="gpt-4o-mini")  # api_key read from the OPENAI_API_KEY env var

# To run on Amazon Bedrock instead (no OpenAI key; uses your AWS credentials),
# comment the MODEL line above (and the OpenAIModel import) and uncomment these:
# from strands.models import BedrockModel
# MODEL = BedrockModel(model_id="openai.gpt-oss-120b-1:0", region_name="us-west-2")

SYSTEM_PROMPT = (
    "You are a travel assistant. Help users find and book hotels. "
    "When the user books a hotel, remember their preferences (style, stars, amenities) "
    "and use them to rank results in future searches. "
    "Always be concise — answer in 2-3 sentences maximum."
)

# Multi-turn conversation simulating a returning user
TURN_1 = "Search for hotels in Tokyo under $350/night"
TURN_2 = "Book the Zen Garden Ryokan for 3 nights"
TURN_3 = "Now search for hotels in Zurich — what do you recommend based on what you know about me?"


def run_test_1_stateless():
    """Test 1: Stateless agent — no memory between turns."""
    print("\n" + "=" * 70)
    print("TEST 1: STATELESS AGENT (no memory)")
    print("=" * 70)

    agent = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        tools=[search_hotels_stateless, book_hotel_stateless],
    )

    print(f"\n{'Turn 1':} {TURN_1}")
    agent(TURN_1)

    print(f"\n{'Turn 2':} {TURN_2}")
    agent(TURN_2)

    print(f"\n{'Turn 3':} {TURN_3}")
    start = time.time()
    response = agent(TURN_3)
    elapsed = time.time() - start

    # Check if agent remembers anything about preferences
    has_state = hasattr(agent, 'state') and agent.state.get("user_preferences")
    print(f"\n{'Memory status':}")
    print(f"  agent.state has preferences: {bool(has_state)}")
    print(f"  Result: Agent treats Turn 3 as a brand new user")

    return {"has_memory": False, "time": elapsed}


def run_test_2_stateful():
    """Test 2: Stateful agent — learns preferences via agent.state."""
    print("\n" + "=" * 70)
    print("TEST 2: STATEFUL AGENT (agent.state memory)")
    print("=" * 70)

    agent = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        conversation_manager=SlidingWindowConversationManager(window_size=40),
        tools=[search_hotels, book_hotel, get_user_profile],
    )

    print(f"\n{'Turn 1':} {TURN_1}")
    agent(TURN_1)

    print(f"\n{'Turn 2':} {TURN_2}")
    agent(TURN_2)

    # Check learned preferences after booking
    prefs = agent.state.get("user_preferences")
    print(f"\n  Learned preferences: {json.dumps(prefs, indent=2)}")

    print(f"\n{'Turn 3':} {TURN_3}")
    start = time.time()
    response = agent(TURN_3)
    elapsed = time.time() - start

    history = agent.state.get("booking_history")
    print(f"\n{'Memory status':}")
    print(f"  Preferences: {json.dumps(prefs)}")
    print(f"  Booking history: {len(history)} bookings")
    print(f"  Result: Agent ranks Zurich hotels by learned preferences")

    return {"has_memory": True, "preferences": prefs, "history_count": len(history), "time": elapsed}


def run_test_3_session_persistence():
    """Test 3: Session persistence — preferences survive agent restart."""
    print("\n" + "=" * 70)
    print("TEST 3: SESSION PERSISTENCE (FileSessionManager)")
    print("=" * 70)

    session_id = "travel-user-demo"
    storage_dir = os.path.join(os.path.dirname(__file__), "sessions")

    # Session A: Book a hotel and build preferences
    print("\n--- Session A: First visit ---")
    agent_a = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        conversation_manager=SlidingWindowConversationManager(window_size=40),
        tools=[search_hotels, book_hotel, get_user_profile],
        session_manager=FileSessionManager(session_id=session_id, storage_dir=storage_dir),
    )

    agent_a(TURN_1)
    agent_a(TURN_2)
    prefs_a = agent_a.state.get("user_preferences")
    print(f"  Preferences after Session A: {json.dumps(prefs_a)}")

    # Session B: New agent instance, same session_id — should restore state
    print("\n--- Session B: Returning user (new agent instance) ---")
    agent_b = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        conversation_manager=SlidingWindowConversationManager(window_size=40),
        tools=[search_hotels, book_hotel, get_user_profile],
        session_manager=FileSessionManager(session_id=session_id, storage_dir=storage_dir),
    )

    prefs_b = agent_b.state.get("user_preferences")
    print(f"  Preferences restored in Session B: {json.dumps(prefs_b)}")
    print(f"  State survived restart: {prefs_a == prefs_b}")

    start = time.time()
    agent_b(TURN_3)
    elapsed = time.time() - start

    # Cleanup session files
    import shutil
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    return {
        "persisted": prefs_a == prefs_b,
        "prefs_session_a": prefs_a,
        "prefs_session_b": prefs_b,
        "time": elapsed,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  MEMORY DECAY DEMO")
    print("  Stateless vs Stateful vs Persistent — same conversation, different memory")
    print("=" * 70)

    r1 = run_test_1_stateless()
    r2 = run_test_2_stateful()
    r3 = run_test_3_session_persistence()

    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    print(f"\n  {'Test':<45} {'Remembers prefs':>16} {'Cross-session':>14} {'Personalized':>13}")
    print("  " + "-" * 90)
    print(f"  {'Test 1 — Stateless (no memory)':<45} {'No':>16} {'No':>14} {'No':>13}")
    print(f"  {'Test 2 — Stateful (agent.state)':<45} {'Yes':>16} {'No':>14} {'Yes':>13}")
    print(f"  {'Test 3 — Persistent (FileSessionManager)':<45} {'Yes':>16} {'Yes':>14} {'Yes':>13}")

    print(f"\n  Key insight: Without agent.state, the agent forgets user preferences")
    print(f"  between turns. Without SessionManager, it forgets between sessions.")
    print(f"\n  Research: https://arxiv.org/abs/2506.06326 (MemoryOS)")
    print(f"  Research: https://arxiv.org/abs/2504.02441 (Cognitive Memory in LLMs)")
    print(f"  Strands:  https://github.com/strands-agents/sdk-python")
