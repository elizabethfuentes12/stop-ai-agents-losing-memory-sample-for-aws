"""
Demo: Core Memory Pattern — Agent-Managed Structured Memory

Based on research:
  - MIRIX: Multi-Agent Memory System (https://arxiv.org/abs/2507.07957) — Wang & Chen, 2025
  - MemGPT: Towards LLMs as Operating Systems (https://arxiv.org/abs/2310.08560) — Packer et al., 2023
  - Enabling Personalized Long-term Interactions (https://arxiv.org/abs/2510.07925) — Westhaeusser et al., 2025

The Core Memory pattern gives the agent explicit tools to manage its own memory:
  - core_memory_read: Read a section
  - core_memory_write: Create a section
  - core_memory_update: Modify existing data
  - core_memory_list: See what's stored

The agent DECIDES when to store/retrieve information — it manages memory like a human.

Tests:
  1: No core memory — generic responses
  2: With core memory — agent builds a user profile through conversation
  3: Multi-turn evolution — profile evolves as preferences change
  4: Cross-session with FileSessionManager — core memory persists
"""

import os
import json
import time
import shutil
from dotenv import load_dotenv
from strands import Agent
# Using OpenAI-compatible interface via Strands SDK (not direct OpenAI usage)
from strands.models.openai import OpenAIModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.session import FileSessionManager
from tools import (
    core_memory_read, core_memory_write, core_memory_update, core_memory_list,
    search_hotels_with_memory, book_hotel_with_memory,
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

SYSTEM_PROMPT_NO_MEMORY = (
    "You are a travel assistant. Help users find and book hotels. "
    "Be concise — answer in 2-3 sentences maximum."
)

SYSTEM_PROMPT_CORE_MEMORY = (
    "You are a travel assistant with core memory capabilities. "
    "You can read and write to your core memory to remember user preferences. "
    "\n\nIMPORTANT memory management rules:"
    "\n1. When a user shares personal info (name, preferences), WRITE it to core_memory 'persona' section"
    "\n2. When a user books something, UPDATE the 'preferences' section with learned preferences"
    "\n3. Before searching, READ 'preferences' to personalize results"
    "\n4. Use core_memory_list to check what you already know"
    "\n\nBe concise — answer in 2-3 sentences maximum."
)


def run_test_1_no_memory():
    """Test 1: No core memory — generic responses."""
    print("\n" + "=" * 70)
    print("TEST 1: NO CORE MEMORY (baseline)")
    print("=" * 70)

    agent = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT_NO_MEMORY,
        tools=[search_hotels_with_memory, book_hotel_with_memory],
    )

    print("\nTurn 1: My name is Alex. I love traditional Japanese culture and prefer 4-star hotels.")
    agent("My name is Alex. I love traditional Japanese culture and prefer 4-star hotels.")

    print("\nTurn 2: Search hotels in Tokyo for me")
    agent("Search hotels in Tokyo for me")

    memory = agent.state.get("core_memory")
    print(f"\nCore memory: {json.dumps(memory) if memory else 'Empty — agent did not store anything'}")
    return {"has_core_memory": bool(memory)}


def run_test_2_with_core_memory():
    """Test 2: With core memory — agent builds a user profile."""
    print("\n" + "=" * 70)
    print("TEST 2: WITH CORE MEMORY (agent manages its own memory)")
    print("=" * 70)

    agent = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT_CORE_MEMORY,
        conversation_manager=SlidingWindowConversationManager(window_size=40),
        tools=[
            core_memory_read, core_memory_write, core_memory_update, core_memory_list,
            search_hotels_with_memory, book_hotel_with_memory,
        ],
    )

    print("\nTurn 1: My name is Alex. I love traditional Japanese culture and prefer 4-star hotels.")
    agent("My name is Alex. I love traditional Japanese culture and prefer 4-star hotels.")

    print("\nTurn 2: Search hotels in Tokyo for me")
    agent("Search hotels in Tokyo for me")

    print("\nTurn 3: Book the Zen Garden Ryokan for 3 nights")
    agent("Book the Zen Garden Ryokan for 3 nights")

    print("\nTurn 4: Now search Zurich hotels — what fits my profile?")
    agent("Now search Zurich hotels — what fits my profile?")

    memory = agent.state.get("core_memory")
    print(f"\nCore memory sections:")
    if memory:
        for section, data in memory.items():
            print(f"  [{section}] v{data.get('version', '?')}: {json.dumps(data.get('data', ''))[:100]}...")
    else:
        print("  Empty")

    return {
        "has_core_memory": bool(memory),
        "sections": list(memory.keys()) if memory else [],
        "memory": memory,
    }


def run_test_3_evolution():
    """Test 3: Core memory evolves as preferences change."""
    print("\n" + "=" * 70)
    print("TEST 3: CORE MEMORY EVOLUTION (preferences change over time)")
    print("=" * 70)

    agent = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT_CORE_MEMORY,
        conversation_manager=SlidingWindowConversationManager(window_size=40),
        tools=[
            core_memory_read, core_memory_write, core_memory_update, core_memory_list,
            search_hotels_with_memory, book_hotel_with_memory,
        ],
    )

    print("\nTurn 1: I'm Alex, I prefer luxury 5-star hotels with spa access")
    agent("I'm Alex, I prefer luxury 5-star hotels with spa access")

    v1_memory = json.dumps(agent.state.get("core_memory") or {})

    print("\nTurn 2: Actually, I've changed my mind. I now prefer boutique hotels under $250")
    agent("Actually, I've changed my mind. I now prefer boutique hotels under $250 with character and garden access")

    v2_memory = json.dumps(agent.state.get("core_memory") or {})
    evolved = v1_memory != v2_memory

    print("\nTurn 3: What are my current preferences?")
    agent("What are my current preferences? Read my profile from core memory.")

    memory = agent.state.get("core_memory")
    print(f"\nMemory evolved: {evolved}")
    if memory:
        for section, data in memory.items():
            print(f"  [{section}] v{data.get('version', '?')}: {json.dumps(data.get('data', ''))[:120]}...")

    return {"evolved": evolved, "memory": memory}


def run_test_4_cross_session():
    """Test 4: Core memory persists across sessions via FileSessionManager."""
    print("\n" + "=" * 70)
    print("TEST 4: CROSS-SESSION PERSISTENCE (FileSessionManager)")
    print("=" * 70)

    session_id = "core-memory-user-demo"
    storage_dir = os.path.join(os.path.dirname(__file__), "sessions")

    # Session A
    print("\n--- Session A: Build core memory ---")
    agent_a = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT_CORE_MEMORY,
        conversation_manager=SlidingWindowConversationManager(window_size=40),
        tools=[
            core_memory_read, core_memory_write, core_memory_update, core_memory_list,
            search_hotels_with_memory, book_hotel_with_memory,
        ],
        session_manager=FileSessionManager(session_id=session_id, storage_dir=storage_dir),
    )

    agent_a("I'm Alex, I love traditional Japanese culture, prefer 4-star hotels with onsen and garden")
    agent_a("Book the Zen Garden Ryokan for 3 nights")

    memory_a = agent_a.state.get("core_memory")
    print(f"  Core memory sections: {list(memory_a.keys()) if memory_a else 'none'}")

    # Session B — new agent, same session_id
    print("\n--- Session B: Returning user (new agent) ---")
    agent_b = Agent(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT_CORE_MEMORY,
        conversation_manager=SlidingWindowConversationManager(window_size=40),
        tools=[
            core_memory_read, core_memory_write, core_memory_update, core_memory_list,
            search_hotels_with_memory, book_hotel_with_memory,
        ],
        session_manager=FileSessionManager(session_id=session_id, storage_dir=storage_dir),
    )

    memory_b = agent_b.state.get("core_memory")
    print(f"  Core memory restored: {bool(memory_b)}")
    print(f"  Sections: {list(memory_b.keys()) if memory_b else 'none'}")

    agent_b("What do you remember about me? Check your core memory.")

    # Cleanup
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    return {"persisted": bool(memory_b), "sections_restored": list(memory_b.keys()) if memory_b else []}


if __name__ == "__main__":
    print("=" * 70)
    print("  CORE MEMORY DEMO")
    print("  Agent-managed structured memory — read, write, update, persist")
    print("=" * 70)

    r1 = run_test_1_no_memory()
    r2 = run_test_2_with_core_memory()
    r3 = run_test_3_evolution()
    r4 = run_test_4_cross_session()

    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    print(f"\n  {'Test':<50} {'Core memory':>12} {'Sections':>10} {'Evolves':>9} {'Persists':>9}")
    print("  " + "-" * 92)
    print(f"  {'Test 1 — No core memory':<50} {'No':>12} {'0':>10} {'—':>9} {'—':>9}")
    print(f"  {'Test 2 — Core memory tools':<50} {'Yes':>12} {len(r2.get('sections', [])):>10} {'—':>9} {'—':>9}")
    print(f"  {'Test 3 — Memory evolution':<50} {'Yes':>12} {'—':>10} {str(r3.get('evolved', '?')):>9} {'—':>9}")
    print(f"  {'Test 4 — Cross-session persistence':<50} {'Yes':>12} {len(r4.get('sections_restored', [])):>10} {'—':>9} {str(r4.get('persisted', '?')):>9}")

    print(f"\n  Key insight: Core memory gives the agent AGENCY over its own memory.")
    print(f"  The agent decides what to remember, when to update, and what to forget.")
    print(f"\n  Research: https://arxiv.org/abs/2507.07957 (MIRIX)")
    print(f"  Research: https://arxiv.org/abs/2310.08560 (MemGPT)")
    print(f"  Strands:  https://github.com/strands-agents/sdk-python")
