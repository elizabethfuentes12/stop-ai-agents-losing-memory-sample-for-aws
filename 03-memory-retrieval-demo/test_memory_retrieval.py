"""
Demo: Memory Retrieval — Finding the Right Memories When Memory Grows Large

Based on research in agent memory systems:
  - Zep: Temporal Knowledge Graph (a time-aware memory structure) for Agent Memory (https://arxiv.org/abs/2501.13956) — 2025
  - PersonaAgent with GraphRAG (Graph-based Retrieval-Augmented Generation — combines knowledge graphs with retrieval) (https://arxiv.org/abs/2511.17467) — 2025
  - HippoRAG 2: From RAG (Retrieval-Augmented Generation) to Memory (memory-focused retrieval) (https://arxiv.org/abs/2502.14802) — 2025

When an agent accumulates many memory sections (persona, preferences, history,
food, work schedule, loyalty programs...), loading all memory into context is
inefficient. This demo compares three retrieval strategies:

  Test 1: Dump all memory — entire memory enters context (expensive)
  Test 2: Keyword search — fast but misses synonyms (brittle)
  Test 3: Semantic search — embedding-based, returns top-k relevant (accurate)
  Test 4: Agent with semantic search — full conversation with smart retrieval
"""

import os
import json
import time
from dotenv import load_dotenv
from strands import Agent
# Using OpenAI-compatible interface via Strands SDK (not direct OpenAI usage)
from strands.models.openai import OpenAIModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from tools import seed_memory, memory_dump_all, memory_search_keyword, memory_search_semantic

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY not set. Get your API key from https://platform.openai.com/api-keys "
        "then either: 1) Add OPENAI_API_KEY=your-key to a .env file, or "
        "2) Run: export OPENAI_API_KEY=your-key"
    )

MODEL = OpenAIModel(model_id="gpt-4o-mini")

# A deliberately synonym-heavy query: it never says "food", "restaurant", or the
# section name "food_preferences". Keyword search struggles (no shared words);
# real semantic search still finds the food section by meaning.
QUERY = "What dietary restrictions and culinary tastes should I keep in mind while dining out abroad?"


def count_context_tokens(agent) -> int:
    """Estimate tokens from all messages in conversation history."""
    total = 0
    for msg in agent.messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        total += len(block["text"]) // 4
                    elif "toolResult" in block:
                        for item in block["toolResult"].get("content", []):
                            if "text" in item:
                                total += len(item["text"]) // 4
                    elif "toolUse" in block:
                        total += len(json.dumps(block["toolUse"].get("input", {}))) // 4
    return total


def run_test_1_dump_all():
    """Test 1: Dump all memory into context."""
    print("\n" + "=" * 70)
    print("TEST 1: DUMP ALL MEMORY (baseline)")
    print("=" * 70)

    agent = Agent(
        model=MODEL,
        system_prompt=(
            "You are a personal assistant. Use memory_dump_all to retrieve the user's "
            "complete memory, then answer their question. Be concise."
        ),
        tools=[memory_dump_all],
    )
    sections = seed_memory(agent)
    print(f"Seeded {sections} memory sections")

    print(f"\nQuery: {QUERY}\n")
    start = time.time()
    agent(QUERY)
    elapsed = time.time() - start
    tokens = count_context_tokens(agent)

    # Size the model-visible content only (exclude the internal embedding vectors,
    # which are an index, never sent to the model).
    memory = agent.state.get("core_memory") or {}
    full_size = len(json.dumps({s: e.get("data") for s, e in memory.items()}))
    print(f"\n  Time: {elapsed:.1f}s | Tokens: {tokens:,} | Memory loaded: {full_size:,} bytes (ALL)")

    return {"tokens": tokens, "time": elapsed, "bytes_loaded": full_size, "strategy": "dump_all"}


def run_test_2_keyword():
    """Test 2: Keyword search over memory."""
    print("\n" + "=" * 70)
    print("TEST 2: KEYWORD SEARCH")
    print("=" * 70)

    agent = Agent(
        model=MODEL,
        system_prompt=(
            "You are a personal assistant. Use memory_search_keyword to find relevant "
            "memories by keyword, then answer the question. Be concise."
        ),
        tools=[memory_search_keyword],
    )
    sections = seed_memory(agent)
    print(f"Seeded {sections} memory sections")

    print(f"\nQuery: {QUERY}\n")
    start = time.time()
    agent(QUERY)
    elapsed = time.time() - start
    tokens = count_context_tokens(agent)

    print(f"\n  Time: {elapsed:.1f}s | Tokens: {tokens:,} | Strategy: keyword")

    return {"tokens": tokens, "time": elapsed, "strategy": "keyword"}


def run_test_3_semantic():
    """Test 3: Semantic search over memory."""
    print("\n" + "=" * 70)
    print("TEST 3: SEMANTIC SEARCH")
    print("=" * 70)

    agent = Agent(
        model=MODEL,
        system_prompt=(
            "You are a personal assistant. Use memory_search_semantic to find the most "
            "relevant memories for the user's question, then answer. Be concise."
        ),
        tools=[memory_search_semantic],
    )
    sections = seed_memory(agent)
    print(f"Seeded {sections} memory sections")

    print(f"\nQuery: {QUERY}\n")
    start = time.time()
    agent(QUERY)
    elapsed = time.time() - start
    tokens = count_context_tokens(agent)

    print(f"\n  Time: {elapsed:.1f}s | Tokens: {tokens:,} | Strategy: semantic (top-3)")

    return {"tokens": tokens, "time": elapsed, "strategy": "semantic"}


def run_test_4_agent_conversation():
    """Test 4: Multi-turn conversation with semantic retrieval."""
    print("\n" + "=" * 70)
    print("TEST 4: FULL AGENT WITH SEMANTIC RETRIEVAL")
    print("=" * 70)

    agent = Agent(
        model=MODEL,
        system_prompt=(
            "You are a personal travel assistant with access to the user's memory. "
            "Use memory_search_semantic to find relevant context BEFORE answering. "
            "Only retrieve what's needed — don't dump all memory. Be concise."
        ),
        conversation_manager=SlidingWindowConversationManager(window_size=40),
        tools=[memory_search_semantic, memory_search_keyword],
    )
    sections = seed_memory(agent)
    print(f"Seeded {sections} memory sections")

    queries = [
        "What restaurants should I look for on my Zurich trip?",
        "When should I avoid scheduling this trip based on my work calendar?",
        "Do I have enough miles to fly United to Zurich?",
    ]

    total_tokens = 0
    for i, q in enumerate(queries, 1):
        print(f"\nTurn {i}: {q}")
        agent(q)
        tokens = count_context_tokens(agent)
        print(f"  Tokens after turn {i}: {tokens:,}")
        total_tokens = tokens

    return {"tokens": total_tokens, "turns": len(queries), "strategy": "semantic_multi_turn"}


if __name__ == "__main__":
    print("=" * 70)
    print("  MEMORY RETRIEVAL DEMO")
    print("  Dump all vs Keyword vs Semantic — same query, different retrieval")
    print("=" * 70)

    r1 = run_test_1_dump_all()
    r2 = run_test_2_keyword()
    r3 = run_test_3_semantic()
    r4 = run_test_4_agent_conversation()

    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    print(f"\n  {'Strategy':<45} {'Tokens':>10} {'Time':>8} {'Precision':>10}")
    print("  " + "-" * 75)
    print(f"  {'Test 1 — Dump all memory':<45} {r1['tokens']:>10,} {r1['time']:>6.1f}s {'Low':>10}")
    print(f"  {'Test 2 — Keyword search':<45} {r2['tokens']:>10,} {r2['time']:>6.1f}s {'Medium':>10}")
    print(f"  {'Test 3 — Semantic search (top-3)':<45} {r3['tokens']:>10,} {r3['time']:>6.1f}s {'High':>10}")
    print(f"  {'Test 4 — Semantic multi-turn (3 turns)':<45} {r4['tokens']:>10,} {'—':>8} {'High':>10}")

    if r1["tokens"] > r3["tokens"] > 0:
        reduction = (1 - r3["tokens"] / r1["tokens"]) * 100
        print(f"\n  Semantic search uses {reduction:.0f}% fewer tokens than dump-all")

    print(f"\n  Key insight: As memory grows, retrieval strategy determines cost and quality.")
    print(f"  Semantic search retrieves only relevant sections — scales to hundreds of memories.")
    print(f"\n  Research: https://arxiv.org/abs/2501.13956 (Zep)")
    print(f"  Research: https://arxiv.org/abs/2502.14802 (HippoRAG 2)")
    print(f"  Strands:  https://github.com/strands-agents/sdk-python")
