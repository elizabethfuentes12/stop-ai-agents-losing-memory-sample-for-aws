"""Tools for the memory retrieval demo.

When core memory grows large (dozens of sections, hundreds of entries),
dumping everything into the context is inefficient. This demo shows three
retrieval strategies:

  1. Dump all — loads entire memory into context (baseline, expensive)
  2. Keyword search — filters by exact keyword match (fast, brittle)
  3. Semantic search — real embeddings + cosine similarity, returns top-k (accurate)

All strategies operate on the same memory store in agent.state.

Semantic search uses a real embedding model (OpenAI text-embedding-3-small),
the same production pattern used by vector databases and RAG systems. Section
embeddings are computed once at seed time and cached in agent.state, so each
query only embeds the query itself — exactly how a production memory store
avoids re-embedding its corpus on every request.

See:
  https://arxiv.org/abs/2501.13956 (Zep — temporal knowledge graph)
  https://arxiv.org/abs/2511.17467 (PersonaAgent with GraphRAG)
  https://arxiv.org/abs/2502.14802 (HippoRAG 2 — associative memory)
"""

from strands import tool, ToolContext
import json
import os
from datetime import datetime, timedelta
import secrets

# ── Embedding backend ────────────────────────────────────────────────────────
# Real embeddings, not a bag-of-words. Default is OpenAI text-embedding-3-small
# (1536 dims), which matches the OpenAI model provider the demo already uses.
#
# To use Amazon Bedrock (Amazon Titan Text Embeddings) instead — no OpenAI key,
# uses your AWS credentials — comment the OpenAI block in _embed_text and
# uncomment the Bedrock block below it.

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

_OPENAI_CLIENT = None
# Simple in-process cache: text -> embedding vector. A production system would
# persist section vectors in a vector DB (see Demo 04 with Neo4j vector index).
_EMBED_CACHE: dict[str, list] = {}


def _get_openai_client():
    """Lazily create a single OpenAI client (reads OPENAI_API_KEY from the env)."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        # Using the OpenAI SDK directly for embeddings (Strands wraps chat models,
        # not the embeddings endpoint). This is a real embedding call, not a stub.
        from openai import OpenAI
        _OPENAI_CLIENT = OpenAI()
    return _OPENAI_CLIENT


def _embed_text(text: str) -> list:
    """Return a real embedding vector for the text, cached by exact text.

    Uses OpenAI text-embedding-3-small by default. Unlike a bag-of-words, this
    captures meaning: 'dietary restrictions' and 'food I avoid' land close
    together even with no shared words.
    """
    if text in _EMBED_CACHE:
        return _EMBED_CACHE[text]

    # --- OpenAI embeddings (default) ---
    client = _get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    vector = resp.data[0].embedding

    # --- Amazon Bedrock (Amazon Titan) embeddings — uncomment to use instead ---
    # import boto3, json as _json
    # bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-west-2"))
    # resp = bedrock.invoke_model(
    #     modelId="amazon.titan-embed-text-v2:0",
    #     body=_json.dumps({"inputText": text}),
    # )
    # vector = _json.loads(resp["body"].read())["embedding"]

    _EMBED_CACHE[text] = vector
    return vector


# ── Memory seed data (simulates a user with rich history) ────────────────────

# Diverse user personas — the demo randomly selects one to show varied profiles
PERSONAS = [
    {
        "persona": {
            "name": "Alex Rivera",
            "role": "senior software engineer",
            "company": "TechCorp",
            "location": "San Francisco",
            "languages": ["English", "Spanish"],
            "travel_frequency": "monthly",
        },
        "travel_preferences": {
            "preferred_style": "boutique",
            "preferred_stars": 4,
            "preferred_amenities": ["wifi", "restaurant", "garden", "quiet"],
            "avoids": ["hostels", "party hotels", "shared rooms"],
            "budget_range": {"min": 150, "max": 400},
            "pet_friendly": False,
        },
        "food_preferences": {
            "dietary": "vegetarian",
            "cuisine_favorites": ["Japanese", "Mediterranean", "Thai"],
            "allergies": ["shellfish"],
            "dislikes": ["fast food", "buffets"],
            "coffee": "oat milk latte, no sugar",
        },
        "work_schedule": {
            "timezone": "PST",
            "meeting_days": ["Monday", "Wednesday", "Friday"],
            "focus_hours": "9am-12pm",
            "travel_blackout": ["Q4 planning (October)", "company retreat (March)"],
            "prefers_redeye": False,
        },
        "past_trips": [
            {"city": "Tokyo", "hotel": "Zen Garden Ryokan", "rating": 5, "date": "2025-01", "notes": "Loved the onsen and garden"},
            {"city": "Barcelona", "hotel": "Casa Camper", "rating": 4, "date": "2025-03", "notes": "Great location, noisy at night"},
            {"city": "Zurich", "hotel": "Alpine Lodge", "rating": 5, "date": "2025-06", "notes": "Spa was exceptional"},
            {"city": "Kyoto", "hotel": "Tawaraya Ryokan", "rating": 5, "date": "2025-08", "notes": "Best traditional experience"},
            {"city": "Lisbon", "hotel": "Memmo Alfama", "rating": 4, "date": "2025-10", "notes": "Rooftop view was worth it"},
            {"city": "Bali", "hotel": "Bambu Indah", "rating": 5, "date": "2025-12", "notes": "Eco-luxury at its best"},
        ],
        "loyalty_programs": {
            "marriott_bonvoy": "Gold Elite",
            "hilton_honors": "Silver",
            "airline_miles": {"united": 45000, "delta": 12000},
            "preferred_airline": "United",
        },
        "communication_style": {
            "tone": "casual but professional",
            "length": "concise, 2-3 sentences",
            "format": "bullet points preferred",
            "language": "English",
            "humor": "light, occasional",
        },
        "emergency_contacts": {
            "primary": "Jordan Rivera (partner)",
            "secondary": "TechCorp travel desk",
            "insurance": "World Nomads policy #WN-2025-1234",
        },
    },
    {
        "persona": {
            "name": "Priya Sharma",
            "role": "university lecturer",
            "company": "Mumbai Institute of Technology",
            "location": "Mumbai",
            "languages": ["Hindi", "English", "Marathi"],
            "travel_frequency": "quarterly",
        },
        "travel_preferences": {
            "preferred_style": "budget",
            "preferred_stars": 3,
            "preferred_amenities": ["wifi", "kitchen", "quiet"],
            "avoids": ["luxury resorts", "party districts"],
            "budget_range": {"min": 30, "max": 120},
            "pet_friendly": False,
        },
        "food_preferences": {
            "dietary": "vegan",
            "cuisine_favorites": ["Indian", "Ethiopian", "Mexican"],
            "allergies": ["peanuts"],
            "dislikes": ["overly processed food"],
            "coffee": "masala chai, no sugar",
        },
        "work_schedule": {
            "timezone": "IST",
            "meeting_days": ["Tuesday", "Thursday"],
            "focus_hours": "6am-9am",
            "travel_blackout": ["exam season (April)", "monsoon break (July)"],
            "prefers_redeye": True,
        },
        "past_trips": [
            {"city": "Jaipur", "hotel": "Heritage Haveli", "rating": 5, "date": "2025-02", "notes": "Authentic Rajasthani experience"},
            {"city": "Nairobi", "hotel": "Backpacker Lodge", "rating": 3, "date": "2025-05", "notes": "Great value, friendly staff"},
            {"city": "Hanoi", "hotel": "Old Quarter Guesthouse", "rating": 4, "date": "2025-09", "notes": "Walking distance to everything"},
        ],
        "loyalty_programs": {
            "air_india": "Gold",
            "airline_miles": {"air_india": 28000},
            "preferred_airline": "Air India",
        },
        "communication_style": {
            "tone": "warm and detailed",
            "length": "thorough, 4-5 sentences",
            "format": "numbered lists",
            "language": "English",
            "humor": "minimal",
        },
        "emergency_contacts": {
            "primary": "Raj Sharma (sibling)",
            "secondary": "MIT travel office",
            "insurance": "ICICI Lombard policy #IL-2025-5678",
        },
    },
    {
        "persona": {
            "name": "Amara Okafor",
            "role": "freelance photographer",
            "company": "Self-employed",
            "location": "Lagos",
            "languages": ["English", "Yoruba", "French"],
            "travel_frequency": "weekly",
        },
        "travel_preferences": {
            "preferred_style": "guesthouse",
            "preferred_stars": 3,
            "preferred_amenities": ["wifi", "parking", "local-tours"],
            "avoids": ["chain hotels", "tourist traps"],
            "budget_range": {"min": 40, "max": 200},
            "pet_friendly": True,
        },
        "food_preferences": {
            "dietary": "no restrictions",
            "cuisine_favorites": ["West African", "Brazilian", "Korean"],
            "allergies": [],
            "dislikes": ["bland hotel breakfasts"],
            "coffee": "black coffee, strong",
        },
        "work_schedule": {
            "timezone": "WAT",
            "meeting_days": ["Monday"],
            "focus_hours": "5am-8am",
            "travel_blackout": ["rainy season shoots (June)"],
            "prefers_redeye": True,
        },
        "past_trips": [
            {"city": "Accra", "hotel": "Coconut Grove", "rating": 4, "date": "2025-01", "notes": "Vibrant art scene nearby"},
            {"city": "Salvador", "hotel": "Pousada Tropical", "rating": 5, "date": "2025-04", "notes": "Best sunset photos"},
            {"city": "Marrakech", "hotel": "Riad Jardin", "rating": 4, "date": "2025-07", "notes": "Colors everywhere"},
            {"city": "Seoul", "hotel": "Bukchon Hanok Stay", "rating": 5, "date": "2025-11", "notes": "Traditional meets modern"},
        ],
        "loyalty_programs": {
            "airline_miles": {"ethiopian": 32000, "latam": 15000},
            "preferred_airline": "Ethiopian Airlines",
        },
        "communication_style": {
            "tone": "enthusiastic and direct",
            "length": "concise, 1-2 sentences",
            "format": "free-form",
            "language": "English",
            "humor": "frequent",
        },
        "emergency_contacts": {
            "primary": "Chidi Okafor (parent)",
            "secondary": "Freelancers Union Lagos",
            "insurance": "AXA Mansard policy #AM-2025-9012",
        },
    },
]


def _section_text(section: str, data) -> str:
    """Flatten a memory section into the text we embed (section name + content)."""
    return f"{section}: {json.dumps(data)}"


def seed_memory(agent, persona_index: int = 0) -> int:
    """Populate agent.state with a realistic user memory profile. Returns memory count.

    Also pre-computes and caches a real embedding for each section — the way a
    production memory store embeds its corpus once at write time, not per query.

    Args:
        agent: The agent whose state will be populated
        persona_index: Index into PERSONAS list (0=Alex, 1=Priya, 2=Amara)
    """
    persona_data = PERSONAS[persona_index % len(PERSONAS)]
    memories = {}
    for section, data in persona_data.items():
        memories[section] = {
            "data": data,
            "embedding": _embed_text(_section_text(section, data)),  # computed once, cached
            "created_at": (datetime.now() - timedelta(days=90)).isoformat(),
            "updated_at": (datetime.now() - timedelta(days=5)).isoformat(),
            "version": 3,
        }

    agent.state.set("core_memory", memories)
    return len(memories)


# ── Strategy 1: Dump all memory (baseline) ──────────────────────────────────

@tool(context=True)
def memory_dump_all(tool_context: ToolContext) -> str:
    """Retrieve ALL core memory sections at once. Returns the complete memory dump.

    Warning: This can be very large and inefficient for targeted queries.
    """
    memory = tool_context.agent.state.get("core_memory") or {}
    if not memory:
        return "Core memory is empty."

    # Dump the content, not the raw embedding vectors (those are an internal index,
    # never sent to the model — dumping 1536-float vectors would just burn tokens).
    dump = {section: {k: v for k, v in entry.items() if k != "embedding"}
            for section, entry in memory.items()}
    total_size = len(json.dumps(dump))
    return f"Full memory dump ({total_size:,} bytes, {len(dump)} sections):\n\n{json.dumps(dump, indent=2)}"


# ── Strategy 2: Keyword search ──────────────────────────────────────────────

@tool(context=True)
def memory_search_keyword(keyword: str, tool_context: ToolContext) -> str:
    """Search core memory by exact keyword match across all sections.

    Searches section names and stringified content for the keyword.
    Fast but brittle — misses synonyms and related concepts.

    Args:
        keyword: Exact keyword to search for (case-insensitive)
    """
    memory = tool_context.agent.state.get("core_memory") or {}
    if not memory:
        return "Core memory is empty."

    keyword_lower = keyword.lower()
    matches = []

    for section, entry in memory.items():
        content_str = json.dumps(entry.get("data", "")).lower()
        if keyword_lower in section.lower() or keyword_lower in content_str:
            matches.append({
                "section": section,
                "data": entry.get("data"),
                "relevance": "keyword_match",
            })

    if not matches:
        return f"No matches for '{keyword}' in core memory."

    total_size = len(json.dumps(matches))
    return f"Found {len(matches)} matches for '{keyword}' ({total_size:,} bytes):\n\n{json.dumps(matches, indent=2)}"


# ── Strategy 3: Semantic search (embedding-based) ───────────────────────────

def _cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity measures how similar two vectors are, ranging from -1 (opposite)
    to 1 (identical). A score of 0 means the vectors are unrelated.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@tool(context=True)
def memory_search_semantic(query: str, tool_context: ToolContext, top_k: int = 3) -> str:
    """Search core memory using semantic similarity. Returns the most relevant sections.

    Embeds the query with a real embedding model, compares it against each section's
    pre-computed embedding by cosine similarity, and returns only the top-k sections.
    Because it matches on meaning (not shared words), it finds the right section even
    when the query uses synonyms the memory never mentions.

    Args:
        query: Natural language query describing what you need
        top_k: Number of top results to return (default 3)
    """
    memory = tool_context.agent.state.get("core_memory") or {}
    if not memory:
        return "Core memory is empty."

    query_embedding = _embed_text(query)

    scored = []
    for section, entry in memory.items():
        # Section embeddings are cached from seed time; embed on the fly if missing.
        section_embedding = entry.get("embedding") or _embed_text(_section_text(section, entry.get("data", "")))
        similarity = _cosine_similarity(query_embedding, section_embedding)
        scored.append({
            "section": section,
            "similarity": round(similarity, 3),
            "data": entry.get("data"),
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top_results = scored[:top_k]

    # Size comparison uses the model-visible payload (data only, never the vectors).
    result_payload = [{"section": r["section"], "similarity": r["similarity"], "data": r["data"]} for r in top_results]
    total_size = len(json.dumps(result_payload))
    all_size = len(json.dumps({s: e.get("data") for s, e in memory.items()}))
    savings = round((1 - total_size / all_size) * 100) if all_size > 0 else 0

    return (
        f"Top {top_k} relevant sections ({total_size:,} bytes vs {all_size:,} full dump, "
        f"{savings}% reduction):\n\n{json.dumps(result_payload, indent=2)}"
    )
