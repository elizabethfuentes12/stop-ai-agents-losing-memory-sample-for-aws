"""Tools for the Core Memory demo.

Core memory allows AI agents to maintain persistent, structured information across
conversations — similar to how humans remember important facts about people they
interact with. This implementation is inspired by MemGPT/MIRIX research.

Implements the Core Memory pattern (MemGPT/MIRIX):
  - core_memory_read: Read a section of core memory
  - core_memory_write: Write/create a new memory entry
  - core_memory_update: Update an existing entry
  - core_memory_list: List all stored memory sections

Core memory is structured into named sections:
  - "persona": Who the user is (name, role, preferences)
  - "preferences": Learned preferences from interactions
  - "history": Key events and past interactions
  - "instructions": User-specified rules and constraints

The agent determines when to read/write core memory based on conversation context
and explicit memory management tools. This is the key insight from MIRIX: the
agent manages its own memory using dedicated tools rather than implicit mechanisms.

State store: agent.state — all memory lives in a single "core_memory" key.

See:
  https://arxiv.org/abs/2507.07957 (MIRIX — 6 memory types)
  https://arxiv.org/abs/2310.08560 (MemGPT — core memory concept)
  https://arxiv.org/abs/2510.07925 (Persistent Memory + User Profiles)
"""

from strands import tool, ToolContext
import json
from datetime import datetime


# ── Core Memory Tools ────────────────────────────────────────────────────────

@tool(context=True)
def core_memory_read(section: str, tool_context: ToolContext) -> str:
    """Read a section of core memory. Returns the stored content for that section.

    Sections: persona, preferences, history, instructions, or any custom section.

    Args:
        section: Memory section name to read (e.g., 'persona', 'preferences')
    """
    memory = tool_context.agent.state.get("core_memory") or {}
    content = memory.get(section)

    if content is None:
        return f"Section '{section}' is empty. No data stored yet."

    return json.dumps({"section": section, "content": content}, indent=2)


@tool(context=True)
def core_memory_write(section: str, content: str, tool_context: ToolContext) -> str:
    """Write or create a new entry in core memory. Overwrites the entire section.

    Use this when storing NEW information about the user for the first time.
    For updates to existing data, prefer core_memory_update.

    Args:
        section: Memory section name (e.g., 'persona', 'preferences')
        content: The content to store (will be JSON-parsed if valid JSON, otherwise stored as string)
    """
    memory = tool_context.agent.state.get("core_memory") or {}

    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        parsed = content

    memory[section] = {
        "data": parsed,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "version": 1,
    }
    tool_context.agent.state.set("core_memory", memory)

    return f"Written to core_memory['{section}']. Version 1."


@tool(context=True)
def core_memory_update(section: str, updates: str, tool_context: ToolContext) -> str:
    """Update an existing section in core memory. Merges new data with existing.

    Use this to ADD or MODIFY fields without losing existing data in the section.

    Args:
        section: Memory section name to update
        updates: JSON string with fields to add or update
    """
    memory = tool_context.agent.state.get("core_memory") or {}

    if section not in memory:
        return f"Section '{section}' does not exist. Use core_memory_write to create it first."

    existing = memory[section]

    try:
        new_data = json.loads(updates)
    except (json.JSONDecodeError, TypeError):
        new_data = updates

    if isinstance(existing["data"], dict) and isinstance(new_data, dict):
        existing["data"].update(new_data)
    else:
        existing["data"] = new_data

    existing["updated_at"] = datetime.now().isoformat()
    existing["version"] = existing.get("version", 1) + 1

    tool_context.agent.state.set("core_memory", memory)

    return f"Updated core_memory['{section}']. Version {existing['version']}."


@tool(context=True)
def core_memory_list(tool_context: ToolContext) -> str:
    """List all sections in core memory with their sizes and last update times."""
    memory = tool_context.agent.state.get("core_memory") or {}

    if not memory:
        return "Core memory is empty. No sections stored."

    sections = []
    for name, entry in memory.items():
        data_size = len(json.dumps(entry.get("data", "")))
        sections.append({
            "section": name,
            "version": entry.get("version", 1),
            "data_size_bytes": data_size,
            "updated_at": entry.get("updated_at", "unknown"),
        })

    return json.dumps({"total_sections": len(sections), "sections": sections}, indent=2)


# ── Hotel tools (same domain as Demo 01, now with core memory) ──────────────

HOTELS = [
    {"name": "Ocean Breeze Resort", "city": "Cancun", "stars": 5, "price": 450, "style": "luxury", "amenities": ["spa", "pool", "gym", "restaurant"], "pet_friendly": False},
    {"name": "Budget Inn Downtown", "city": "Cancun", "stars": 2, "price": 65, "style": "budget", "amenities": ["wifi", "parking"], "pet_friendly": True},
    {"name": "Casa Serena Boutique", "city": "Cancun", "stars": 4, "price": 220, "style": "boutique", "amenities": ["pool", "restaurant", "garden"], "pet_friendly": False},
    {"name": "Grand Palace Hotel", "city": "Tokyo", "stars": 5, "price": 520, "style": "luxury", "amenities": ["spa", "pool", "gym", "restaurant", "concierge"], "pet_friendly": False},
    {"name": "Sakura Capsule", "city": "Tokyo", "stars": 2, "price": 40, "style": "capsule", "amenities": ["wifi", "locker"], "pet_friendly": False},
    {"name": "Zen Garden Ryokan", "city": "Tokyo", "stars": 4, "price": 310, "style": "traditional", "amenities": ["onsen", "garden", "restaurant"], "pet_friendly": False},
    {"name": "Metro Business Hotel", "city": "Tokyo", "stars": 3, "price": 150, "style": "business", "amenities": ["wifi", "gym", "restaurant"], "pet_friendly": False},
    {"name": "Alpine Lodge", "city": "Zurich", "stars": 4, "price": 380, "style": "lodge", "amenities": ["spa", "restaurant", "ski-access"], "pet_friendly": True},
    {"name": "Lake View Apartment", "city": "Zurich", "stars": 3, "price": 200, "style": "apartment", "amenities": ["kitchen", "wifi", "parking"], "pet_friendly": True},
    {"name": "The Ritz Zurich", "city": "Zurich", "stars": 5, "price": 650, "style": "luxury", "amenities": ["spa", "pool", "gym", "restaurant", "concierge"], "pet_friendly": False},
]


@tool(context=True)
def search_hotels_with_memory(city: str, tool_context: ToolContext, max_price: int = 9999) -> str:
    """Search hotels using core memory to personalize results.

    Reads persona and preferences from core memory to rank results.
    Falls back to default ordering if no memory exists.

    Args:
        city: City name to search
        max_price: Maximum price per night in USD (default: no limit)
    """
    results = [h for h in HOTELS if h["city"].lower() == city.lower() and h["price"] <= max_price]
    if not results:
        return f"No hotels found in {city} under ${max_price}/night"

    memory = tool_context.agent.state.get("core_memory") or {}
    prefs_entry = memory.get("preferences", {})
    prefs = prefs_entry.get("data", {}) if isinstance(prefs_entry, dict) else {}

    if prefs and isinstance(prefs, dict):
        preferred_style = prefs.get("preferred_style")
        preferred_stars = prefs.get("preferred_stars")
        preferred_amenities = prefs.get("preferred_amenities", [])

        def score(hotel):
            s = 0
            if preferred_style and hotel["style"] == preferred_style:
                s += 10
            if preferred_stars and hotel["stars"] >= preferred_stars:
                s += 5
            if isinstance(preferred_amenities, list):
                for amenity in preferred_amenities:
                    if amenity in hotel["amenities"]:
                        s += 2
            return s

        results.sort(key=score, reverse=True)
        header = f"Ranked by core memory preferences: {json.dumps(prefs)}\n\n"
    else:
        header = "No preferences in core memory. Results in default order.\n\n"

    return header + json.dumps(results, indent=2)


@tool(context=True)
def book_hotel_with_memory(hotel_name: str, tool_context: ToolContext, nights: int = 1) -> str:
    """Book a hotel. The agent should update core memory with learned preferences.

    Note: This tool books the hotel but does NOT automatically update core memory.
    The agent must decide to call core_memory_update with the learned preferences.
    This is intentional — the agent manages its own memory.

    Args:
        hotel_name: Exact hotel name to book
        nights: Number of nights (default 1)
    """
    import secrets  # Python standard library for cryptographic random generation
    hotel = next((h for h in HOTELS if h["name"].lower() == hotel_name.lower()), None)
    if not hotel:
        return f"Hotel '{hotel_name}' not found"

    total = hotel["price"] * nights
    booking = {
        "status": "CONFIRMED",
        "hotel": hotel["name"],
        "city": hotel["city"],
        "style": hotel["style"],
        "stars": hotel["stars"],
        "amenities": hotel["amenities"],
        "nights": nights,
        "total_cost": total,
        "booking_id": f"BK-{secrets.randbelow(900000) + 100000}",
    }

    return json.dumps(booking, indent=2) + "\n\nTip: Update core memory with the user's preferences learned from this booking."
