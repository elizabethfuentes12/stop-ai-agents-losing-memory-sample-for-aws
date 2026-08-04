"""Tools for the agent-managed memory demo — the agent decides what to remember.

Demo 01 hardwired WHAT gets remembered (the booking tool always wrote the same
preference fields). Here the memory tools are generic — read / write / update /
list over named sections — and the AGENT decides what is worth keeping, where to
file it, and when to update it. The pattern comes from MemGPT ("core memory")
and MIRIX.

MEMORY TYPES: the demo instructs the agent (via system prompt) to organize its
memory into four typed sections. These are the same four types Amazon Bedrock
AgentCore Memory later gives you as built-in strategies — here you see the
concept implemented with nothing but a prompt and generic tools:

  | Section       | What the agent files there            | AgentCore built-in |
  |---------------|---------------------------------------|--------------------|
  | facts         | durable facts about the user's world  | semantic           |
  | preferences   | likes/dislikes revealed by the user   | userPreference     |
  | trip_summary  | rolling summary of the current plan   | summary            |
  | episodes      | notable events, one entry per episode | episodic           |

The flight and climate tools call live APIs (Duffel sandbox, Open-Meteo) — no
hardcoded data. They are deliberately memory-free here: in this demo ALL
remembering flows through the agent's explicit memory decisions.

State store: agent.state — all sections live under one "core_memory" key.

See:
  https://arxiv.org/abs/2507.07957 (MIRIX — typed memory)
  https://arxiv.org/abs/2310.08560 (MemGPT — core memory concept)
  https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/built-in-strategies.html
"""

import json
from datetime import datetime

from strands import tool, ToolContext

import flights_api
import weather_api


# ── Memory tools (generic on purpose — the TYPES come from the system prompt) ──

@tool(context=True)
def core_memory_read(section: str, tool_context: ToolContext) -> str:
    """Read one section of your memory before answering questions about the user.

    Use this tool when:
    - the user asks something their profile could answer ("what do I like?")
    - you are about to personalize a recommendation and need their preferences
    - you need to check what you already know before writing (avoid duplicates)

    Args:
        section: Section name, e.g. "facts", "preferences", "trip_summary", "episodes".

    Returns:
        JSON: {"section": "...", "content": ...} or a message if the section is empty.
    """
    memory = tool_context.agent.state.get("core_memory") or {}
    content = memory.get(section)
    if content is None:
        return f"Section '{section}' is empty. No data stored yet."
    return json.dumps({"section": section, "content": content}, indent=1)


@tool(context=True)
def core_memory_write(section: str, content: str, tool_context: ToolContext) -> str:
    """Create a memory section the FIRST time you learn something worth keeping.

    Use this tool when:
    - the user reveals durable information you don't have yet (name, home airport,
      dietary needs, mileage program...)
    - a new episode happens worth recording (a booking, a cancelled trip)
    Overwrites the whole section — for incremental changes use core_memory_update.

    Args:
        section: Section name, e.g. "facts", "preferences", "trip_summary", "episodes".
        content: What to store. Valid JSON is parsed; anything else is kept as text.

    Returns:
        Confirmation with the section name and version (always 1 for a fresh write).
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
    return f"Stored core_memory['{section}'] (version 1)."


@tool(context=True)
def core_memory_update(section: str, updates: str, tool_context: ToolContext) -> str:
    """Update an existing memory section when information changes or grows.

    Use this tool when:
    - the user changes their mind ("actually I prefer window seats now")
    - you learn an additional detail for a section that already exists
    Merges dict updates into the existing data (other shapes are replaced), and
    bumps the section version so evolution is visible.

    Args:
        section: Existing section name to update.
        updates: JSON string with the fields to add or change.

    Returns:
        Confirmation with the new version, or guidance to write first if missing.
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
    return f"Updated core_memory['{section}'] (version {existing['version']})."


@tool(context=True)
def core_memory_list(tool_context: ToolContext) -> str:
    """Check which memory sections exist before reading or writing.

    Use this tool when:
    - starting to help a user and you want to know what you already know
    - deciding whether to write a new section or update an existing one

    Returns:
        JSON: {"total_sections": N, "sections": [{"section", "version",
        "data_size_bytes", "updated_at"}, ...]} or a message if memory is empty.
    """
    memory = tool_context.agent.state.get("core_memory") or {}
    if not memory:
        return "Core memory is empty. No sections stored."
    sections = [{
        "section": name,
        "version": entry.get("version", 1),
        "data_size_bytes": len(json.dumps(entry.get("data", ""))),
        "updated_at": entry.get("updated_at", "unknown"),
    } for name, entry in memory.items()]
    return json.dumps({"total_sections": len(sections), "sections": sections}, indent=1)


# ── Domain tools (live APIs, deliberately memory-free in this demo) ─────────

@tool
def search_flights(origin: str, destination: str, departure_date: str,
                   cabin_class: str = "economy") -> str:
    """Search live flight offers when the user wants to fly somewhere on a date.

    Use this tool when the user:
    - asks for flights between two airports ("flights from JFK to Paris")
    - wants prices for a route on a date ("how much to Tokyo on the 15th?")
    Note: results are NOT personalized by this tool — read the user's memory
    sections first and apply their preferences when you present options.

    Args:
        origin: IATA airport code of departure, e.g. "JFK".
        destination: IATA airport code of arrival, e.g. "CDG".
        departure_date: ISO date, e.g. "2026-09-15".
        cabin_class: One of: economy, premium_economy, business, first.

    Returns:
        JSON list of real offers, cheapest first. Each offer:
        {"offer_id": "off_...", "price": 225.08, "currency": "USD",
         "cabin": "economy", "slices": [{"origin", "destination", "stops",
         "segments": [{"carrier", ...}]}]}
    """
    offers = flights_api.search_offers(origin, destination, departure_date, cabin_class)
    return json.dumps(offers, indent=1)


@tool
def book_flight(offer_id: str) -> str:
    """Confirm a booking when the user has chosen one of the searched offers.

    Use this tool when the user:
    - picks an option from a previous search ("book the Iberia one")
    - says "book it", "reserve that flight", "take the cheapest"
    After booking, decide what this action reveals about the user and file it
    into the right memory sections (preferences, episodes, trip_summary).

    Args:
        offer_id: The Duffel offer id from a previous search result, e.g. "off_0000B8...".

    Returns:
        JSON: {"status": "CONFIRMED", "offer_id", "price", "currency", "route",
               "cabin", "carriers"} — or an error string if the offer expired.
    """
    offer = flights_api.get_offer(offer_id)
    if offer is None:
        return f"Offer '{offer_id}' not found or expired. Search again and book a fresh offer."
    carriers = sorted({seg["carrier"] for sl in offer["slices"] for seg in sl["segments"] if seg["carrier"]})
    return json.dumps({
        "status": "CONFIRMED", "offer_id": offer_id,
        "price": offer["price"], "currency": offer["currency"],
        "route": " -> ".join(f"{sl['origin']}-{sl['destination']}" for sl in offer["slices"]),
        "cabin": offer["cabin"], "carriers": carriers,
    })


@tool
def best_time_to_visit(city: str) -> str:
    """Answer "when should I visit X?" with real historical climate by month.

    Use this tool when the user:
    - asks the best time/season/month to travel somewhere
    - is choosing travel dates and wants to avoid heat or rainy season
    Cross-check the months you recommend against the user's remembered
    constraints (e.g. school holidays, weather dislikes) if you have them.

    Args:
        city: City name, e.g. "Tokyo".

    Returns:
        JSON with 12 rows of real monthly averages (avg_high_c, avg_low_c,
        avg_precip_mm) plus the years averaged and the data source.
    """
    climate = weather_api.monthly_climate(city)
    if climate is None:
        return f"Could not retrieve climate data for '{city}'. Check the city name."
    return json.dumps(climate, indent=1)
