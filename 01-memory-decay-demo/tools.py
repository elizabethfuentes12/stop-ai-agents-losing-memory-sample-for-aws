"""Tools for the memory decay demo.

Simulates a travel assistant that learns user preferences over multiple
conversation turns. Two versions of the same tool demonstrate the difference
between stateless (no memory) and stateful (agent.state) approaches.

State store: agent.state — preferences persist across turns within a session.

See:
  https://github.com/strands-agents/sdk-python#agent-state
  https://github.com/strands-agents/sdk-python#conversation-management
"""

from strands import tool, ToolContext
import json
import secrets

# ── Simulated hotel database ────────────────────────────────────────────────

HOTELS = [
    {"name": "Ocean Breeze Resort", "city": "Cancun", "stars": 5, "price": 450, "style": "luxury", "amenities": ["spa", "pool", "gym", "restaurant"], "pet_friendly": False},
    {"name": "Budget Inn Downtown", "city": "Cancun", "stars": 2, "price": 65, "style": "budget", "amenities": ["wifi", "parking"], "pet_friendly": True},
    {"name": "Casa Serena Boutique", "city": "Cancun", "stars": 4, "price": 220, "style": "boutique", "amenities": ["pool", "restaurant", "garden"], "pet_friendly": False},
    {"name": "Backpacker's Haven", "city": "Cancun", "stars": 1, "price": 25, "style": "hostel", "amenities": ["wifi", "kitchen"], "pet_friendly": False},
    {"name": "Grand Palace Hotel", "city": "Tokyo", "stars": 5, "price": 520, "style": "luxury", "amenities": ["spa", "pool", "gym", "restaurant", "concierge"], "pet_friendly": False},
    {"name": "Sakura Capsule", "city": "Tokyo", "stars": 2, "price": 40, "style": "capsule", "amenities": ["wifi", "locker"], "pet_friendly": False},
    {"name": "Zen Garden Ryokan", "city": "Tokyo", "stars": 4, "price": 310, "style": "traditional", "amenities": ["onsen", "garden", "restaurant"], "pet_friendly": False},
    {"name": "Metro Business Hotel", "city": "Tokyo", "stars": 3, "price": 150, "style": "business", "amenities": ["wifi", "gym", "restaurant"], "pet_friendly": False},
    {"name": "Alpine Lodge", "city": "Zurich", "stars": 4, "price": 380, "style": "lodge", "amenities": ["spa", "restaurant", "ski-access"], "pet_friendly": True},
    {"name": "Lake View Apartment", "city": "Zurich", "stars": 3, "price": 200, "style": "apartment", "amenities": ["kitchen", "wifi", "parking"], "pet_friendly": True},
    {"name": "The Ritz Zurich", "city": "Zurich", "stars": 5, "price": 650, "style": "luxury", "amenities": ["spa", "pool", "gym", "restaurant", "concierge"], "pet_friendly": False},
    {"name": "Hostel Zurich Central", "city": "Zurich", "stars": 1, "price": 35, "style": "hostel", "amenities": ["wifi", "kitchen"], "pet_friendly": False},
    {"name": "Riad Jardin", "city": "Marrakech", "stars": 3, "price": 130, "style": "boutique", "amenities": ["garden", "restaurant", "wifi"], "pet_friendly": False},
    {"name": "Backpacker Lodge", "city": "Nairobi", "stars": 1, "price": 20, "style": "hostel", "amenities": ["wifi", "kitchen", "tours"], "pet_friendly": False},
    {"name": "Grand Mekong Hotel", "city": "Phnom Penh", "stars": 4, "price": 90, "style": "business", "amenities": ["pool", "gym", "restaurant", "wifi"], "pet_friendly": False},
    {"name": "Pousada Tropical", "city": "Salvador", "stars": 3, "price": 75, "style": "guesthouse", "amenities": ["pool", "wifi", "parking"], "pet_friendly": True},
]


# ── Stateless tools (no memory — baseline) ──────────────────────────────────

@tool
def search_hotels_stateless(city: str, max_price: int = 9999) -> str:
    """Search hotels in a city. Returns matching hotels as JSON.

    Args:
        city: City name to search
        max_price: Maximum price per night in USD (default: no limit)
    """
    results = [h for h in HOTELS if h["city"].lower() == city.lower() and h["price"] <= max_price]
    if not results:
        return f"No hotels found in {city} under ${max_price}/night"
    return json.dumps(results, indent=2)


@tool
def book_hotel_stateless(hotel_name: str, nights: int = 1) -> str:
    """Book a hotel by name.

    Args:
        hotel_name: Exact hotel name to book
        nights: Number of nights (default 1)
    """
    hotel = next((h for h in HOTELS if h["name"].lower() == hotel_name.lower()), None)
    if not hotel:
        return f"Hotel '{hotel_name}' not found"
    total = hotel["price"] * nights
    return json.dumps({
        "status": "CONFIRMED",
        "hotel": hotel["name"],
        "city": hotel["city"],
        "nights": nights,
        "total_cost": total,
        "booking_id": f"BK-{secrets.randbelow(900000) + 100000}",
    }, indent=2)


# ── Stateful tools (with agent.state memory) ────────────────────────────────

@tool(context=True)
def search_hotels(city: str, tool_context: ToolContext, max_price: int = 9999) -> str:
    """Search hotels in a city. Learns user preferences from past interactions.

    Args:
        city: City name to search
        max_price: Maximum price per night in USD (default: no limit)
    """
    results = [h for h in HOTELS if h["city"].lower() == city.lower() and h["price"] <= max_price]
    if not results:
        return f"No hotels found in {city} under ${max_price}/night"

    # Read learned preferences from agent.state
    prefs = tool_context.agent.state.get("user_preferences") or {}

    # Rank results based on known preferences
    if prefs:
        preferred_style = prefs.get("preferred_style")
        preferred_stars = prefs.get("preferred_stars")
        preferred_amenities = prefs.get("preferred_amenities", [])

        def score(hotel):
            s = 0
            if preferred_style and hotel["style"] == preferred_style:
                s += 10
            if preferred_stars and hotel["stars"] >= preferred_stars:
                s += 5
            for amenity in preferred_amenities:
                if amenity in hotel["amenities"]:
                    s += 2
            return s

        results.sort(key=score, reverse=True)
        header = f"Ranked by your preferences: {json.dumps(prefs)}\n\n"
    else:
        header = "No preferences learned yet. Book a hotel to start building your profile.\n\n"

    return header + json.dumps(results, indent=2)


@tool(context=True)
def book_hotel(hotel_name: str, tool_context: ToolContext, nights: int = 1) -> str:
    """Book a hotel and learn user preferences from the choice.

    Args:
        hotel_name: Exact hotel name to book
        nights: Number of nights (default 1)
    """
    hotel = next((h for h in HOTELS if h["name"].lower() == hotel_name.lower()), None)
    if not hotel:
        return f"Hotel '{hotel_name}' not found"

    total = hotel["price"] * nights
    booking = {
        "status": "CONFIRMED",
        "hotel": hotel["name"],
        "city": hotel["city"],
        "nights": nights,
        "total_cost": total,
        "booking_id": f"BK-{secrets.randbelow(900000) + 100000}",
    }

    # Update user preferences in agent.state
    prefs = tool_context.agent.state.get("user_preferences") or {}
    prefs["preferred_style"] = hotel["style"]
    prefs["preferred_stars"] = hotel["stars"]
    prefs["preferred_amenities"] = list(set(prefs.get("preferred_amenities", []) + hotel["amenities"]))
    prefs["price_range"] = {"min": min(prefs.get("price_range", {}).get("min", hotel["price"]), hotel["price"]),
                            "max": max(prefs.get("price_range", {}).get("max", hotel["price"]), hotel["price"])}

    # Track booking history
    history = tool_context.agent.state.get("booking_history") or []
    history.append({"hotel": hotel["name"], "city": hotel["city"], "style": hotel["style"], "stars": hotel["stars"]})

    tool_context.agent.state.set("user_preferences", prefs)
    tool_context.agent.state.set("booking_history", history)

    booking["preferences_updated"] = prefs
    return json.dumps(booking, indent=2)


@tool(context=True)
def get_user_profile(tool_context: ToolContext) -> str:
    """Retrieve the current user profile built from past interactions."""
    prefs = tool_context.agent.state.get("user_preferences")
    history = tool_context.agent.state.get("booking_history")

    if not prefs and not history:
        return "No user profile yet. The agent has not learned any preferences."

    profile = {
        "preferences": prefs or {},
        "booking_history": history or [],
        "total_bookings": len(history) if history else 0,
    }
    return json.dumps(profile, indent=2)
