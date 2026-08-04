"""Tools for the persistent-memory demo — real APIs, no hardcoded data.

A brand-new user arrives with EMPTY memory and searches flights. Two versions of
the same tools demonstrate the difference between stateless (no memory) and
stateful (agent.state) designs:

  - Stateless: search/book return real Duffel offers, learn nothing.
  - Stateful: booking a flight LEARNS the user's preferences (cabin, price range,
    non-stop tolerance, carriers); the next search RANKS real offers by them.

Data sources (both real, both verified live):
  - Flights: Duffel sandbox API (flights_api.py) — real offers, real carriers.
  - Climate: Open-Meteo historical archive (weather_api.py) — real monthly averages.

Tool descriptions follow the research-backed pattern (ToolLLM/AgentTuning): the
first sentence says WHEN to use the tool, trigger phrases are listed, parameters
carry example values, and the return shape is documented. The system prompt does
NOT re-describe the tools — the agent learns their purpose from this context.

State store: agent.state — preferences persist across turns within a session;
FileSessionManager persists them across restarts (Test 3).
"""

import json

from strands import tool, ToolContext

import flights_api
import weather_api


# ── Stateless tools (no memory — baseline) ──────────────────────────────────

@tool
def search_flights_stateless(origin: str, destination: str, departure_date: str,
                             cabin_class: str = "economy") -> str:
    """Search live flight offers when the user wants to fly somewhere on a date.

    Use this tool when the user:
    - asks for flights between two cities/airports ("flights from JFK to Paris")
    - wants prices for a route on a date ("how much to Tokyo on the 15th?")
    - asks to compare airlines or fares for a trip

    Args:
        origin: IATA airport code of departure, e.g. "JFK".
        destination: IATA airport code of arrival, e.g. "CDG".
        departure_date: ISO date, e.g. "2026-09-15".
        cabin_class: One of: economy, premium_economy, business, first.

    Returns:
        JSON list of offers, cheapest first. Each offer:
        {"offer_id": "off_...", "price": 225.08, "currency": "USD",
         "cabin": "economy", "slices": [{"origin": "JFK", "destination": "CDG",
         "stops": 0, "segments": [{"carrier": "British Airways", ...}]}]}
    """
    offers = flights_api.search_offers(origin, destination, departure_date, cabin_class)
    return json.dumps(offers, indent=1)


@tool
def book_flight_stateless(offer_id: str) -> str:
    """Confirm a booking when the user has chosen one of the searched offers.

    Use this tool when the user:
    - picks an option from a previous search ("book the British Airways one")
    - says "book it", "reserve that flight", "take the cheapest"

    Args:
        offer_id: The Duffel offer id from a previous search result, e.g. "off_0000B8...".

    Returns:
        JSON: {"status": "CONFIRMED", "offer_id": "off_...", "price": 1392.10,
               "currency": "USD", "route": "JFK-CDG"}
        Or an error string if the offer expired (search again for a fresh one).
    """
    # Identical business logic to the stateful book_flight — retrieve the REAL chosen
    # offer and confirm it. The ONLY difference is what comes after: this version has
    # no ToolContext, so nothing about the choice can be remembered.
    offer = flights_api.get_offer(offer_id)
    if offer is None:
        return f"Offer '{offer_id}' not found or expired. Search again and book a fresh offer."
    return json.dumps({
        "status": "CONFIRMED", "offer_id": offer_id,
        "price": offer["price"], "currency": offer["currency"],
        "route": " -> ".join(f"{sl['origin']}-{sl['destination']}" for sl in offer["slices"]),
    })


# ── Stateful tools (with agent.state memory) ────────────────────────────────

@tool(context=True)
def search_flights(origin: str, destination: str, departure_date: str,
                   tool_context: ToolContext, cabin_class: str = "economy") -> str:
    """Search live flight offers when the user wants to fly somewhere on a date.

    Use this tool when the user:
    - asks for flights between two cities/airports ("flights from JFK to Paris")
    - wants prices for a route on a date ("how much to Tokyo on the 15th?")
    - asks for recommendations "for me" (results come ranked by their learned profile)

    Results are ranked by the user's learned preferences when a profile exists
    (preferred cabin, non-stop tolerance, typical price range, carriers flown).

    Args:
        origin: IATA airport code of departure, e.g. "JFK".
        destination: IATA airport code of arrival, e.g. "CDG".
        departure_date: ISO date, e.g. "2026-09-15".
        cabin_class: One of: economy, premium_economy, business, first.
            A learned cabin preference overrides this default.

    Returns:
        A header line stating which preferences ranked the results (or that no
        profile exists yet), then the same JSON offer list as the stateless search.
    """
    prefs = tool_context.agent.state.get("user_preferences") or {}
    effective_cabin = prefs.get("preferred_cabin") or cabin_class

    offers = flights_api.search_offers(origin, destination, departure_date, effective_cabin)

    if prefs:
        max_price = prefs.get("typical_price", {}).get("max")

        def score(offer):
            s = 0
            if all(sl["stops"] == 0 for sl in offer["slices"]) and prefs.get("prefers_nonstop"):
                s += 10
            if max_price and offer["price"] <= max_price:
                s += 5
            carriers = {seg["carrier"] for sl in offer["slices"] for seg in sl["segments"]}
            if carriers & set(prefs.get("carriers_flown", [])):
                s += 2
            return s

        offers.sort(key=score, reverse=True)
        header = f"Ranked by your preferences: {json.dumps(prefs)}\n\n"
    else:
        header = "No preferences learned yet. Book a flight to start building your profile.\n\n"

    return header + json.dumps(offers, indent=1)


@tool(context=True)
def book_flight(offer_id: str, tool_context: ToolContext) -> str:
    """Confirm a booking AND learn the user's preferences from their choice.

    Use this tool when the user:
    - picks an option from a previous search ("book the Iberia one")
    - says "book it", "reserve that flight", "take the cheapest"

    Booking is the strongest preference signal there is: the chosen offer's cabin,
    stops, price, and carriers are stored in the user's profile so future searches
    rank by them.

    Args:
        offer_id: The Duffel offer id from a previous search result, e.g. "off_0000B8...".

    Returns:
        JSON: {"status": "CONFIRMED", "offer_id": "off_...",
               "preferences_updated": {...the learned profile...}}
        Or an error string if the offer expired (search again for a fresh one).
    """
    offer = flights_api.get_offer(offer_id)
    if offer is None:
        return f"Offer '{offer_id}' not found or expired. Search again and book a fresh offer."

    prefs = tool_context.agent.state.get("user_preferences") or {}
    prefs["preferred_cabin"] = offer["cabin"]
    prefs["prefers_nonstop"] = all(sl["stops"] == 0 for sl in offer["slices"])
    carriers = sorted({seg["carrier"] for sl in offer["slices"] for seg in sl["segments"] if seg["carrier"]})
    prefs["carriers_flown"] = sorted(set(prefs.get("carriers_flown", [])) | set(carriers))
    price_range = prefs.get("typical_price", {})
    prefs["typical_price"] = {
        "min": min(price_range.get("min", offer["price"]), offer["price"]),
        "max": max(price_range.get("max", offer["price"]), offer["price"]),
    }

    history = tool_context.agent.state.get("booking_history") or []
    history.append({
        "offer_id": offer["offer_id"], "price": offer["price"], "currency": offer["currency"],
        "cabin": offer["cabin"],
        "route": " -> ".join(f"{sl['origin']}-{sl['destination']}" for sl in offer["slices"]),
        "carriers": carriers,
    })

    tool_context.agent.state.set("user_preferences", prefs)
    tool_context.agent.state.set("booking_history", history)

    return json.dumps({"status": "CONFIRMED", "offer_id": offer_id,
                       "preferences_updated": prefs})


@tool(context=True)
def get_user_profile(tool_context: ToolContext) -> str:
    """Show what the agent has learned about this user so far.

    Use this tool when the user:
    - asks "what do you know about me?" / "what are my preferences?"
    - asks for personalized advice and you need their profile first
    - wants to verify or correct what has been remembered

    Returns:
        JSON: {"preferences": {...}, "booking_history": [...], "total_bookings": N}
        Or a plain message when no profile exists yet.
    """
    prefs = tool_context.agent.state.get("user_preferences")
    history = tool_context.agent.state.get("booking_history")
    if not prefs and not history:
        return "No user profile yet. The agent has not learned any preferences."
    return json.dumps({
        "preferences": prefs or {},
        "booking_history": history or [],
        "total_bookings": len(history) if history else 0,
    }, indent=1)


# ── Shared (works with either version — climate is user-independent) ────────

@tool
def best_time_to_visit(city: str) -> str:
    """Answer "when should I visit X?" with real historical climate by month.

    Use this tool when the user:
    - asks the best time/season/month to travel somewhere
    - asks what the weather is like in a city at some time of year
    - is choosing travel dates and wants to avoid heat or rainy season

    Args:
        city: City name, e.g. "Tokyo".

    Returns:
        JSON with 12 rows of real monthly averages:
        {"city": "Tokyo", "country": "Japan", "years_averaged": "2021-2025",
         "months": [{"month": "January", "avg_high_c": 9.9, "avg_low_c": 0.6,
                     "avg_precip_mm": 33.0}, ...],
         "source": "Open-Meteo ERA5 archive (CC-BY 4.0), geocoding by GeoNames"}
        Recommend months balancing mild temperatures against low precipitation.
    """
    climate = weather_api.monthly_climate(city)
    if climate is None:
        return f"Could not retrieve climate data for '{city}'. Check the city name."
    return json.dumps(climate, indent=1)
