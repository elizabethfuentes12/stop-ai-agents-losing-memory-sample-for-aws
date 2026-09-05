"""Domain tools for the selective-memory demo, live flight and climate APIs.

Memory is NOT handled here. In this demo, remembering is done by Strands' native
memory framework (`MemoryManager` + a `MemoryStore` + a `ModelExtractor`; see
`memory_config.py`), not by hand-rolled memory tools and not by instructions in the
chat agent's system prompt. These tools are the agent's *domain* capabilities only.

The flight and climate tools call live APIs (Duffel sandbox, Open-Meteo), no
hardcoded data. They are deliberately memory-free: the agent just talks and acts,
and the `MemoryManager` extracts what's worth keeping off the conversation path.
"""

import json

from strands import tool

import flights_api
import weather_api


@tool
def search_flights(origin: str, destination: str, departure_date: str,
                   cabin_class: str = "economy") -> str:
    """Search live flight offers when the user wants to fly somewhere on a date.

    Use this tool when the user:
    - asks for flights between two airports ("flights from JFK to Paris")
    - wants prices for a route on a date ("how much to Tokyo on the 15th?")

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

    Args:
        offer_id: The Duffel offer id from a previous search result, e.g. "off_0000B8...".

    Returns:
        JSON: {"status": "CONFIRMED", "offer_id", "price", "currency", "route",
               "cabin", "carriers"}, or an error string if the offer expired.
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
