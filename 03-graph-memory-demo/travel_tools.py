"""Business tools for the graph-memory travel assistant.

Memory is NOT handled here anymore. In this demo the agent's memory is a native
Strands ``MemoryStore`` (see ``graph_memory_store.GraphMemoryStore``) wired through
the ``MemoryManager``, which registers the recall tool, runs extraction, and injects
recalled memories automatically. These are just the domain tools the assistant calls.

  - search_flights: search flight offers (Duffel).
  - book_flight: confirm a booking.
  - best_time_to_visit: historical climate to answer "when should I go to X?".
"""

import json

from strands import tool

import flights_api
import weather_api


@tool
def search_flights(origin: str, destination: str, departure_date: str,
                   cabin_class: str = "economy") -> str:
    """Search flight offers for a route on a date.

    Args:
        origin: IATA code of departure, e.g. "JFK".
        destination: IATA code of arrival, e.g. "MAD".
        departure_date: ISO date, e.g. "2026-10-15".
        cabin_class: economy, premium_economy, business, or first.
    """
    try:
        offers = flights_api.search_offers(origin, destination, departure_date,
                                           cabin_class, max_results=4)
    except Exception as exc:
        return f"Flight search failed: {exc}. Ask the user to retry."
    return json.dumps(offers, indent=1) if offers else "No offers found."


@tool
def book_flight(offer_id: str) -> str:
    """Confirm a booking for a chosen offer id from a previous search.

    Args:
        offer_id: the Duffel offer id, e.g. "off_0000B8...".
    """
    offer = flights_api.get_offer(offer_id)
    if offer is None:
        return f"Offer '{offer_id}' not found or expired. Search again."
    return json.dumps({"status": "CONFIRMED", "offer_id": offer_id,
                       "price": offer["price"], "currency": offer["currency"]})


@tool
def best_time_to_visit(city: str) -> str:
    """Answer "when should I visit X?" with historical monthly climate.

    Args:
        city: city name, e.g. "Madrid".
    """
    climate = weather_api.monthly_climate(city)
    return json.dumps(climate, indent=1) if climate else f"No climate data for '{city}'."
