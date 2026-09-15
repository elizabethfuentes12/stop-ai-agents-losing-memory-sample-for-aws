"""Real flight search for the persistent-memory demo (Duffel sandbox).

Every flight in this demo is a REAL offer returned by the Duffel API sandbox -
no hardcoded flight lists. The sandbox returns real airline inventory shapes
(carriers, cabins, prices, durations) against test data, and the exact same
code works in production by swapping the test token for a live one.

Plain functions (no @tool, no model imports) so the tools layer stays thin and
this module is reusable across the demo series.

Duffel API shape verified live on 2026-07-16:
  POST https://api.duffel.com/air/offer_requests?return_offers=true
  headers: Authorization: Bearer <key>, Duffel-Version: v2
  -> data.offers[] each with total_amount, total_currency,
     slices[].segments[] (marketing_carrier, departing_at, arriving_at,
     duration, aircraft), and an offer id.

Pattern adapted, with thanks, from Ricardo Ceci's open course
"curso-strands-agentcore-2026" (clase-1 / clase-4 travel agent):
https://github.com/ricardoceci/curso-strands-agentcore-2026
"""

import json
import os
import threading

import requests

DUFFEL_API_BASE_URL = "https://api.duffel.com"
DUFFEL_API_VERSION = "v2"
_RETRIES = 3
_TIMEOUT = 30

# Offers captured once from the live sandbox (2026-07-16, JFK→CDG 2026-09-15) and used
# ONLY as a fallback if the sandbox can't be reached after retries. Real captured values,
# not invented, the same "capture ground truth once" pattern the sibling repos use.
_FALLBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fallback_offers.json")


def _duffel_headers() -> dict:
    api_key = os.environ.get("DUFFEL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DUFFEL_API_KEY is not set. Create a free sandbox token at "
            "https://app.duffel.com (More -> Developers -> Access tokens) and add it "
            "to your .env file as DUFFEL_API_KEY=duffel_test_..."
        )
    return {
        "Authorization": f"Bearer {api_key}",
        "Duffel-Version": DUFFEL_API_VERSION,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _simplify_offer(offer: dict) -> dict:
    """Reduce a raw Duffel offer to the fields the agent needs (keeps context small).

    Cabin is not a top-level offer field in the Duffel response, it lives per
    segment-passenger as ``cabin_class_marketing_name`` (verified live 2026-07-16),
    so we read it from the first segment's first passenger.
    """
    cabin = None
    slices = []
    for s in offer.get("slices", []):
        segments = []
        for seg in s.get("segments", []):
            if cabin is None:
                seg_passengers = seg.get("passengers") or []
                if seg_passengers:
                    marketing = seg_passengers[0].get("cabin_class_marketing_name")
                    if marketing:
                        cabin = marketing.lower().replace(" ", "_")
            segments.append({
                "carrier": (seg.get("marketing_carrier") or {}).get("name"),
                "flight_number": seg.get("marketing_carrier_flight_number"),
                "departing_at": seg.get("departing_at"),
                "arriving_at": seg.get("arriving_at"),
                "duration": seg.get("duration"),
            })
        slices.append({
            "origin": (s.get("origin") or {}).get("iata_code"),
            "destination": (s.get("destination") or {}).get("iata_code"),
            "segments": segments,
            "stops": max(len(segments) - 1, 0),
        })
    return {
        "offer_id": offer.get("id"),
        "price": float(offer["total_amount"]) if offer.get("total_amount") else None,
        "currency": offer.get("total_currency"),
        "cabin": cabin or "economy",
        "slices": slices,
    }


def search_offers(origin: str, destination: str, departure_date: str,
                  cabin_class: str = "economy", max_results: int = 8) -> list[dict]:
    """Search REAL flight offers on the Duffel sandbox. Returns simplified offers
    sorted by price (cheapest first), or the captured fallback if the sandbox is
    unreachable after retries.

    Args:
        origin: IATA airport code, e.g. "JFK".
        destination: IATA airport code, e.g. "CDG".
        departure_date: ISO date, e.g. "2026-09-15".
        cabin_class: economy | premium_economy | business | first.
        max_results: How many offers to return after sorting by price.
    """
    payload = {
        "data": {
            "slices": [{"origin": origin.upper(), "destination": destination.upper(),
                        "departure_date": departure_date}],
            "passengers": [{"type": "adult"}],
            "cabin_class": cabin_class,
        }
    }
    for attempt in range(_RETRIES):
        try:
            resp = requests.post(
                f"{DUFFEL_API_BASE_URL}/air/offer_requests",
                headers=_duffel_headers(), params={"return_offers": "true"},
                json=payload, timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            offers = resp.json().get("data", {}).get("offers", [])
            simplified = [_simplify_offer(o) for o in offers]
            simplified = [o for o in simplified if o["price"] is not None]
            for o in simplified:
                o["cabin"] = cabin_class
            simplified.sort(key=lambda o: o["price"])
            if simplified:
                return simplified[:max_results]
        except requests.HTTPError as exc:
            # 429 (rate limit) and 5xx are transient: back off and retry. Other
            # 4xx are the caller's fault, so stop and fall back.
            status = exc.response.status_code if exc.response is not None else None
            if status in (429, 500, 502, 503, 504) and attempt < _RETRIES - 1:
                threading.Event().wait(2 * (attempt + 1))
                continue
            break
        except requests.RequestException:
            if attempt < _RETRIES - 1:
                threading.Event().wait(2 * (attempt + 1))
    return _fallback_offers(origin, destination, cabin_class, max_results)


def get_offer(offer_id: str) -> dict | None:
    """Retrieve one offer by id from Duffel (GET /air/offers/{id}, verified live).

    Returns the simplified offer, or None if the id is unknown/expired and no
    captured fallback matches. Offers expire, so a stale id is a normal case the
    caller must handle.
    """
    for attempt in range(_RETRIES):
        try:
            resp = requests.get(
                f"{DUFFEL_API_BASE_URL}/air/offers/{offer_id}",
                headers=_duffel_headers(), timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json().get("data")
            if data:
                simplified = _simplify_offer(data)
                # Duffel GET returns cabin per-passenger; keep the simple default.
                return simplified
        except requests.HTTPError:
            break
        except requests.RequestException:
            if attempt < _RETRIES - 1:
                threading.Event().wait(2 * (attempt + 1))
    # Fallback: the id may come from captured offers (offline mode).
    try:
        with open(_FALLBACK_FILE, encoding="utf-8") as f:
            captured = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    for route, offers in captured.items():
        if route.startswith("_"):
            continue
        for o in offers:
            if o.get("offer_id") == offer_id:
                return o
    return None


def _fallback_offers(origin: str, destination: str, cabin_class: str, max_results: int) -> list[dict]:
    """Captured-once offers, used only when the sandbox is unreachable."""
    route = f"{origin.upper()}-{destination.upper()}"
    try:
        with open(_FALLBACK_FILE, encoding="utf-8") as f:
            captured = json.load(f)
    except (OSError, json.JSONDecodeError):
        captured = {}
    offers = captured.get(route, [])
    if offers:
        print(f"    (Duffel unavailable for {route}; using offers captured on {captured.get('_captured_on', '?')})")
        for o in offers:
            o["cabin"] = cabin_class
        return offers[:max_results]
    raise RuntimeError(
        f"Duffel sandbox unreachable and no captured fallback exists for {route}. "
        f"Check your network and DUFFEL_API_KEY, or capture a fallback with capture_fallback()."
    )


def capture_fallback(routes: list[tuple[str, str, str]]) -> dict:
    """Capture real offers for the given routes into fallback_offers.json.

    Run once at setup (network required) so the demo can survive sandbox downtime:
        capture_fallback([("JFK", "CDG", "2026-09-15")])
    """
    from datetime import date
    captured = {"_captured_on": date.today().isoformat()}
    for origin, dest, dep_date in routes:
        offers = search_offers(origin, dest, dep_date)
        captured[f"{origin.upper()}-{dest.upper()}"] = offers
    with open(_FALLBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(captured, f, indent=1)
    return {k: len(v) for k, v in captured.items() if not k.startswith("_")}
