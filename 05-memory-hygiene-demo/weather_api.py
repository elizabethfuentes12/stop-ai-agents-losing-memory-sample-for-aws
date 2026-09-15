"""Real climate data for "when should I visit X?" (Open-Meteo, free, no API key).

Answers the traveler question "what's the best time of year to visit Tokyo?" with
REAL historical climate data, no hardcoded weather tables. Monthly averages are
computed from the Open-Meteo historical archive (ERA5 reanalysis), aggregating the
last N full years of daily observations.

Endpoints verified live on 2026-07-16:
  GET https://geocoding-api.open-meteo.com/v1/search?name=Tokyo&count=1
    -> results[] with name, latitude, longitude, timezone, country
  GET https://archive-api.open-meteo.com/v1/archive?latitude=..&longitude=..
      &start_date=..&end_date=..&daily=temperature_2m_max,temperature_2m_min,
      precipitation_sum&timezone=<tz>
    -> daily.time[] + parallel arrays per variable

Free tier is for non-commercial use (this public demo qualifies) and requires
attribution: weather data by Open-Meteo (CC-BY 4.0), geocoding by GeoNames.
"""

import threading
from datetime import date

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_RETRIES = 3
_TIMEOUT = 30

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


def _get_json(url: str, params: dict) -> dict | None:
    for attempt in range(_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError:
            return None
        except requests.RequestException:
            if attempt < _RETRIES - 1:
                threading.Event().wait(2 * (attempt + 1))
    return None


def geocode(city: str) -> dict | None:
    """Resolve a city name to coordinates. Returns {name, latitude, longitude,
    timezone, country} or None if not found / unreachable."""
    data = _get_json(GEOCODING_URL, {"name": city, "count": 1, "language": "en", "format": "json"})
    results = (data or {}).get("results") or []
    if not results:
        return None
    top = results[0]
    return {k: top.get(k) for k in ("name", "latitude", "longitude", "timezone", "country")}


def monthly_climate(city: str, years: int = 5) -> dict | None:
    """Average climate by calendar month for a city, from real daily history.

    Aggregates the last ``years`` full calendar years of ERA5 daily data into
    12 rows: average daily max/min temperature (°C) and average total monthly
    precipitation (mm). Returns None if the city or the archive is unavailable.
    """
    place = geocode(city)
    if place is None:
        return None

    last_full_year = date.today().year - 1
    start = date(last_full_year - years + 1, 1, 1)
    end = date(last_full_year, 12, 31)

    data = _get_json(ARCHIVE_URL, {
        "latitude": place["latitude"], "longitude": place["longitude"],
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": place["timezone"] or "UTC",
    })
    daily = (data or {}).get("daily") or {}
    days = daily.get("time") or []
    if not days:
        return None

    tmax, tmin, prcp = daily["temperature_2m_max"], daily["temperature_2m_min"], daily["precipitation_sum"]
    # month -> accumulators over all days of all years
    acc = {m: {"tmax": [], "tmin": [], "precip_by_year": {}} for m in range(1, 13)}
    for i, day in enumerate(days):
        year, month = int(day[:4]), int(day[5:7])
        if tmax[i] is not None:
            acc[month]["tmax"].append(tmax[i])
        if tmin[i] is not None:
            acc[month]["tmin"].append(tmin[i])
        if prcp[i] is not None:
            acc[month]["precip_by_year"][year] = acc[month]["precip_by_year"].get(year, 0.0) + prcp[i]

    months = []
    for m in range(1, 13):
        a = acc[m]
        monthly_totals = list(a["precip_by_year"].values())
        months.append({
            "month": MONTH_NAMES[m - 1],
            "avg_high_c": round(sum(a["tmax"]) / len(a["tmax"]), 1) if a["tmax"] else None,
            "avg_low_c": round(sum(a["tmin"]) / len(a["tmin"]), 1) if a["tmin"] else None,
            "avg_precip_mm": round(sum(monthly_totals) / len(monthly_totals), 0) if monthly_totals else None,
        })

    return {
        "city": place["name"], "country": place["country"],
        "years_averaged": f"{start.year}-{end.year}",
        "months": months,
        "source": "Open-Meteo ERA5 archive (CC-BY 4.0), geocoding by GeoNames",
    }
