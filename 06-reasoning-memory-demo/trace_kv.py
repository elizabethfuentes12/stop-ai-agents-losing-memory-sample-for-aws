"""Decision traces over a key-value store (agent.state), the standalone track.

Most memory stores *what* the agent knows (demos 01-04 of this series). Almost nothing stores *why it
decided*: the question → tool steps → evidence → outcome chain. Without it, "why did you
recommend X?" a week later gets a confabulated answer, the agent invents a plausible
justification because the real one was never kept.

This module holds two things:

  1. ``DecisionTraceRecorder``, a Strands ``HookProvider`` that captures a decision
     trace automatically from the agent's lifecycle events. **Zero changes to your
     tools**: it subscribes to the hook events every tool call already emits.

  2. A flat trace store over ``agent.state`` plus the deterministic query functions the
     measurement uses. The flat store answers "why did I decide X?" perfectly well, the
     trace is right there as one blob. Where it falls short is the *reverse* question:
     "evidence source S turned out to be false, which of my decisions depended on it?"
     A flat scan finds decisions that cite S **directly**; a decision that depended on S
     *through another decision's output* doesn't mention S anywhere in its own blob. You
     can rebuild that chain in application code on every query, or store it as a graph
     and traverse it, that contrast is trace_graph.py, and it is the point of this demo.

Honesty note (stated in the README too): "reasoning memory" is an engineering pattern,
not an established category in the academic memory taxonomies. The closest verified
research is the traceability theme in recent memory systems:
  https://arxiv.org/abs/2601.18204 (MemWeaver, traceable long-horizon agentic reasoning)
  https://arxiv.org/abs/2606.09900 ("Less Context, More Accuracy", the Engram system;
      every stored fact keeps provenance and a supersession chain. Single-author preprint.)
"""

from strands import tool, ToolContext
from strands.hooks import (
    AfterInvocationEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    HookProvider,
    HookRegistry,
)

TRACES_KEY = "decision_traces"


# ── The recorder (a Strands HookProvider, no changes to any tool) ───────────
class DecisionTraceRecorder(HookProvider):
    """Records one decision trace per agent invocation into ``agent.state``.

    Attach with ``Agent(hooks=[DecisionTraceRecorder()])``. The recorder listens to:
      - ``BeforeInvocationEvent``, opens a trace with the user's question,
      - ``AfterToolCallEvent``   , appends a step (tool, input, evidence) per tool call,
      - ``AfterInvocationEvent`` , closes the trace with the final outcome and persists
        it under ``agent.state["decision_traces"]`` (JSON-serializable, so it survives
        with a session manager just like any other state).
    """

    def __init__(self):
        self._current = None

    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(BeforeInvocationEvent, self._on_start)
        registry.add_callback(AfterToolCallEvent, self._on_tool)
        registry.add_callback(AfterInvocationEvent, self._on_end)

    def _on_start(self, event: BeforeInvocationEvent) -> None:
        self._current = {"question": _last_user_text(event.messages), "steps": []}

    def _on_tool(self, event: AfterToolCallEvent) -> None:
        if self._current is None:
            return
        self._current["steps"].append({
            "n": len(self._current["steps"]) + 1,
            "tool": event.tool_use["name"],
            "input": event.tool_use.get("input") or {},
            "evidence": {
                "name": f"{event.tool_use['name']}-result",
                "text": _result_text(event.result),
                "source": event.tool_use["name"],
            },
        })

    def _on_end(self, event: AfterInvocationEvent) -> None:
        if self._current is None:
            return
        traces = event.agent.state.get(TRACES_KEY) or []
        self._current["id"] = f"trace-{len(traces) + 1}"
        self._current["outcome"] = str(event.result).strip()
        traces.append(self._current)
        event.agent.state.set(TRACES_KEY, traces)
        self._current = None


def _last_user_text(messages) -> str:
    for message in reversed(messages or []):
        if message.get("role") == "user":
            for block in message.get("content", []):
                if isinstance(block, dict) and "text" in block:
                    return block["text"]
    return ""


def _result_text(result) -> str:
    if not result:
        return ""
    texts = [block.get("text", "") for block in result.get("content", []) if isinstance(block, dict)]
    return " ".join(t for t in texts if t).strip()


# ── Seeded decision history (shared with the graph track) ────────────────────
# Five past decisions of a travel assistant, in the exact shape the recorder produces,
# plus explicit evidence provenance ("source"). Four depend on the reviews feed, two
# directly, two only through other decisions' outputs (a provenance chain one and two
# hops deep), and one (the control) does not depend on it at all. Deterministic, so
# the measurement is reproducible.

COMPROMISED_SOURCE = "fare_alerts_feed"

# External evidence sources (everything else in a "source" field names another
# evidence record, a derivation, not an origin).
EXTERNAL_SOURCES = {"flight_search", "fare_alerts_feed", "weather_api", "dining_guide"}

# Decisions whose evidence chain truly reaches fare_alerts_feed (ground truth for the
# check): two cite it directly, six only through other decisions' outputs, at depths
# from one to five hops, across two branches. The two controls depend on other sources.
AFFECTED_IDS = {"madrid-flight", "tokyo-flight", "trip-budget", "trip-itinerary",
                "trip-insurance", "calendar-block", "visa-check", "team-notify"}
CONTROL_IDS = {"packing-list", "restaurant-picks"}
CONTROL_ID = "packing-list"  # kept for backward compatibility with existing callers

SEED_TRACES = [
    {
        "id": "madrid-flight",
        "question": "Which flight should I book to Madrid?",
        "outcome": "Recommended the Iberia non-stop JFK-MAD.",
        "steps": [
            {"n": 1, "tool": "search_flights", "input": {"origin": "JFK", "destination": "MAD"},
             "evidence": {"name": "madrid-candidates",
                          "text": "Candidates JFK-MAD: Iberia non-stop, TAP one-stop via Lisbon.",
                          "source": "flight_search"}},
            {"n": 2, "tool": "check_fare_alert", "input": {"route": "JFK-MAD"},
             "evidence": {"name": "madrid-fare-alert",
                          "text": "Fare alert: Iberia JFK-MAD at $612 is 30% below typical.",
                          "source": "fare_alerts_feed"}},
        ],
    },
    {
        "id": "tokyo-flight",
        "question": "Which flight should I book to Tokyo?",
        "outcome": "Recommended the ANA non-stop JFK-HND.",
        "steps": [
            {"n": 1, "tool": "search_flights", "input": {"origin": "JFK", "destination": "HND"},
             "evidence": {"name": "tokyo-candidates",
                          "text": "Candidates JFK-HND: ANA non-stop, United one-stop via SFO.",
                          "source": "flight_search"}},
            {"n": 2, "tool": "check_fare_alert", "input": {"route": "JFK-HND"},
             "evidence": {"name": "tokyo-fare-alert",
                          "text": "Fare alert: ANA JFK-HND at $890 is 25% below typical.",
                          "source": "fare_alerts_feed"}},
        ],
    },
    {
        # The indirect dependency: this decision starts from the *outputs* of the two
        # flight decisions. Its own blob never mentions fare_alerts_feed, the link to
        # the compromised source exists only through the evidence chain.
        "id": "trip-budget",
        "question": "What flight budget do I need for the two-city trip?",
        "outcome": "Budget $1,502 total ($612 JFK-MAD on Iberia + $890 JFK-HND on ANA).",
        "steps": [
            {"n": 1, "tool": "sum_fares", "input": {"fare": "madrid"},
             "evidence": {"name": "madrid-fare",
                          "text": "The chosen Madrid flight: Iberia at $612.",
                          "source": "madrid-fare-alert"}},
            {"n": 2, "tool": "sum_fares", "input": {"fare": "tokyo"},
             "evidence": {"name": "tokyo-fare",
                          "text": "The chosen Tokyo flight: ANA at $890.",
                          "source": "tokyo-fare-alert"}},
        ],
    },
    {
        # Two hops from the source: builds on the budget decision, which itself built on
        # the flight decisions. Its blob is even further from any mention of the feed.
        "id": "trip-itinerary",
        "question": "Put together the final two-city itinerary.",
        "outcome": "Itinerary: Madrid first (Iberia), then Tokyo (ANA), within the $1,502 flight budget.",
        "steps": [
            {"n": 1, "tool": "compose_itinerary", "input": {"trip": "Madrid+Tokyo"},
             "evidence": {"name": "itinerary-draft",
                          "text": "Combined the approved $1,502 flight budget into a two-city sequence.",
                          "source": "madrid-fare"}},
        ],
    },
    {
        # The control: depends on the weather API, not on the fare-alerts feed. A correct
        # audit must NOT flag this one.
        "id": "packing-list",
        "question": "What should I pack for Madrid in October?",
        "outcome": "Pack light layers: mild days, cool evenings, little rain.",
        "steps": [
            {"n": 1, "tool": "check_weather", "input": {"city": "Madrid", "month": "October"},
             "evidence": {"name": "madrid-weather",
                          "text": "Madrid in October: 21C average highs, low rainfall.",
                          "source": "weather_api"}},
        ],
    },
    {
        # Three hops from the source: trip insurance is priced off the itinerary, which
        # was built on the budget, which was built on the flights. Nothing in this blob
        # mentions the feed; the link is three derivations deep.
        "id": "trip-insurance",
        "question": "How much trip insurance should I buy for the two-city trip?",
        "outcome": "Insurance $95, covering the $1,502 flight budget for the Madrid+Tokyo itinerary.",
        "steps": [
            {"n": 1, "tool": "price_insurance", "input": {"trip": "Madrid+Tokyo"},
             "evidence": {"name": "insurance-quote",
                          "text": "Insurance premium is 6.3% of the covered itinerary value.",
                          "source": "itinerary-draft"}},
        ],
    },
    {
        # Four hops from the source: the calendar block is placed from the insured
        # itinerary. This is the deepest dependency; a flat scan has no chance of
        # connecting it back to the fare-alerts feed.
        "id": "calendar-block",
        "question": "Block my calendar for the confirmed trip.",
        "outcome": "Blocked Oct 12-20 for the insured Madrid+Tokyo itinerary.",
        "steps": [
            {"n": 1, "tool": "block_calendar", "input": {"trip": "Madrid+Tokyo"},
             "evidence": {"name": "calendar-hold",
                          "text": "Held the dates for the insured two-city itinerary.",
                          "source": "insurance-quote"}},
        ],
    },
    {
        # A second control, on a different external source. Two controls make "correctly
        # not flagged" a stronger check: neither weather nor the restaurant guide feed
        # touches the fare-alerts feed.
        "id": "restaurant-picks",
        "question": "Where should I eat in Madrid?",
        "outcome": "Booked two tapas bars in La Latina from the local guide.",
        "steps": [
            {"n": 1, "tool": "find_restaurants", "input": {"city": "Madrid"},
             "evidence": {"name": "madrid-restaurants",
                          "text": "La Latina tapas bars, highly rated in the local dining guide.",
                          "source": "dining_guide"}},
        ],
    },
    {
        # Five hops from the source: the visa check reads the calendar hold to know the
        # travel dates. calendar-hold <- insurance-quote <- itinerary-draft <- madrid-fare
        # <- madrid-fare-alert <- fare_alerts_feed. The deepest dependency in the demo.
        "id": "visa-check",
        "question": "Do I need a visa for these travel dates?",
        "outcome": "No visa needed for the Oct 12-20 dates held for the trip.",
        "steps": [
            {"n": 1, "tool": "check_visa", "input": {"trip": "Madrid+Tokyo"},
             "evidence": {"name": "visa-note",
                          "text": "Checked visa rules against the held travel dates.",
                          "source": "calendar-hold"}},
        ],
    },
    {
        # A separate branch, two hops from the source via the budget: the team is
        # notified of the trip cost. Depends on the budget (which built on the flights),
        # not on the itinerary line, so it widens the affected set beyond the main chain.
        "id": "team-notify",
        "question": "Tell my team what this trip will cost.",
        "outcome": "Notified the team the flight budget is $1,502.",
        "steps": [
            {"n": 1, "tool": "notify_team", "input": {"amount": 1502},
             "evidence": {"name": "team-message",
                          "text": "Shared the approved $1,502 flight budget with the team.",
                          "source": "madrid-fare"}},
        ],
    },
]


# ── Deterministic queries over the flat store (the measurement functions) ────
def replay_why(traces: list, topic: str) -> dict | None:
    """Answer "why did I decide X?" from the flat store: find the trace whose outcome
    or question mentions the topic and return it whole. This is where a flat trace
    store shines, the full chain is one lookup away.
    """
    topic_lower = topic.lower()
    for trace in traces:
        if topic_lower in trace["outcome"].lower() or topic_lower in trace["question"].lower():
            return trace
    return None


def find_affected_decisions_kv(traces: list, source_name: str) -> list:
    """The reverse audit over the flat store: which decisions cite this source DIRECTLY?

    A linear scan of each trace's own evidence. This is the query a flat store naturally
    answers, and its honest limitation: a decision that depended on the source only
    through another decision's output has no mention of the source in its own blob, so
    the scan cannot see it. (Chasing ``source`` fields across blobs recursively would
    mean rebuilding the graph in application code on every query.)
    """
    affected = []
    for trace in traces:
        if any(step["evidence"]["source"] == source_name for step in trace["steps"]):
            affected.append(trace["id"])
    return affected


# ── Demo tools for the live agent (real Duffel offers, no hardcoded data) ────
@tool
def search_flights(origin: str, destination: str, departure_date: str,
                   cabin_class: str = "economy") -> str:
    """Search live flight offers when the user wants to fly somewhere on a date.

    Use this tool when the user:
    - asks for flights between two airports ("find me a flight to Madrid")
    - wants options or prices for a route on a date

    Args:
        origin: IATA airport code of departure, e.g. "JFK".
        destination: IATA airport code of arrival, e.g. "MAD".
        departure_date: ISO date, e.g. "2026-10-10".
        cabin_class: One of: economy, premium_economy, business, first.

    Returns:
        JSON list of real offers, cheapest first (offer_id, price, currency,
        cabin, slices with carrier/stops).
    """
    import flights_api
    offers = flights_api.search_offers(origin, destination, departure_date, cabin_class,
                                       max_results=4)
    import json as _json
    return _json.dumps(offers, indent=1)


@tool
def check_fare_alert(route: str) -> str:
    """Check whether a route currently has a below-typical fare alert.

    Use this tool after searching, when deciding WHICH offer to recommend -
    an active alert is evidence that a fare is unusually good.

    Args:
        route: Route as ORIGIN-DEST, e.g. "JFK-MAD".

    Returns:
        One line stating whether an alert is active for the route and why.
    """
    # Deterministic evidence source for the demo's audit trail: derived from the
    # captured real offers (fallback_offers.json), not invented at runtime.
    import flights_api, json as _json
    try:
        with open(flights_api._FALLBACK_FILE, encoding="utf-8") as f:
            captured = _json.load(f)
    except OSError:
        captured = {}
    offers = captured.get(route.upper().strip(), [])
    if not offers:
        return f"No fare data captured for {route}; no alert."
    cheapest = min(o["price"] for o in offers)
    median = sorted(o["price"] for o in offers)[len(offers) // 2]
    if cheapest < 0.8 * median:
        return f"Fare alert: {route} at ${cheapest:.0f} is well below the ${median:.0f} typical fare."
    return f"No alert: {route} cheapest ${cheapest:.0f} is near the ${median:.0f} typical fare."


# ── The Strands harness layer: the agent answers "why?" from its own traces ──
@tool(context=True)
def why_did_i(topic: str, tool_context: ToolContext) -> str:
    """Replay the decision trace behind a past decision, the real reasoning, not a guess.

    Use this whenever the user asks WHY a past recommendation or decision was made.
    Looks the decision up in the recorded traces and returns the step-by-step chain:
    question, each tool call with its evidence, and the outcome.

    Args:
        topic: A keyword from the decision to explain (e.g. an airline or city name).
    """
    traces = tool_context.agent.state.get(TRACES_KEY) or []
    trace = replay_why(traces, topic)
    if trace is None:
        return f"No decision trace found mentioning '{topic}'. I cannot explain that decision."
    lines = [f"Decision: {trace['question']}"]
    for step in trace["steps"]:
        lines.append(f"  step {step['n']}: {step['tool']}({step['input']}) -> {step['evidence']['text']}")
    lines.append(f"Outcome: {trace['outcome']}")
    return "\n".join(lines)
