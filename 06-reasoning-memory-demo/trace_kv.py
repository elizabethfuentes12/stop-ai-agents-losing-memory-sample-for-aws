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


# ── Deterministic query over the flat store (the measurement function) ───────
def replay_why(traces: list, topic: str) -> dict | None:
    """Answer "why did I decide X?" from the flat store: find the recorded trace whose
    outcome or question mentions the topic and return it whole. This is where a flat
    trace store shines, the full chain is one lookup away, no model call.
    """
    topic_lower = topic.lower()
    for trace in traces:
        if topic_lower in trace["outcome"].lower() or topic_lower in trace["question"].lower():
            return trace
    return None


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


@tool
def check_weather(city: str) -> str:
    """Get typical monthly climate for a city, from real historical data (Open-Meteo).

    Use this when the user asks what to pack or what the weather is like.

    Args:
        city: City name, e.g. "Madrid".
    """
    import weather_api
    climate = weather_api.monthly_climate(city)
    if not climate:
        return f"No climate data available for {city}."
    return f"{city} climate (source: {climate['source']}): {climate['months'][:3]}..."


@tool
def best_time_to_visit(city: str) -> str:
    """Recommend the best months to visit a city, from real historical climate.

    Use this when the user asks when to travel somewhere.

    Args:
        city: City name, e.g. "Tokyo".
    """
    import weather_api
    climate = weather_api.monthly_climate(city)
    if not climate:
        return f"No climate data available for {city}."
    # Prefer months with mild highs and low rain, from the real monthly averages.
    months = [m for m in climate["months"] if m.get("avg_high_c") is not None]
    mild = sorted(months, key=lambda m: (abs((m["avg_high_c"] or 0) - 22), m.get("avg_precip_mm") or 0))[:3]
    names = ", ".join(m["month"] for m in mild)
    return f"Best time to visit {city}: {names} (mild temperatures, from {climate['source']})."


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
