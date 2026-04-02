import sys, os, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Tools"))

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
Each step MUST follow this EXACT schema:
  {"step": int, "description": str, "tool": str or null, "args": dict or null}

Available tools and their EXACT argument names:
  - calculator(expression: str)               -> evaluate a math expression
  - search_web(query: str)                    -> search the web for information
  - search_news(query: str)                   -> search for latest news
  - get_current_weather(city: str)            -> get current weather for a city
  - get_weather_forecast(city: str, days: int)-> get weather forecast

Use null for tool/args on reasoning or synthesis steps.
Return ONLY a valid JSON array. No markdown, no explanation."""

TOOL_ARG_MAP = {
    "calculator": "expression",
    "search_web": "query",
    "search_news": "query",
    "get_current_weather": "city",
    "get_weather_forecast": "city",
}


class AgentState(TypedDict):
    goal: str
    plan: List[dict]
    current_step: int
    results: List[dict]


def _safe_args(tool_name: str, raw_args: dict) -> dict:
    expected = TOOL_ARG_MAP.get(tool_name)
    if not expected or expected in raw_args:
        return raw_args
    first_val = next(iter(raw_args.values()), tool_name)
    return {expected: str(first_val)}


def _call_tool(tool_name: str, args: dict) -> str:
    try:
        if tool_name == "calculator":
            import math_server as ms
            return str(ms.calculator(**args))
        elif tool_name == "search_web":
            import search_server as ss
            return str(ss.search_web(**args))
        elif tool_name == "search_news":
            import search_server as ss
            return str(ss.search_news(**args))
        elif tool_name == "get_current_weather":
            import weather_server as ws
            return str(ws.get_current_weather(**args))
        elif tool_name == "get_weather_forecast":
            import weather_server as ws
            return str(ws.get_weather_forecast(**args))
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Tool error ({tool_name}): {e}"


def planner_node(state: AgentState) -> AgentState:
    response = llm.invoke([
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=state["goal"])
    ])
    raw = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
    clean = re.sub(r"```json|```", "", raw).strip()
    plan = json.loads(clean)
    return {"goal": state["goal"], "plan": plan, "current_step": 0, "results": []}


def executor_node(state: AgentState) -> AgentState:
    idx = state["current_step"]
    step = state["plan"][idx]
    tool_name = step.get("tool")

    if tool_name:
        corrected = _safe_args(tool_name, step.get("args") or {})
        result = _call_tool(tool_name, corrected)
    else:
        context = "\n".join([f"Step {r['step']}: {r['result']}" for r in state["results"]])
        prompt = f"{step['description']}\n\nContext:\n{context}" if context else step["description"]
        resp = llm.invoke([HumanMessage(content=prompt)])
        result = resp.content if isinstance(resp.content, str) else resp.content[0].get("text", "")

    new_results = state["results"] + [{"step": step["step"], "description": step["description"], "result": str(result)}]
    return {**state, "current_step": idx + 1, "results": new_results}


def should_continue(state: AgentState) -> str:
    if state["current_step"] >= len(state["plan"]):
        return "end"
    return "executor"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges("executor", should_continue, {"executor": "executor", "end": END})
    return builder.compile()
