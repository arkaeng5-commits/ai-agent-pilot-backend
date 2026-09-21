from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, TypedDict

from groq import Groq
from langgraph.graph import END, START, StateGraph
from supabase import Client, create_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supabase + Groq clients
# ---------------------------------------------------------------------------
_supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_ROLE_KEY"],
)
_groq = Groq(api_key=os.environ["GROQ_API_KEY"])
_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

VALID_STATUSES = {
    "pending",
    "needs_input",
    "internally_consistent",
    "needs_human_review",
    "ready_for_next_step",
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class PilotState(TypedDict):
    project_id: str
    role: str
    user_text: str
    topic_info: dict[str, Any]
    topic: str
    finance_advice: dict[str, Any]
    rnd_advice: dict[str, Any]
    ceo_advice: dict[str, Any]
    assumptions: dict[str, Any]
    validation_status: str
    ai_reply: str


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class ProjectNotFound(Exception):
    pass


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def get_project(project_id: str) -> dict[str, Any]:
    row = (
        _supabase.table("projects")
        .select("id, name, assumptions, validation_status")
        .eq("id", project_id)
        .maybe_single()
        .execute()
    )
    if row.data is None:
        raise ProjectNotFound(f"Project {project_id} not found")
    return row.data


def get_recent_messages(project_id: str, limit: int = 20) -> list[dict[str, Any]]:
    rows = (
        _supabase.table("messages")
        .select("role, text, created_at")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(rows.data or []))


def insert_message(project_id: str, role: str, text: str) -> None:
    _supabase.table("messages").insert(
        {"project_id": project_id, "role": role, "text": text}
    ).execute()


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def parse_json(raw: str, context: str = "") -> dict[str, Any]:
    text = raw.strip()
    # strip markdown code fences
    if text.startswith("
```"):
lines = text.splitlines()
text = "\n".join(
l for l in lines if not l.strip().startswith("
```")
        ).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # best-effort: grab content between first { and last }
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    logger.warning("parse_json failed (%s): %s", context, text[:200])
    return {}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def ask_llm(system: str, user: str) -> str:
    resp = _groq.chat.completions.create(
        model=_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_completion_tokens=600,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Node: topic_detector
# ---------------------------------------------------------------------------
def topic_detector(state: PilotState) -> dict[str, Any]:
    system = (
        "You are a topic classification assistant. "
        "Analyze the user's message and return ONLY valid JSON with keys: "
        "topic (short label), domain (one of: business, research, education, "
        "operations, software, health, workflow, or another suitable label), "
        "decision_needed (bool), context_summary (one sentence)."
    )
    user = f"User message:\n{state['user_text']}"
    raw = ask_llm(system, user)
    info = parse_json(raw, "topic_detector")
    topic = info.get("topic") or "General"
    return {"topic_info": info, "topic": topic}


# ---------------------------------------------------------------------------
# Agent prompt factory
# ---------------------------------------------------------------------------
def agent_prompt(agent_name: str, focus: str) -> str:
    return (
        f"You are the {agent_name} in a multi-agent review system. "
        f"Your focus: {focus}. "
        "The conversation may concern ANY subject. "
        "Do not force financial, technical, timeline, or resource assumptions "
        "when they do not apply. "
        "Return ONLY valid JSON:\n"
        '{"summary": "...", '
        '"assumptions": [{"name": "...", "value": "...", "confidence": "low|medium|high"}], '
        '"risks": ["..."], '
        '"open_questions": ["..."], '
        '"recommended_actions": ["..."]}\n'
        "Return empty arrays when a category does not apply."
    )


# ---------------------------------------------------------------------------
# Nodes: finance, rnd, ceo
# ---------------------------------------------------------------------------
def finance_agent(state: PilotState) -> dict[str, Any]:
    history = get_recent_messages(state["project_id"])
    history_text = "\n".join(f"{m['role']}: {m['text']}" for m in history)

    system = agent_prompt(
        "Finance Analyst",
        "cost, value, affordability, incentives, commercial impact, "
        "and resource implications only when relevant",
    )
    user = (
        f"Topic: {state['topic']}\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Current message: {state['user_text']}"
    )
    raw = ask_llm(system, user)
    advice = parse_json(raw, "finance_agent")
    insert_message(state["project_id"], "finance", advice.get("summary", raw[:500]))
    return {"finance_advice": advice}


def rnd_agent(state: PilotState) -> dict[str, Any]:
    finance_summary = state.get("finance_advice", {}).get("summary", "")
    history = get_recent_messages(state["project_id"])
    history_text = "\n".join(f"{m['role']}: {m['text']}" for m in history)

    system = agent_prompt(
        "R&D Analyst",
        "feasibility, evidence quality, technical or operational constraints, "
        "experimentation, and uncertainty",
    )
    user = (
        f"Topic: {state['topic']}\n\n"
        f"Finance perspective: {finance_summary}\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Current message: {state['user_text']}"
    )
    raw = ask_llm(system, user)
    advice = parse_json(raw, "rnd_agent")
    insert_message(state["project_id"], "rnd", advice.get("summary", raw[:500]))
    return {"rnd_advice": advice}


def ceo_agent(state: PilotState) -> dict[str, Any]:
    finance_summary = state.get("finance_advice", {}).get("summary", "")
    rnd_summary = state.get("rnd_advice", {}).get("summary", "")
    history = get_recent_messages(state["project_id"])
    history_text = "\n".join(f"{m['role']}: {m['text']}" for m in history)

    system = (
        "You are the CEO in a multi-agent review system. "
        "Synthesize all perspectives and decide whether information is sufficient "
        "to take a next step, or whether human review or more input is needed. "
        "Return ONLY valid JSON:\n"
        '{"summary": "...", '
        '"assumptions": [{"name": "...", "value": "...", "confidence": "low|medium|high"}], '
        '"risks": ["..."], '
        '"open_questions": ["..."], '
        '"recommended_actions": ["..."], '
        '"status_recommendation": "pending|needs_input|internally_consistent|needs_human_review|ready_for_next_step"}'
    )
    user = (
        f"Topic: {state['topic']}\n\n"
        f"Finance: {finance_summary}\n"
        f"R&D: {rnd_summary}\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Current message: {state['user_text']}"
    )
    raw = ask_llm(system, user)
    advice = parse_json(raw, "ceo_agent")
    insert_message(state["project_id"], "ceo", advice.get("summary", raw[:500]))
    return {"ceo_advice": advice}


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------
def _text_list(val: Any, max_len: int = 500) -> list[str]:
    if not isinstance(val, list):
        return []
    return [str(v)[:max_len] for v in val if v]


def _unique(items: list[str], cap: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= cap:
            break
    return out


def _normalise_assumption(a: Any) -> dict[str, str] | None:
    if not isinstance(a, dict):
        return None
    name = str(a.get("name", ""))[:200].strip()
    value = str(a.get("value", ""))[:500].strip()
    confidence = a.get("confidence", "medium")
    if confidence not in ("low", "medium", "high"):
        confidence = "medium"
    if not name:
        return None
    return {"name": name, "value": value, "confidence": confidence}


def merge_assumptions(
    *sources: list[Any],
) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for source in sources:
        for raw in source:
            norm = _normalise_assumption(raw)
            if norm:
                merged[norm["name"].lower()] = norm
    result = list(merged.values())[:12]
    return result


# ---------------------------------------------------------------------------
# Node: aggregator
# ---------------------------------------------------------------------------
def aggregator(state: PilotState) -> dict[str, Any]:
    f = state.get("finance_advice", {})
    r = state.get("rnd_advice", {})
    c = state.get("ceo_advice", {})

    assumptions = merge_assumptions(
        f.get("assumptions", []),
        r.get("assumptions", []),
        c.get("assumptions", []),
    )
    risks = _unique(
        _text_list(f.get("risks")) +
        _text_list(r.get("risks")) +
        _text_list(c.get("risks"))
    )
    open_questions = _unique(
        _text_list(f.get("open_questions")) +
        _text_list(r.get("open_questions")) +
        _text_list(c.get("open_questions"))
    )
    actions = _unique(
        _text_list(f.get("recommended_actions")) +
        _text_list(r.get("recommended_actions")) +
        _text_list(c.get("recommended_actions"))
    )[:3]

    raw_status = c.get("status_recommendation", "needs_human_review")
    status = raw_status if raw_status in VALID_STATUSES else "needs_human_review"
    if not assumptions and open_questions:
        status = "needs_input"

    summary = c.get("summary") or r.get("summary") or f.get("summary") or ""

    full_assumptions: dict[str, Any] = {
        "topic": state.get("topic", ""),
        "domain": state.get("topic_info", {}).get("domain", ""),
        "decision_needed": state.get("topic_info", {}).get("decision_needed", False),
        "summary": summary,
        "assumptions": assumptions,
        "risks": risks,
        "open_questions": open_questions,
        "recommended_actions": actions,
        "last_reviewed_at": datetime.now(timezone.utc).isoformat(),
    }

    actions_text = "\n".join(f"- {a}" for a in actions) if actions else "- No actions identified."
    ai_reply = (
        f"Topic: {state.get('topic', 'General')}\n\n"
        f"{summary}\n\n"
        f"Review status: {status}.\n\n"
        f"Next actions:\n{actions_text}"
    )

    insert_message(state["project_id"], "assistant", ai_reply)

    return {
        "assumptions": full_assumptions,
        "validation_status": status,
        "ai_reply": ai_reply,
    }


# ---------------------------------------------------------------------------
# Node: save_project
# ---------------------------------------------------------------------------
def save_project(state: PilotState) -> dict[str, Any]:
    _supabase.table("projects").update(
        {
            "assumptions": state["assumptions"],
            "validation_status": state["validation_status"],
        }
    ).eq("id", state["project_id"]).execute()
    return {}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph() -> Any:
    g = StateGraph(PilotState)
    g.add_node("topic_detector", topic_detector)
    g.add_node("finance_agent", finance_agent)
    g.add_node("rnd_agent", rnd_agent)
    g.add_node("ceo_agent", ceo_agent)
    g.add_node("aggregator", aggregator)
    g.add_node("save_project", save_project)

    g.add_edge(START, "topic_detector")
    g.add_edge("topic_detector", "finance_agent")
    g.add_edge("finance_agent", "rnd_agent")
    g.add_edge("rnd_agent", "ceo_agent")
    g.add_edge("ceo_agent", "aggregator")
    g.add_edge("aggregator", "save_project")
    g.add_edge("save_project", END)
    return g.compile()


PILOT_GRAPH = build_graph()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_pilot(project_id: str, role: str, text: str) -> dict[str, Any]:
    get_project(project_id)  # raises ProjectNotFound early if missing

    initial: PilotState = {
        "project_id": project_id,
        "role": role,
        "user_text": text,
        "topic_info": {},
        "topic": "",
        "finance_advice": {},
        "rnd_advice": {},
        "ceo_advice": {},
        "assumptions": {},
        "validation_status": "pending",
        "ai_reply": "",
    }

    insert_message(project_id, role, text)
    result = PILOT_GRAPH.invoke(initial)

    return {
        "ai_reply": result["ai_reply"],
        "topic": result["topic"],
        "assumptions": result["assumptions"],
        "validation_status": result["validation_status"],
    }
