import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from groq import Groq
from langgraph.graph import END, START, StateGraph
from supabase import create_client
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

VALID_STATUSES = {
    "needs_input",
    "internally_consistent",
    "needs_human_review",
    "ready_for_next_step",
}


class ProjectNotFound(Exception):
    pass


class PilotState(TypedDict, total=False):
    project_id: str
    project: dict[str, Any]
    history: list[dict[str, Any]]
    topic_info: dict[str, Any]
    finance_advice: dict[str, Any]
    rnd_advice: dict[str, Any]
    ceo_advice: dict[str, Any]
    assumptions: dict[str, Any]
    validation_status: str
    ai_reply: str
    topic: str


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing Render environment variable: {name}")
    return value


@lru_cache
def db():
    return create_client(
        required_env("SUPABASE_URL"),
        required_env("SUPABASE_SERVICE_ROLE_KEY"),
    )


@lru_cache
def groq_client():
    return Groq(api_key=required_env("GROQ_API_KEY"))


def get_project(project_id: str) -> dict[str, Any]:
    result = (
        db()
        .table("projects")
        .select("*")
        .eq("id", project_id)
        .execute()
    )

    rows = result.data or []
    if not rows:
        raise ProjectNotFound()

    return rows[0]


def get_recent_messages(project_id: str) -> list[dict[str, Any]]:
    result = (
        db()
        .table("messages")
        .select("role,text,created_at")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )

    return list(reversed(result.data or []))


def insert_message(project_id: str, role: str, text: str) -> None:
    (
        db()
        .table("messages")
        .insert(
            {
                "project_id": project_id,
                "role": role,
                "text": text,
            }
        )
        .execute()
    )


def parse_json(content: str, fallback: dict[str, Any]) -> dict[str, Any]:
    content = (content or "").strip()

    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]).strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        return fallback

    try:
        result = json.loads(content[start : end + 1])
        return result if isinstance(result, dict) else fallback
    except json.JSONDecodeError:
        return fallback


def ask_llm(system_prompt: str, context: dict[str, Any]) -> dict[str, Any]:
    response = groq_client().chat.completions.create(
        model=required_env("GROQ_MODEL"),
        temperature=0.2,
        max_completion_tokens=600,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context)},
        ],
    )

    return parse_json(
        response.choices[0].message.content or "",
        {
            "summary": "The model did not return usable structured output.",
            "assumptions": [],
            "risks": [],
            "open_questions": [],
            "recommended_actions": [],
        },
    )


def topic_detector(state: PilotState) -> dict[str, Any]:
    logger.info("Running topic_detector for project %s", state["project_id"])

    prompt = """
You are the topic-detection step of a company decision-support system.

Read the conversation. Detect its actual topic, even if it is unrelated to
business, budgets, or software. Identify the type of decision or problem being
discussed. Do not assume a fixed domain.

Return ONLY valid JSON:

{
  "topic": "short description of the topic",
  "domain": "business, research, education, operations, software, health workflow, or another suitable label",
  "decision_needed": "what the user appears to need decided or clarified",
  "context_summary": "one short summary"
}
"""

    info = ask_llm(
        prompt,
        {
            "conversation": state["history"],
            "existing_project_data": state["project"].get("assumptions", {}),
        },
    )

    return {"topic_info": info}


def agent_prompt(agent_name: str, responsibility: str) -> str:
    return f"""
You are the {agent_name} in a multi-agent review system.

Your role:
{responsibility}

The conversation may concern ANY subject. First understand the detected topic.
Do not force financial, technical, timeline, or resource assumptions where they
are not relevant.

You may identify:
- assumptions that should be made explicit;
- contradictions or risks;
- information that is missing;
- practical next actions.

Return ONLY valid JSON:

{{
  "summary": "one or two short sentences",
  "assumptions": [
    {{
      "name": "the assumption",
      "value": "the assumed value or statement",
      "confidence": "low, medium, or high"
    }}
  ],
  "risks": ["risk or concern"],
  "open_questions": ["question needing an answer"],
  "recommended_actions": ["practical next step"]
}}

Return empty arrays when a category does not apply.
"""


def finance_agent(state: PilotState) -> dict[str, Any]:
    logger.info("Running finance_agent for project %s", state["project_id"])

    advice = ask_llm(
        agent_prompt(
            "Finance Agent",
            "Review cost, value, affordability, incentives, commercial impact, "
            "and resource implications only when they are relevant to the topic.",
        ),
        {
            "topic_info": state["topic_info"],
            "conversation": state["history"],
            "previous_advice": {},
        },
    )

    return {"finance_advice": advice}


def rnd_agent(state: PilotState) -> dict[str, Any]:
    logger.info("Running rnd_agent for project %s", state["project_id"])

    advice = ask_llm(
        agent_prompt(
            "R&D Agent",
            "Review feasibility, evidence quality, technical or operational "
            "constraints, experimentation, and uncertainty when relevant.",
        ),
        {
            "topic_info": state["topic_info"],
            "conversation": state["history"],
            "previous_advice": {
                "finance": state.get("finance_advice", {}),
            },
        },
    )

    return {"rnd_advice": advice}


def ceo_agent(state: PilotState) -> dict[str, Any]:
    logger.info("Running ceo_agent for project %s", state["project_id"])

    prompt = agent_prompt(
        "CEO Agent",
        "Synthesize the other perspectives into a clear decision-oriented "
        "recommendation. Decide whether the information is sufficient to take "
        "a next step, or whether human review or more input is needed.",
    ) + """

Also include one extra field:

"status_recommendation": one of:
- needs_input
- internally_consistent
- needs_human_review
- ready_for_next_step
"""

    advice = ask_llm(
        prompt,
        {
            "topic_info": state["topic_info"],
            "conversation": state["history"],
            "previous_advice": {
                "finance": state.get("finance_advice", {}),
                "rnd": state.get("rnd_advice", {}),
            },
        },
    )

    return {"ceo_advice": advice}


def text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        str(item).strip()[:500]
        for item in value
        if isinstance(item, (str, int, float)) and str(item).strip()
    ]


def unique(items: list[str], limit: int = 12) -> list[str]:
    result = []
    seen = set()

    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)

    return result[:limit]


def normalise_assumption(item: Any) -> dict[str, str] | None:
    if not isinstance(item, dict):
        return None

    name = str(item.get("name", "")).strip()[:200]
    value = str(item.get("value", "")).strip()[:500]
    confidence = str(item.get("confidence", "medium")).lower().strip()

    if not name or not value:
        return None

    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"

    return {
        "name": name,
        "value": value,
        "confidence": confidence,
    }


def merge_assumptions(advice_items: list[dict[str, Any]]) -> list[dict[str, str]]:
    merged = {}

    for advice in advice_items:
        for item in advice.get("assumptions", []):
            clean = normalise_assumption(item)
            if clean:
                # Later agents, especially CEO, can refine earlier assumptions.
                merged[clean["name"].lower()] = clean

    return list(merged.values())[:12]


def aggregator(state: PilotState) -> dict[str, Any]:
    logger.info("Running aggregator for project %s", state["project_id"])

    finance = state.get("finance_advice", {})
    rnd = state.get("rnd_advice", {})
    ceo = state.get("ceo_advice", {})
    topic_info = state.get("topic_info", {})

    topic = str(topic_info.get("topic", "General discussion")).strip()[:200]
    domain = str(topic_info.get("domain", "general")).strip()[:100]

    assumptions = merge_assumptions([finance, rnd, ceo])

    risks = unique(
        text_list(finance.get("risks"))
        + text_list(rnd.get("risks"))
        + text_list(ceo.get("risks"))
    )

    open_questions = unique(
        text_list(finance.get("open_questions"))
        + text_list(rnd.get("open_questions"))
        + text_list(ceo.get("open_questions"))
    )

    actions = unique(
        text_list(finance.get("recommended_actions"))
        + text_list(rnd.get("recommended_actions"))
        + text_list(ceo.get("recommended_actions"))
    )

    status = str(
        ceo.get("status_recommendation", "needs_human_review")
    ).strip()

    if status not in VALID_STATUSES:
        status = "needs_human_review"

    if not assumptions and open_questions:
        status = "needs_input"

    new_assumptions = {
        "topic": topic,
        "domain": domain,
        "decision_needed": str(
            topic_info.get("decision_needed", "")
        ).strip()[:500],
        "summary": str(ceo.get("summary", "")).strip()[:1000],
        "assumptions": assumptions,
        "risks": risks,
        "open_questions": open_questions,
        "recommended_actions": actions,
        "last_reviewed_at": datetime.now(timezone.utc).isoformat(),
    }

    summary = new_assumptions["summary"] or (
        "The leadership team reviewed the available information."
    )

    actions_text = (
        "\n\nNext actions:\n- " + "\n- ".join(actions[:3])
        if actions
        else ""
    )

    ai_reply = (
        f"Topic: {topic}\n\n"
        f"{summary}\n\n"
        f"Review status: {status}."
        f"{actions_text}"
    )

    return {
        "topic": topic,
        "assumptions": new_assumptions,
        "validation_status": status,
        "ai_reply": ai_reply,
    }


def save_project(state: PilotState) -> dict[str, Any]:
    (
        db()
        .table("projects")
        .update(
            {
                "assumptions": state["assumptions"],
                "validation_status": state["validation_status"],
            }
        )
        .eq("id", state["project_id"])
        .execute()
    )

    logger.info("Project %s updated", state["project_id"])
    return {}


def build_graph():
    graph = StateGraph(PilotState)

    graph.add_node("topic_detector", topic_detector)
    graph.add_node("finance_agent", finance_agent)
    graph.add_node("rnd_agent", rnd_agent)
    graph.add_node("ceo_agent", ceo_agent)
    graph.add_node("aggregator", aggregator)
    graph.add_node("save_project", save_project)

    graph.add_edge(START, "topic_detector")
    graph.add_edge("topic_detector", "finance_agent")
    graph.add_edge("finance_agent", "rnd_agent")
    graph.add_edge("rnd_agent", "ceo_agent")
    graph.add_edge("ceo_agent", "aggregator")
    graph.add_edge("aggregator", "save_project")
    graph.add_edge("save_project", END)

    return graph.compile()


PILOT_GRAPH = build_graph()


def run_pilot(project_id: str, role: str, text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Message text cannot be empty.")

    project = get_project(project_id)

    # Save the user's message, then give the full recent conversation to agents.
    insert_message(project_id, role, text)
    history = get_recent_messages(project_id)

    final_state = PILOT_GRAPH.invoke(
        {
            "project_id": project_id,
            "project": project,
            "history": history,
        }
    )

    ai_reply = final_state.get("ai_reply", "")
    if not ai_reply:
        raise RuntimeError("The graph returned no AI reply.")

    insert_message(project_id, "assistant", ai_reply)

    return {
        "ai_reply": ai_reply,
        "topic": final_state["topic"],
        "assumptions": final_state["assumptions"],
        "validation_status": final_state["validation_status"],
    }
