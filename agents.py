import json
import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from groq import Groq
from langgraph.graph import END, START, StateGraph
from supabase import create_client
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

VALID_STATUSES = {
    "pending",
    "needs_input",
    "internally_consistent",
    "needs_human_review",
    "ready_for_next_step",
}

SHAREABLE_AUDIENCES = {
    "finance",
    "rnd",
    "ceo",
    "user",
}


class ProjectNotFound(Exception):
    pass


class PilotState(TypedDict, total=False):
    project_id: str
    project: dict[str, Any]
    history: list[dict[str, Any]]

    public_state: dict[str, Any]
    ceo_private_state: dict[str, Any]
    agent_inboxes: dict[str, list[dict[str, Any]]]

    topic_info: dict[str, Any]
    finance_result: dict[str, Any]
    rnd_result: dict[str, Any]
    ceo_decision: dict[str, Any]

    finance_reply: str
    rnd_reply: str
    ceo_reply: str

    updated_public_state: dict[str, Any]
    updated_ceo_private_state: dict[str, Any]
    updated_agent_inboxes: dict[str, list[dict[str, Any]]]
    validation_status: str


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
        .select("role,text,metadata,created_at")
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )

    return list(reversed(result.data or []))


def insert_message(
    project_id: str,
    role: str,
    text: str,
    metadata: dict[str, Any],
) -> None:
    (
        db()
        .table("messages")
        .insert(
            {
                "project_id": project_id,
                "role": role,
                "text": text,
                "metadata": metadata,
            }
        )
        .execute()
    )


def safe_text(value: Any, limit: int = 1000) -> str:
    return str(value or "").strip()[:limit]


def safe_list(value: Any, limit: int = 10) -> list[str]:
    if not isinstance(value, list):
        return []

    result = []

    for item in value:
        text = safe_text(item, 500)

        if text:
            result.append(text)

    return result[:limit]


def safe_audience(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        item
        for item in value
        if isinstance(item, str) and item in SHAREABLE_AUDIENCES
    ]


def ask_llm(
    system_prompt: str,
    context: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    response = groq_client().chat.completions.create(
        model=required_env("GROQ_MODEL"),
        temperature=0.2,
        max_completion_tokens=700,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(context),
            },
        ],
    )

    content = (response.choices[0].message.content or "").strip()

    try:
        result = json.loads(content)

        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        logger.warning("LLM did not return valid JSON: %s", content[:300])

    return fallback


def empty_inboxes() -> dict[str, list[dict[str, Any]]]:
    return {
        "finance": [],
        "rnd": [],
    }


def load_project_state(
    project: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    stored = project.get("assumptions") or {}

    if not isinstance(stored, dict):
        stored = {}

    # Supports both the new format and your earlier format.
    if isinstance(stored.get("public"), dict):
        public_state = deepcopy(stored["public"])
    else:
        public_state = {
            "topic": safe_text(stored.get("topic", "New discussion"), 200),
            "parameters": {},
            "status": project.get("validation_status", "pending"),
        }

    ceo_private_state = deepcopy(stored.get("private_ceo") or {})

    public_state.setdefault("topic", "New discussion")
    public_state.setdefault("parameters", {})
    public_state.setdefault("status", "pending")

    ceo_private_state.setdefault("parameters", {})

    inboxes = empty_inboxes()
    saved_inboxes = stored.get("agent_inboxes") or {}

    if isinstance(saved_inboxes, dict):
        for agent in inboxes:
            messages = saved_inboxes.get(agent)

            if isinstance(messages, list):
                inboxes[agent] = messages[-20:]

    return public_state, ceo_private_state, inboxes


def visible_public_state(
    public_state: dict[str, Any],
    audience: str,
) -> dict[str, Any]:
    visible_parameters = {}

    for name, parameter in public_state.get("parameters", {}).items():
        if not isinstance(parameter, dict):
            continue

        if audience not in parameter.get("shared_with", []):
            continue

        visible_parameters[name] = {
            "value": parameter.get("value"),
            "reason": parameter.get("reason"),
            "last_revised_by": parameter.get("last_revised_by"),
            "revision": parameter.get("revision"),
        }

    return {
        "topic": public_state.get("topic"),
        "decision_needed": public_state.get("decision_needed", ""),
        "status": public_state.get("status", "pending"),
        "parameters": visible_parameters,
    }


def topic_detector(state: PilotState) -> dict[str, Any]:
    logger.info("Running Topic Detector")

    result = ask_llm(
        """
You identify the topic and decision being discussed.

The topic can be anything. Do not assume it is a business, finance,
technology, or budget discussion.

Return JSON only:

{
  "topic": "short title",
  "decision_needed": "what should be decided or clarified"
}
""",
        {
            "conversation": state["history"],
            "public_project_state": visible_public_state(
                state["public_state"],
                "user",
            ),
        },
        {
            "topic": "New discussion",
            "decision_needed": "More information is needed.",
        },
    )

    return {"topic_info": result}


def finance_agent(state: PilotState) -> dict[str, Any]:
    logger.info("Running Finance Agent")

    result = ask_llm(
        """
You are the Finance Agent.

You can see only the supplied public project information. Do not invent
private facts or assume every topic needs a budget.

Write a direct, useful, natural-language reply for the user. Discuss cost,
value, incentives, affordability, or resources only when relevant.

You may propose parameters, but the CEO has final authority. Your proposals
are not final decisions.

Return JSON only:

{
  "public_reply": "Your real response to the user.",
  "proposed_parameters": [
    {
      "name": "parameter name",
      "value": "suggested value",
      "reason": "why"
    }
  ]
}
""",
        {
            "topic": state["topic_info"],
            "public_project_state": visible_public_state(
                state["public_state"],
                "finance",
            ),
            "ceo_notifications": state["agent_inboxes"].get("finance", []),
            "conversation": state["history"],
        },
        {
            "public_reply": "Finance could not complete its review.",
            "proposed_parameters": [],
        },
    )

    return {
        "finance_result": result,
        "finance_reply": safe_text(
            result.get("public_reply"),
            3000,
        ),
    }


def rnd_agent(state: PilotState) -> dict[str, Any]:
    logger.info("Running R&D Agent")

    result = ask_llm(
        """
You are the R&D Agent.

You can see only the supplied public project information. Do not invent
private facts or assume every topic is technical.

Write a direct, useful, natural-language reply for the user. Discuss
feasibility, evidence, experimentation, technical or operational constraints,
and uncertainty only when relevant.

You may propose parameters, but the CEO has final authority. Your proposals
are not final decisions.

Return JSON only:

{
  "public_reply": "Your real response to the user.",
  "proposed_parameters": [
    {
      "name": "parameter name",
      "value": "suggested value",
      "reason": "why"
    }
  ]
}
""",
        {
            "topic": state["topic_info"],
            "public_project_state": visible_public_state(
                state["public_state"],
                "rnd",
            ),
            "ceo_notifications": state["agent_inboxes"].get("rnd", []),
            "finance_agent_reply": state["finance_reply"],
            "conversation": state["history"],
        },
        {
            "public_reply": "R&D could not complete its review.",
            "proposed_parameters": [],
        },
    )

    return {
        "rnd_result": result,
        "rnd_reply": safe_text(
            result.get("public_reply"),
            3000,
        ),
    }


def ceo_decision_agent(state: PilotState) -> dict[str, Any]:
    logger.info("Running CEO decision agent")

    result = ask_llm(
        """
You are the CEO and have final decision authority.

Finance and R&D proposals are suggestions. You decide whether to approve,
revise, or reject them.

A revision replaces an existing parameter with the same name. Example:
Finance proposes "budget = $300"; CEO revises "budget = $250".
Your revised value is the final value.

For every public parameter, choose exactly who may see it:
finance, rnd, ceo, user.

Use private_parameters for CEO-only secrets. These private parameters must
never appear in approved_parameters or revisions. Do not describe private
parameters in a public reply.

Return JSON only:

{
  "approved_parameters": [
    {
      "name": "parameter",
      "value": "final value",
      "reason": "reason",
      "share_with": ["finance", "rnd", "ceo", "user"]
    }
  ],
  "revisions": [
    {
      "name": "existing parameter",
      "value": "revised final value",
      "reason": "reason",
      "share_with": ["finance", "rnd", "ceo", "user"]
    }
  ],
  "private_parameters": [
    {
      "name": "CEO-only parameter",
      "value": "secret value",
      "reason": "why it remains CEO-only"
    }
  ],
  "status": "needs_input, internally_consistent, needs_human_review, or ready_for_next_step"
}
""",
        {
            "topic": state["topic_info"],
            "public_project_state_visible_to_ceo": visible_public_state(
                state["public_state"],
                "ceo",
            ),
            "ceo_private_state": state["ceo_private_state"],
            "finance_proposals": state["finance_result"],
            "rnd_proposals": state["rnd_result"],
        },
        {
            "approved_parameters": [],
            "revisions": [],
            "private_parameters": [],
            "status": "needs_human_review",
        },
    )

    return {"ceo_decision": result}


def normalise_parameter(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    name = safe_text(item.get("name"), 100).lower()
    value = safe_text(item.get("value"), 500)
    reason = safe_text(item.get("reason"), 500)
    share_with = safe_audience(item.get("share_with"))

    if not name or not value:
        return None

    return {
        "name": name,
        "value": value,
        "reason": reason,
        "share_with": share_with,
    }


def apply_ceo_decisions(state: PilotState) -> dict[str, Any]:
    logger.info("Applying CEO parameter decisions")

    public_state = deepcopy(state["public_state"])
    private_state = deepcopy(state["ceo_private_state"])
    inboxes = deepcopy(state["agent_inboxes"])

    public_parameters = public_state.get("parameters", {})
    private_parameters = private_state.get("parameters", {})

    decision = state["ceo_decision"]

    # Revisions are placed after approvals. Therefore, if both use the same
    # parameter name, the CEO revision always becomes the saved final value.
    directives = (
        decision.get("approved_parameters", [])
        + decision.get("revisions", [])
    )

    for raw_parameter in directives:
        parameter = normalise_parameter(raw_parameter)

        if not parameter:
            continue

        name = parameter["name"]
        share_with = parameter["share_with"]

        # CEO-only parameter: stored privately and never sent to other agents.
        if not share_with or share_with == ["ceo"]:
            private_parameters[name] = {
                "value": parameter["value"],
                "reason": parameter["reason"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            continue

        old_parameter = public_parameters.get(name, {})
        revision = int(old_parameter.get("revision", 0)) + 1

        public_parameters[name] = {
            "value": parameter["value"],
            "reason": parameter["reason"],
            "shared_with": share_with,
            "last_revised_by": "CEO",
            "revision": revision,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # A section receives the CEO update only if CEO chose to share it.
        # It will see this notification during its next agent turn.
        for agent in ("finance", "rnd"):
            if agent in share_with:
                inboxes[agent].append(
                    {
                        "type": "CEO parameter update",
                        "parameter": name,
                        "value": parameter["value"],
                        "reason": parameter["reason"],
                    }
                )

    # Explicit secrets always remain separate from public state.
    for raw_parameter in decision.get("private_parameters", []):
        parameter = normalise_parameter(
            {
                **raw_parameter,
                "share_with": [],
            }
        )

        if parameter:
            private_parameters[parameter["name"]] = {
                "value": parameter["value"],
                "reason": parameter["reason"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    status = safe_text(
        decision.get("status", "needs_human_review"),
        100,
    )

    if status not in VALID_STATUSES:
        status = "needs_human_review"

    public_state["topic"] = safe_text(
        state["topic_info"].get("topic", public_state["topic"]),
        200,
    )
    public_state["decision_needed"] = safe_text(
        state["topic_info"].get("decision_needed"),
        500,
    )
    public_state["parameters"] = public_parameters
    public_state["status"] = status
    public_state["updated_at"] = datetime.now(timezone.utc).isoformat()

    private_state["parameters"] = private_parameters

    return {
        "updated_public_state": public_state,
        "updated_ceo_private_state": private_state,
        "updated_agent_inboxes": inboxes,
        "validation_status": status,
    }


def ceo_public_reply_agent(state: PilotState) -> dict[str, Any]:
    logger.info("Running CEO public reply agent")

    # This call intentionally receives no CEO-private state.
    # It cannot reveal a CEO-private parameter it was never given.
    result = ask_llm(
        """
You are the CEO speaking to the user.

Write a clear, direct natural-language response. Use ONLY the supplied public
project state. Never mention a parameter that is not visible to the user.

Explain relevant approved or revised public parameters. Do not mention that
private information exists.

Return JSON only:

{
  "public_reply": "Your complete response to the user."
}
""",
        {
            "topic": state["topic_info"],
            "user_visible_project_state": visible_public_state(
                state["updated_public_state"],
                "user",
            ),
            "finance_agent_reply": state["finance_reply"],
            "rnd_agent_reply": state["rnd_reply"],
            "status": state["validation_status"],
        },
        {
            "public_reply": "The CEO completed the review.",
        },
    )

    return {
        "ceo_reply": safe_text(
            result.get("public_reply"),
            3000,
        )
    }


def save_project(state: PilotState) -> dict[str, Any]:
    assumptions = {
        "public": state["updated_public_state"],
        "private_ceo": state["updated_ceo_private_state"],
        "agent_inboxes": state["updated_agent_inboxes"],
    }

    (
        db()
        .table("projects")
        .update(
            {
                "assumptions": assumptions,
                "validation_status": state["validation_status"],
            }
        )
        .eq("id", state["project_id"])
        .execute()
    )

    logger.info("Project updated successfully")
    return {}


def build_graph():
    graph = StateGraph(PilotState)

    graph.add_node("topic_detector", topic_detector)
    graph.add_node("finance_agent", finance_agent)
    graph.add_node("rnd_agent", rnd_agent)
    graph.add_node("ceo_decision_agent", ceo_decision_agent)
    graph.add_node("apply_ceo_decisions", apply_ceo_decisions)
    graph.add_node("ceo_public_reply_agent", ceo_public_reply_agent)
    graph.add_node("save_project", save_project)

    graph.add_edge(START, "topic_detector")
    graph.add_edge("topic_detector", "finance_agent")
    graph.add_edge("finance_agent", "rnd_agent")
    graph.add_edge("rnd_agent", "ceo_decision_agent")
    graph.add_edge("ceo_decision_agent", "apply_ceo_decisions")
    graph.add_edge("apply_ceo_decisions", "ceo_public_reply_agent")
    graph.add_edge("ceo_public_reply_agent", "save_project")
    graph.add_edge("save_project", END)

    return graph.compile()


PILOT_GRAPH = build_graph()


def run_pilot(project_id: str, role: str, text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Message text cannot be empty.")

    project = get_project(project_id)

    public_state, ceo_private_state, inboxes = load_project_state(project)

    insert_message(
        project_id,
        role,
        text,
        {
            "agent": "User",
            "visibility": "public",
        },
    )

    history = get_recent_messages(project_id)

    final_state = PILOT_GRAPH.invoke(
        {
            "project_id": project_id,
            "project": project,
            "history": history,
            "public_state": public_state,
            "ceo_private_state": ceo_private_state,
            "agent_inboxes": inboxes,
        }
    )

    agent_messages = [
        {
            "agent": "Finance Agent",
            "text": final_state["finance_reply"],
        },
        {
            "agent": "R&D Agent",
            "text": final_state["rnd_reply"],
        },
        {
            "agent": "CEO Agent",
            "text": final_state["ceo_reply"],
        },
    ]

    # Save only public LLM replies. CEO secrets are never saved in messages.
    for message in agent_messages:
        insert_message(
            project_id,
            "assistant",
            message["text"],
            {
                "agent": message["agent"],
                "visibility": "public",
            },
        )

    return {
        "ai_reply": final_state["ceo_reply"],
        "topic": final_state["updated_public_state"]["topic"],
        "assumptions": final_state["updated_public_state"],
        "validation_status": final_state["validation_status"],
        "agent_messages": agent_messages,
    }
