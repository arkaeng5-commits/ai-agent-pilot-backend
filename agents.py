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

ALLOWED_AUDIENCES = {"finance", "rnd", "ceo", "user"}

VALID_STATUSES = {
    "pending",
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

    public_state: dict[str, Any]
    private_ceo_state: dict[str, Any]
    agent_inboxes: dict[str, list[dict[str, Any]]]

    topic_info: dict[str, Any]
    finance_result: dict[str, Any]
    rnd_result: dict[str, Any]
    ceo_decision: dict[str, Any]

    updated_public_state: dict[str, Any]
    updated_private_ceo_state: dict[str, Any]
    updated_agent_inboxes: dict[str, list[dict[str, Any]]]

    finance_reply: str
    rnd_reply: str
    ceo_public_reply: str
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
    metadata: dict[str, Any] | None = None,
) -> None:
    (
        db()
        .table("messages")
        .insert(
            {
                "project_id": project_id,
                "role": role,
                "text": text,
                "metadata": metadata or {},
            }
        )
        .execute()
    )


def ask_llm(
    system_prompt: str,
    context: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    response = groq_client().chat.completions.create(
        model=required_env("GROQ_MODEL"),
        temperature=0.2,
        max_completion_tokens=700,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context)},
        ],
    )

    content = (response.choices[0].message.content or "").strip()

    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]).strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        return fallback

    try:
        output = json.loads(content[start : end + 1])
        return output if isinstance(output, dict) else fallback
    except json.JSONDecodeError:
        return fallback


def safe_text(value: Any, max_length: int = 500) -> str:
    return str(value or "").strip()[:max_length]


def safe_audience(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    return [
        item
        for item in value
        if isinstance(item, str) and item in ALLOWED_AUDIENCES
    ]


def empty_inboxes() -> dict[str, list[dict[str, Any]]]:
    return {
        "finance": [],
        "rnd": [],
        "ceo": [],
    }


def load_project_state(project: dict[str, Any]) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
]:
    stored = project.get("assumptions") or {}

    if not isinstance(stored, dict):
        stored = {}

    # Converts old assumption formats into the new format safely.
    if "public" in stored:
        public_state = deepcopy(stored.get("public") or {})
    else:
        public_state = {
            "topic": safe_text(stored.get("topic", "New discussion")),
            "parameters": {},
            "status": stored.get("validation_status", "pending"),
        }

    private_ceo_state = deepcopy(stored.get("private_ceo") or {})

    saved_inboxes = stored.get("agent_inboxes") or {}
    agent_inboxes = empty_inboxes()

    for agent in agent_inboxes:
        inbox = saved_inboxes.get(agent, [])
        if isinstance(inbox, list):
            agent_inboxes[agent] = inbox[-20:]

    public_state.setdefault("topic", "New discussion")
    public_state.setdefault("parameters", {})
    public_state.setdefault("status", "pending")

    private_ceo_state.setdefault("parameters", {})

    return public_state, private_ceo_state, agent_inboxes


def visible_public_state(
    public_state: dict[str, Any],
    audience: str,
) -> dict[str, Any]:
    visible_parameters = {}

    for name, parameter in public_state.get("parameters", {}).items():
        shared_with = parameter.get("shared_with", [])

        if audience in shared_with:
            visible_parameters[name] = {
                "value": parameter.get("value"),
                "reason": parameter.get("reason"),
                "last_revised_by": parameter.get("last_revised_by"),
                "revision": parameter.get("revision"),
            }

    return {
        "topic": public_state.get("topic"),
        "status": public_state.get("status"),
        "parameters": visible_parameters,
    }


def topic_detector(state: PilotState) -> dict[str, Any]:
    logger.info("Running topic detector")

    result = ask_llm(
        """
You detect the topic of a discussion.

The subject may be anything: business, education, research, operations,
technology, policy, or another area.

Return ONLY JSON:

{
  "topic": "short topic title",
  "decision_needed": "what needs to be decided",
  "summary": "short context summary"
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
            "decision_needed": "Needs clarification",
            "summary": "",
        },
    )

    return {"topic_info": result}


def finance_agent(state: PilotState) -> dict[str, Any]:
    logger.info("Running Finance Agent")

    result = ask_llm(
        """
You are the Finance Agent.

Review only information that is visible to you. Never invent hidden facts.
Consider costs, value, resources, incentives, affordability, or commercial
impact only when relevant to the topic.

Write a real, helpful response addressed to the user.

Return ONLY JSON:

{
  "public_reply": "Your natural-language reply for the user.",
  "proposed_parameters": [
    {
      "name": "parameter name",
      "value": "proposed value",
      "reason": "why"
    }
  ],
  "risks": ["public risk"],
  "questions": ["public question"]
}
""",
        {
            "topic": state["topic_info"],
            "visible_project_state": visible_public_state(
                state["public_state"],
                "finance",
            ),
            "notifications_for_finance": state["agent_inboxes"].get(
                "finance",
                [],
            ),
            "conversation": state["history"],
        },
        {
            "public_reply": "Finance could not produce a structured response.",
            "proposed_parameters": [],
            "risks": [],
            "questions": [],
        },
    )

    inboxes = deepcopy(state["agent_inboxes"])
    inboxes["finance"] = []

    return {
        "finance_result": result,
        "finance_reply": safe_text(result.get("public_reply")),
        "agent_inboxes": inboxes,
    }


def rnd_agent(state: PilotState) -> dict[str, Any]:
    logger.info("Running R&D Agent")

    result = ask_llm(
        """
You are the R&D Agent.

Review only information visible to you. Consider feasibility, evidence,
technical or operational constraints, experimentation, and uncertainty only
when relevant to the topic.

Write a real, helpful response addressed to the user.

Return ONLY JSON:

{
  "public_reply": "Your natural-language reply for the user.",
  "proposed_parameters": [
    {
      "name": "parameter name",
      "value": "proposed value",
      "reason": "why"
    }
  ],
  "risks": ["public risk"],
  "questions": ["public question"]
}
""",
        {
            "topic": state["topic_info"],
            "visible_project_state": visible_public_state(
                state["public_state"],
                "rnd",
            ),
            "notifications_for_rnd": state["agent_inboxes"].get(
                "rnd",
                [],
            ),
            "finance_agent_public_reply": state["finance_reply"],
            "conversation": state["history"],
        },
        {
            "public_reply": "R&D could not produce a structured response.",
            "proposed_parameters": [],
            "risks": [],
            "questions": [],
        },
    )

    inboxes = deepcopy(state["agent_inboxes"])
    inboxes["rnd"] = []

    return {
        "rnd_result": result,
        "rnd_reply": safe_text(result.get("public_reply")),
        "agent_inboxes": inboxes,
    }


def ceo_decision_agent(state: PilotState) -> dict[str, Any]:
    logger.info("Running CEO private decision step")

    result = ask_llm(
        """
You are the CEO decision-maker.

You may see private CEO information. Never place private information into
approved_parameters or revisions unless you explicitly decide it should be
shared.

Finance and R&D proposals are suggestions only. You have the final authority
to approve, revise, or reject parameters.

If you revise a parameter, use the same exact parameter name. For example,
Finance can propose budget = $300 and you can revise budget = $250.

For every approved or revised parameter, decide who may see it:
finance, rnd, ceo, user.

Put CEO-only secrets in private_parameters. These are never sent to the user,
Finance, or R&D.

Return ONLY JSON:

{
  "approved_parameters": [
    {
      "name": "parameter",
      "value": "approved value",
      "reason": "reason",
      "share_with": ["finance", "rnd", "ceo", "user"]
    }
  ],
  "revisions": [
    {
      "name": "existing parameter",
      "value": "revised value",
      "reason": "reason",
      "share_with": ["finance", "rnd", "ceo", "user"]
    }
  ],
  "private_parameters": [
    {
      "name": "CEO-only parameter",
      "value": "secret value",
      "reason": "why it remains private"
    }
  ],
  "status": "needs_input, internally_consistent, needs_human_review, or ready_for_next_step"
}
""",
        {
            "topic": state["topic_info"],
            "ceo_visible_public_state": visible_public_state(
                state["public_state"],
                "ceo",
            ),
            "ceo_private_state": state["private_ceo_state"],
            "finance_proposal": state["finance_result"],
            "rnd_proposal": state["rnd_result"],
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
    logger.info("Applying CEO decisions")

    public_state = deepcopy(state["public_state"])
    private_state = deepcopy(state["private_ceo_state"])
    inboxes = deepcopy(state["agent_inboxes"])

    public_parameters = public_state.get("parameters", {})
    private_parameters = private_state.get("parameters", {})

    decision = state["ceo_decision"]

    # Approved parameters are applied first.
    # Revisions are applied second and therefore always win.
    directives = (
        decision.get("approved_parameters", [])
        + decision.get("revisions", [])
    )

    for raw_item in directives:
        item = normalise_parameter(raw_item)

        if not item:
            continue

        name = item["name"]
        audience = item["share_with"]

        # A parameter shared with nobody becomes CEO-private.
        if not audience or audience == ["ceo"]:
            private_parameters[name] = {
                "value": item["value"],
                "reason": item["reason"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            continue

        previous = public_parameters.get(name, {})
        revision_number = int(previous.get("revision", 0)) + 1

        public_parameters[name] = {
            "value": item["value"],
            "reason": item["reason"],
            "shared_with": audience,
            "last_revised_by": "ceo",
            "revision": revision_number,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Finance and R&D receive an internal notice on their next turn.
        for agent in ("finance", "rnd"):
            if agent in audience:
                inboxes[agent].append(
                    {
                        "type": "CEO parameter update",
                        "parameter": name,
                        "value": item["value"],
                        "reason": item["reason"],
                    }
                )

    # Explicit CEO secrets never become public parameters.
    for raw_item in decision.get("private_parameters", []):
        item = normalise_parameter(
            {
                **raw_item,
                "share_with": [],
            }
        )

        if item:
            private_parameters[item["name"]] = {
                "value": item["value"],
                "reason": item["reason"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

    status = safe_text(
        decision.get("status", "needs_human_review"),
        100,
    )

    if status not in VALID_STATUSES:
        status = "needs_human_review"

    public_state["topic"] = safe_text(
        state["topic_info"].get("topic", public_state.get("topic")),
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
        "updated_private_ceo_state": private_state,
        "updated_agent_inboxes": inboxes,
        "validation_status": status,
    }


def ceo_public_reply_agent(state: PilotState) -> dict[str, Any]:
    logger.info("Running CEO public reply step")

    # This LLM call never receives CEO-private state.
    # Therefore it cannot reveal private CEO parameters.
    result = ask_llm(
        """
You are the CEO communicating with the user.

Write a clear, helpful, natural-language reply based ONLY on the public
information supplied to you. Do not mention hidden or private information.

Explain only the parameters shared with the user. If a parameter is not shown
in the public project state, do not mention it.

Return ONLY JSON:

{
  "public_reply": "Your complete reply for the user."
}
""",
        {
            "topic": state["topic_info"],
            "public_project_state_visible_to_user": visible_public_state(
                state["updated_public_state"],
                "user",
            ),
            "finance_agent_public_reply": state["finance_reply"],
            "rnd_agent_public_reply": state["rnd_reply"],
            "status": state["validation_status"],
        },
        {
            "public_reply": "The CEO completed the review.",
        },
    )

    return {
        "ceo_public_reply": safe_text(
            result.get("public_reply"),
            3000,
        ),
    }


def save_project(state: PilotState) -> dict[str, Any]:
    stored_assumptions = {
        "public": state["updated_public_state"],
        "private_ceo": state["updated_private_ceo_state"],
        "agent_inboxes": state["updated_agent_inboxes"],
    }

    (
        db()
        .table("projects")
        .update(
            {
                "assumptions": stored_assumptions,
                "validation_status": state["validation_status"],
            }
        )
        .eq("id", state["project_id"])
        .execute()
    )

    logger.info("Project updated")
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
    public_state, private_ceo_state, inboxes = load_project_state(project)

    insert_message(
        project_id,
        role,
        text,
        {"agent": "user", "visibility": "public"},
    )

    history = get_recent_messages(project_id)

    final_state = PILOT_GRAPH.invoke(
        {
            "project_id": project_id,
            "project": project,
            "history": history,
            "public_state": public_state,
            "private_ceo_state": private_ceo_state,
            "agent_inboxes": inboxes,
        }
    )

    # These are real LLM-generated public replies.
    # No private CEO decision is inserted into messages.
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
            "text": final_state["ceo_public_reply"],
        },
    ]

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
        "ai_reply": final_state["ceo_public_reply"],
        "topic": final_state["updated_public_state"]["topic"],
        "assumptions": final_state["updated_public_state"],
        "validation_status": final_state["validation_status"],
        "agent_messages": agent_messages,
    }
