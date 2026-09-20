import json
import logging
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from groq import Groq
from langgraph.graph import END, START, StateGraph
from supabase import create_client
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

OPEN_PROPOSAL_STATUSES = {
    "awaiting_ceo",
    "matches_ceo",
    "conflicts_with_ceo",
}


class ProjectNotFound(Exception):
    pass


class EmployeeState(TypedDict, total=False):
    project_id: str
    section_tag: str
    text: str
    message_id: str
    history: list[dict[str, Any]]
    visible_parameters: list[dict[str, Any]]
    employee_result: dict[str, Any]
    reply: str


class CEOState(TypedDict, total=False):
    project_id: str
    section_tag: str
    text: str
    message_id: str
    history: list[dict[str, Any]]
    public_parameters: list[dict[str, Any]]
    secret_parameters: list[dict[str, Any]]
    open_proposals: list[dict[str, Any]]
    ceo_result: dict[str, Any]
    reply: str
    decisions_saved: int


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_text(value: Any, limit: int = 1000) -> str:
    return str(value or "").strip()[:limit]


def normalise_key(value: Any) -> str:
    text = safe_text(value, 100).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80]


def clean_json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
        return value

    return safe_text(value, 500)


def values_match(first: Any, second: Any) -> bool:
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def normalise_sections(value: Any) -> list[str]:
    raw_sections = value if isinstance(value, list) else []

    sections = ["ceo"]

    for item in raw_sections:
        section = normalise_key(item)

        if section and section not in sections:
            sections.append(section)

    return sections


def ask_json(
    system_prompt: str,
    context: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    response = groq_client().chat.completions.create(
        model=required_env("GROQ_MODEL"),
        temperature=0.2,
        max_completion_tokens=900,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(context)},
        ],
    )

    content = (response.choices[0].message.content or "").strip()

    try:
        parsed = json.loads(content)

        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        logger.warning("Invalid JSON from Groq: %s", content[:300])

    return fallback


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


def insert_message(
    project_id: str,
    section_tag: str,
    actor_type: str,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    result = (
        db()
        .table("messages")
        .insert(
            {
                "project_id": project_id,
                "role": actor_type,
                "section_tag": section_tag,
                "actor_type": actor_type,
                "text": text,
                "metadata": metadata,
            }
        )
        .execute()
    )

    rows = result.data or []

    if not rows:
        raise RuntimeError("Supabase did not return the saved message.")

    return rows[0]


def get_section_messages(
    project_id: str,
    section_tag: str,
) -> list[dict[str, Any]]:
    result = (
        db()
        .table("messages")
        .select(
            "id,section_tag,actor_type,text,metadata,created_at"
        )
        .eq("project_id", project_id)
        .eq("section_tag", section_tag)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )

    return list(reversed(result.data or []))


def get_public_parameters(project_id: str) -> list[dict[str, Any]]:
    result = (
        db()
        .table("ceo_parameters")
        .select(
            "id,parameter_key,parameter_label,value_json,"
            "value_text,revision_number,updated_at"
        )
        .eq("project_id", project_id)
        .order("parameter_key")
        .execute()
    )

    return [
        {
            **row,
            "visibility": "public",
        }
        for row in (result.data or [])
    ]


def get_all_secret_parameters(project_id: str) -> list[dict[str, Any]]:
    result = (
        db()
        .table("ceo_secret_parameters")
        .select(
            "id,parameter_key,parameter_label,value_json,"
            "value_text,allowed_sections,revision_number,updated_at"
        )
        .eq("project_id", project_id)
        .order("parameter_key")
        .execute()
    )

    return [
        {
            **row,
            "visibility": "secret",
        }
        for row in (result.data or [])
    ]


def get_visible_parameters(
    project_id: str,
    section_tag: str,
) -> list[dict[str, Any]]:
    parameters = get_public_parameters(project_id)
    secret_parameters = get_all_secret_parameters(project_id)

    for parameter in secret_parameters:
        allowed_sections = parameter.get("allowed_sections") or []

        if section_tag in allowed_sections:
            parameters.append(parameter)

    return parameters


def get_open_proposals(project_id: str) -> list[dict[str, Any]]:
    result = (
        db()
        .table("parameter_proposals")
        .select(
            "id,section_tag,parameter_key,parameter_label,"
            "value_json,value_text,status,created_at"
        )
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .limit(100)
        .execute()
    )

    return [
        proposal
        for proposal in (result.data or [])
        if proposal.get("status") in OPEN_PROPOSAL_STATUSES
    ]


def parameter_map(
    parameters: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        parameter["parameter_key"]: parameter
        for parameter in parameters
    }


def normalise_proposal(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    label = safe_text(raw.get("name"), 100)
    key = normalise_key(raw.get("key") or label)
    value = clean_json_value(raw.get("value"))
    value_text = safe_text(raw.get("value_text") or value, 500)

    if not key or not value_text:
        return None

    return {
        "parameter_key": key,
        "parameter_label": label or key.replace("_", " ").title(),
        "value_json": value,
        "value_text": value_text,
    }


def employee_analyze(state: EmployeeState) -> dict[str, Any]:
    logger.info("Analyzing employee message from %s", state["section_tag"])

    result = ask_json(
        """
You are an AI assistant for one employee section of a project.

Write a direct and helpful reply for the employee. The reply is displayed
exactly to the user. Do not put hidden reasoning, technical analysis, or JSON
inside the reply.

Extract every meaningful parameter the employee states or proposes. A parameter
may be a cost, date, policy, quantity, location, priority, target, rule,
resource, requirement, or another named value.

All official CEO parameters supplied to you are binding. If the employee asks
about or proposes a parameter already defined by CEO, clearly state the
current CEO value. The employee can make a suggestion, but cannot overwrite
the CEO value.

Return JSON only:

{
  "reply": "Natural-language response for the employee.",
  "parameters": [
    {
      "name": "Human-readable parameter name",
      "key": "normalised_parameter_key",
      "value": "Proposed value",
      "value_text": "Readable proposed value"
    }
  ]
}
""",
        {
            "section": state["section_tag"],
            "section_chat_history": state["history"],
            "current_message": state["text"],
            "official_parameters_visible_to_this_section": (
                state["visible_parameters"]
            ),
        },
        {
            "reply": "I could not analyse this message.",
            "parameters": [],
        },
    )

    return {
        "employee_result": result,
        "reply": safe_text(result.get("reply"), 3500),
    }


def store_employee_proposals(state: EmployeeState) -> dict[str, Any]:
    official = parameter_map(state["visible_parameters"])

    raw_parameters = state["employee_result"].get("parameters", [])

    if not isinstance(raw_parameters, list):
        return {}

    for raw_parameter in raw_parameters:
        proposal = normalise_proposal(raw_parameter)

        if not proposal:
            continue

        current_value = official.get(proposal["parameter_key"])

        if not current_value:
            status = "awaiting_ceo"
        elif values_match(
            proposal["value_json"],
            current_value["value_json"],
        ):
            status = "matches_ceo"
        else:
            status = "conflicts_with_ceo"

        (
            db()
            .table("parameter_proposals")
            .insert(
                {
                    "project_id": state["project_id"],
                    "section_tag": state["section_tag"],
                    "parameter_key": proposal["parameter_key"],
                    "parameter_label": proposal["parameter_label"],
                    "value_json": proposal["value_json"],
                    "value_text": proposal["value_text"],
                    "source_message_id": state["message_id"],
                    "status": status,
                }
            )
            .execute()
        )

    return {}


def ceo_analyze(state: CEOState) -> dict[str, Any]:
    logger.info("Analyzing CEO message")

    result = ask_json(
        """
You are an AI assistant helping a human CEO.

The CEO is the only decision-maker. Never make a decision yourself.

Employee proposals are shown with their exact section tags. You can explain and
compare them for the CEO.

Create a public_decision only when the CEO explicitly sets, revises, approves,
or adopts a value in the CEO's current message.

Create a secret_decision only when the CEO explicitly says that a value is
secret, confidential, restricted, or should be shared only with named sections.

If the CEO asks a question, considers options, or is unclear, return empty
decision lists. Never infer a decision.

For a public adoption, use the exact proposal_id shown in the proposals list.

Return JSON only:

{
  "reply": "Natural-language reply for the CEO.",
  "public_decisions": [
    {
      "decision_type": "set or adopt",
      "parameter_key": "normalised_parameter_key",
      "parameter_label": "Readable name",
      "value": "CEO value for set only",
      "value_text": "Readable value for set only",
      "proposal_id": "Proposal UUID for adopt only"
    }
  ],
  "secret_decisions": [
    {
      "parameter_key": "normalised_parameter_key",
      "parameter_label": "Readable secret parameter name",
      "value": "CEO secret value",
      "value_text": "Readable secret value",
      "allowed_sections": ["finance", "operations"]
    }
  ]
}
""",
        {
            "ceo_chat_history": state["history"],
            "current_ceo_message": state["text"],
            "public_ceo_parameters": state["public_parameters"],
            "secret_ceo_parameters": state["secret_parameters"],
            "employee_proposals": state["open_proposals"],
        },
        {
            "reply": "I could not analyse the CEO message.",
            "public_decisions": [],
            "secret_decisions": [],
        },
    )

    return {
        "ceo_result": result,
        "reply": safe_text(result.get("reply"), 3500),
    }


def update_proposal_statuses(
    project_id: str,
    parameter_key: str,
    adopted_proposal_id: str | None,
) -> None:
    result = (
        db()
        .table("parameter_proposals")
        .select("id,status")
        .eq("project_id", project_id)
        .eq("parameter_key", parameter_key)
        .execute()
    )

    for proposal in result.data or []:
        if proposal["status"] not in OPEN_PROPOSAL_STATUSES:
            continue

        status = (
            "adopted_by_ceo"
            if proposal["id"] == adopted_proposal_id
            else "superseded_by_ceo"
        )

        (
            db()
            .table("parameter_proposals")
            .update(
                {
                    "status": status,
                    "reviewed_at": now_iso(),
                }
            )
            .eq("id", proposal["id"])
            .execute()
        )


def save_public_parameter(
    project_id: str,
    ceo_message_id: str,
    parameter_key: str,
    parameter_label: str,
    value_json: Any,
    value_text: str,
    decision_type: str,
    adopted_proposal_id: str | None,
    public_map: dict[str, dict[str, Any]],
) -> None:
    previous = public_map.get(parameter_key)
    revision_number = (
        int(previous["revision_number"]) + 1
        if previous
        else 1
    )

    (
        db()
        .table("ceo_parameters")
        .upsert(
            {
                "project_id": project_id,
                "parameter_key": parameter_key,
                "parameter_label": parameter_label,
                "value_json": value_json,
                "value_text": value_text,
                "revision_number": revision_number,
                "ceo_message_id": ceo_message_id,
            },
            on_conflict="project_id,parameter_key",
        )
        .execute()
    )

    (
        db()
        .table("ceo_parameter_revisions")
        .insert(
            {
                "project_id": project_id,
                "parameter_key": parameter_key,
                "old_value_json": (
                    previous["value_json"] if previous else None
                ),
                "old_value_text": (
                    previous["value_text"] if previous else None
                ),
                "new_value_json": value_json,
                "new_value_text": value_text,
                "decision_type": decision_type,
                "ceo_message_id": ceo_message_id,
                "adopted_proposal_id": adopted_proposal_id,
            }
        )
        .execute()
    )

    # Making a parameter public removes any older secret version.
    (
        db()
        .table("ceo_secret_parameters")
        .delete()
        .eq("project_id", project_id)
        .eq("parameter_key", parameter_key)
        .execute()
    )

    update_proposal_statuses(
        project_id,
        parameter_key,
        adopted_proposal_id,
    )


def save_secret_parameter(
    project_id: str,
    ceo_message_id: str,
    parameter_key: str,
    parameter_label: str,
    value_json: Any,
    value_text: str,
    allowed_sections: list[str],
    secret_map: dict[str, dict[str, Any]],
) -> None:
    previous = secret_map.get(parameter_key)
    revision_number = (
        int(previous["revision_number"]) + 1
        if previous
        else 1
    )

    (
        db()
        .table("ceo_secret_parameters")
        .upsert(
            {
                "project_id": project_id,
                "parameter_key": parameter_key,
                "parameter_label": parameter_label,
                "value_json": value_json,
                "value_text": value_text,
                "allowed_sections": allowed_sections,
                "revision_number": revision_number,
                "ceo_message_id": ceo_message_id,
            },
            on_conflict="project_id,parameter_key",
        )
        .execute()
    )

    (
        db()
        .table("ceo_secret_parameter_revisions")
        .insert(
            {
                "project_id": project_id,
                "parameter_key": parameter_key,
                "old_value_json": (
                    previous["value_json"] if previous else None
                ),
                "old_value_text": (
                    previous["value_text"] if previous else None
                ),
                "new_value_json": value_json,
                "new_value_text": value_text,
                "allowed_sections": allowed_sections,
                "ceo_message_id": ceo_message_id,
            }
        )
        .execute()
    )

    # Making a parameter secret removes any older public version.
    (
        db()
        .table("ceo_parameters")
        .delete()
        .eq("project_id", project_id)
        .eq("parameter_key", parameter_key)
        .execute()
    )

    update_proposal_statuses(
        project_id,
        parameter_key,
        None,
    )


def apply_ceo_decisions(state: CEOState) -> dict[str, Any]:
    result = state["ceo_result"]

    public_decisions = result.get("public_decisions", [])
    secret_decisions = result.get("secret_decisions", [])

    if not isinstance(public_decisions, list):
        public_decisions = []

    if not isinstance(secret_decisions, list):
        secret_decisions = []

    public_map = parameter_map(state["public_parameters"])
    secret_map = parameter_map(state["secret_parameters"])

    proposals_by_id = {
        proposal["id"]: proposal
        for proposal in state["open_proposals"]
    }

    saved = 0

    for raw_decision in public_decisions:
        if not isinstance(raw_decision, dict):
            continue

        decision_type = safe_text(
            raw_decision.get("decision_type"),
            20,
        ).lower()

        if decision_type == "adopt":
            proposal_id = safe_text(
                raw_decision.get("proposal_id"),
                100,
            )
            proposal = proposals_by_id.get(proposal_id)

            if not proposal:
                continue

            parameter_key = proposal["parameter_key"]
            parameter_label = proposal["parameter_label"]
            value_json = proposal["value_json"]
            value_text = proposal["value_text"]
            adopted_proposal_id = proposal_id

        elif decision_type == "set":
            parameter_label = safe_text(
                raw_decision.get("parameter_label"),
                100,
            )
            parameter_key = normalise_key(
                raw_decision.get("parameter_key") or parameter_label
            )
            value_json = clean_json_value(raw_decision.get("value"))
            value_text = safe_text(
                raw_decision.get("value_text") or value_json,
                500,
            )
            adopted_proposal_id = None

            if not parameter_key or not value_text:
                continue

        else:
            continue

        save_public_parameter(
            project_id=state["project_id"],
            ceo_message_id=state["message_id"],
            parameter_key=parameter_key,
            parameter_label=parameter_label or (
                parameter_key.replace("_", " ").title()
            ),
            value_json=value_json,
            value_text=value_text,
            decision_type=decision_type,
            adopted_proposal_id=adopted_proposal_id,
            public_map=public_map,
        )

        public_map[parameter_key] = {
            "parameter_key": parameter_key,
            "parameter_label": parameter_label,
            "value_json": value_json,
            "value_text": value_text,
            "revision_number": (
                int(
                    public_map.get(
                        parameter_key,
                        {"revision_number": 0},
                    )["revision_number"]
                )
                + 1
            ),
        }

        saved += 1

    for raw_decision in secret_decisions:
        if not isinstance(raw_decision, dict):
            continue

        parameter_label = safe_text(
            raw_decision.get("parameter_label"),
            100,
        )
        parameter_key = normalise_key(
            raw_decision.get("parameter_key") or parameter_label
        )
        value_json = clean_json_value(raw_decision.get("value"))
        value_text = safe_text(
            raw_decision.get("value_text") or value_json,
            500,
        )

        if not parameter_key or not value_text:
            continue

        allowed_sections = normalise_sections(
            raw_decision.get("allowed_sections")
        )

        save_secret_parameter(
            project_id=state["project_id"],
            ceo_message_id=state["message_id"],
            parameter_key=parameter_key,
            parameter_label=parameter_label or (
                parameter_key.replace("_", " ").title()
            ),
            value_json=value_json,
            value_text=value_text,
            allowed_sections=allowed_sections,
            secret_map=secret_map,
        )

        secret_map[parameter_key] = {
            "parameter_key": parameter_key,
            "parameter_label": parameter_label,
            "value_json": value_json,
            "value_text": value_text,
            "revision_number": (
                int(
                    secret_map.get(
                        parameter_key,
                        {"revision_number": 0},
                    )["revision_number"]
                )
                + 1
            ),
        }

        saved += 1

    return {"decisions_saved": saved}


def build_employee_graph():
    graph = StateGraph(EmployeeState)

    graph.add_node("employee_analyze", employee_analyze)
    graph.add_node("store_employee_proposals", store_employee_proposals)

    graph.add_edge(START, "employee_analyze")
    graph.add_edge("employee_analyze", "store_employee_proposals")
    graph.add_edge("store_employee_proposals", END)

    return graph.compile()


def build_ceo_graph():
    graph = StateGraph(CEOState)

    graph.add_node("ceo_analyze", ceo_analyze)
    graph.add_node("apply_ceo_decisions", apply_ceo_decisions)

    graph.add_edge(START, "ceo_analyze")
    graph.add_edge("ceo_analyze", "apply_ceo_decisions")
    graph.add_edge("apply_ceo_decisions", END)

    return graph.compile()


EMPLOYEE_GRAPH = build_employee_graph()
CEO_GRAPH = build_ceo_graph()


def run_project_chat(
    project_id: str,
    section_tag: str,
    text: str,
) -> dict[str, Any]:
    if not text:
        raise ValueError("Message text cannot be empty.")

    get_project(project_id)

    is_ceo = section_tag == "ceo"

    message = insert_message(
        project_id=project_id,
        section_tag=section_tag,
        actor_type="ceo" if is_ceo else "employee",
        text=text,
        metadata={
            "label": "CEO" if is_ceo else section_tag.title(),
        },
    )

    history = get_section_messages(project_id, section_tag)

    if is_ceo:
        final_state = CEO_GRAPH.invoke(
            {
                "project_id": project_id,
                "section_tag": section_tag,
                "text": text,
                "message_id": message["id"],
                "history": history,
                "public_parameters": get_public_parameters(project_id),
                "secret_parameters": get_all_secret_parameters(project_id),
                "open_proposals": get_open_proposals(project_id),
            }
        )

        reply = final_state["reply"]
        decisions_saved = final_state.get("decisions_saved", 0)

    else:
        final_state = EMPLOYEE_GRAPH.invoke(
            {
                "project_id": project_id,
                "section_tag": section_tag,
                "text": text,
                "message_id": message["id"],
                "history": history,
                "visible_parameters": get_visible_parameters(
                    project_id,
                    section_tag,
                ),
            }
        )

        reply = final_state["reply"]
        decisions_saved = 0

    insert_message(
        project_id=project_id,
        section_tag=section_tag,
        actor_type="assistant",
        text=reply,
        metadata={
            "label": (
                "CEO Assistant"
                if is_ceo
                else f"{section_tag.title()} Assistant"
            ),
        },
    )

    return {
        "reply": reply,
        "section_tag": section_tag,
        "decisions_saved": decisions_saved,
        "visible_parameters": get_visible_parameters(
            project_id,
            section_tag,
        ),
    }
