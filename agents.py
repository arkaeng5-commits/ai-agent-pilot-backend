from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
import os, json
from supabase import create_client
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # harmless if no .env; cloud host will set env vars

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

llm_client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.1-70b-versatile"

def call_llm(prompt: str) -> str:
    resp = llm_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

class Assumption(BaseModel):
    id: str
    owner: str
    topic: str
    proposed_value: Optional[float] = None
    approved_value: Optional[float] = None
    unit: Optional[str] = None
    text: str
    status: str = "proposed"

class ProjectState(BaseModel):
    project_id: str
    project_name: str = ""
    validation_status: str = "pending"
    assumptions: List[Assumption] = []
    last_processed_chat_id: str = ""

def load_project_state(project_id: str) -> ProjectState:
    resp = supabase.table("projects").select("*").eq("id", project_id).execute()
    row = resp.data[0]
    return ProjectState(
        project_id=row["id"],
        project_name=row["project_name"] or "",
        validation_status=row["validation_status"] or "pending",
        assumptions=[Assumption(**a) for a in (row["assumptions"] or [])],
        last_processed_chat_id=row["last_processed_message_id"] or "",
    )

def save_project_state(state: ProjectState):
    supabase.table("projects").update({
        "validation_status": state.validation_status,
        "assumptions": [a.model_dump() for a in state.assumptions],
        "last_processed_message_id": state.last_processed_chat_id,
    }).eq("id", state.project_id).execute()

def load_new_messages(project_id: str, last_id: str):
    resp = (
        supabase
        .table("messages")
        .select("*")
        .eq("project_id", project_id)
        .order("created_at", desc=False)
        .execute()
    )
    all_msgs = resp.data
    if not last_id:
        return all_msgs
    new_msgs = []
    seen = False
    for m in all_msgs:
        if m["id"] == last_id:
            seen = True
            continue
        if seen:
            new_msgs.append(m)
    return new_msgs

GraphState = Dict[str, Any]

def process_chat_node(state: GraphState) -> GraphState:
    project = load_project_state(state["project_id"])
    new_msgs = load_new_messages(project.project_id, project.last_processed_chat_id)
    if not new_msgs:
        return {"project_id": project.project_id}

    chat_texts = [f"[{m['role']}] {m['text']}" for m in new_msgs]

    prompt = f"""
You are an assistant that reads chat messages from a multi-agent system with three roles: finance, rd, ceo.
People discuss project assumptions about various topics (budget, timeline, headcount, equipment, etc.).

Recent new messages:
{json.dumps(chat_texts, indent=2, ensure_ascii=False)}

Tasks:
1. Detect all topics being discussed (e.g. "budget", "timeline", "headcount", "equipment", "travel", etc.).
2. For each topic where someone proposes a numeric value, create a topic object with:
   - topic: short lowercase name (e.g. "budget", "timeline_months", "headcount")
   - proposed_value: the numeric value mentioned (float or null)
   - unit: short unit if mentioned (e.g. "USD", "months", "people"), or null
   - text: short description of the assumption (1 sentence)
3. Return ONLY valid JSON with this shape:
{{
  "new_topics": [
    {{
      "id": "<unique-id>",
      "owner": "<finance|rd|ceo>",
      "topic": "<topic name>",
      "proposed_value": <float or null>,
      "unit": "<unit or null>",
      "text": "<short description>"
    }}
  ],
  "new_last_processed_message_id": "<id of last message in messages>"
}}

Rules:
- Infer the owner from the message prefix [finance], [rd], [ceo].
- If multiple messages refer to the same topic, create one topic per distinct idea (use your judgment).
- Do not include any explanation, only JSON.
"""

    content = call_llm(prompt)
    delta = json.loads(content)

    for t in delta.get("new_topics", []):
        project.assumptions.append(Assumption(
            id=t["id"],
            owner=t["owner"],
            text=t["text"],
            topic=t["topic"],
            proposed_value=t.get("proposed_value"),
            approved_value=None,
            unit=t.get("unit"),
            status="proposed",
        ))

    project.last_processed_chat_id = delta["new_last_processed_message_id"]
    save_project_state(project)
    return {"project_id": project.project_id}

def parent_validator_node(state: GraphState) -> GraphState:
    project = load_project_state(state["project_id"])

    RULES = {
        "budget": {"max": 50.0},
        "timeline_months": {"max": 3.0},
        "headcount": {"max": 5.0},
    }

    for a in project.assumptions:
        if a.status != "proposed":
            continue
        if a.proposed_value is None:
            a.approved_value = a.proposed_value
            a.status = "approved"
            continue

        rule = RULES.get(a.topic)
        if rule is None:
            a.approved_value = a.proposed_value
            a.status = "approved"
            continue

        max_val = rule["max"]
        if a.proposed_value > max_val:
            a.approved_value = max_val
            a.status = "corrected"
        else:
            a.approved_value = a.proposed_value
            a.status = "approved"

    has_corrected = any(a.status == "corrected" for a in project.assumptions)
    project.validation_status = "corrected" if has_corrected else "ok"

    save_project_state(project)
    return {"project_id": project.project_id}

def generate_reply_node(state: GraphState) -> GraphState:
    project = load_project_state(state["project_id"])
    topics_text = "\n".join(
        f"- {a.topic}: proposed {a.proposed_value}, approved {a.approved_value} ({a.status})"
        for a in project.assumptions
    )

    prompt = f"""
You are an AI assistant for a multi-agent project system.
Based on the updated project state, write a short, friendly chat reply.

Current topics:
{topics_text}

Validation status: {project.validation_status}

Tasks:
- Briefly confirm what was recorded.
- If any topic was corrected, clearly explain the correction.
- Show a short summary of current topics (budget, timeline, headcount).
- Keep it concise and natural, like a chat message.
"""

    reply = call_llm(prompt)
    return {"project_id": project.project_id, "ai_reply": reply}

builder = StateGraph(GraphState)

builder.add_node("process_chat", process_chat_node)
builder.add_node("parent_validator", parent_validator_node)
builder.add_node("generate_reply", generate_reply_node)

builder.add_edge(START, "process_chat")
builder.add_edge("process_chat", "parent_validator")
builder.add_edge("parent_validator", "generate_reply")
builder.add_edge("generate_reply", END)

graph = builder.compile()