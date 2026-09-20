import logging
import os
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents import ProjectNotFound, get_section_messages, run_project_chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

origins = [
    item.strip().rstrip("/")
    for item in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if item.strip()
]

app = FastAPI(title="Section and CEO Project API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    project_id: UUID
    section_tag: str = Field(min_length=2, max_length=50)
    text: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/projects/{project_id}/messages")
def project_messages(
    project_id: UUID,
    section_tag: str,
):
    try:
        return {
            "messages": get_section_messages(
                project_id=str(project_id),
                section_tag=section_tag.strip().lower(),
            )
        }
    except Exception:
        logger.exception("Could not load messages")
        raise HTTPException(
            status_code=500,
            detail="Could not load this section's messages.",
        )


@app.post("/run_graph")
def run_graph(request: ChatRequest):
    try:
        return run_project_chat(
            project_id=str(request.project_id),
            section_tag=request.section_tag.strip().lower(),
            text=request.text.strip(),
        )
    except ProjectNotFound:
        raise HTTPException(
            status_code=404,
            detail="Project ID was not found.",
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
    except Exception:
        logger.exception("run_graph failed")
        raise HTTPException(
            status_code=500,
            detail="The server failed. Check Render Logs.",
        )
