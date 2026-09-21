import logging
import os
from typing import Any, Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents import ProjectNotFound, run_pilot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

origins = [
    i.strip().rstrip("/")
    for i in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if i.strip()
]

app = FastAPI(title="Any-Topic AI Pilot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class RunGraphRequest(BaseModel):
    project_id: UUID
    role: Literal["finance", "rnd", "ceo"]
    text: str = Field(min_length=1, max_length=4000)


class RunGraphResponse(BaseModel):
    ai_reply: str
    topic: str
    assumptions: dict[str, Any]
    validation_status: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run_graph", response_model=RunGraphResponse)
def run_graph(req: RunGraphRequest) -> RunGraphResponse:
    try:
        result = run_pilot(
            project_id=str(req.project_id),
            role=req.role,
            text=req.text.strip(),
        )
        return RunGraphResponse(**result)
    except ProjectNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled error in /run_graph")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
