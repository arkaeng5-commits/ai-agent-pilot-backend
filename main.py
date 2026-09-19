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
    item.strip().rstrip("/")
    for item in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if item.strip()
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
def health():
    return {"status": "ok"}


@app.post("/run_graph", response_model=RunGraphResponse)
def run_graph(request: RunGraphRequest):
    logger.info(
        "run_graph started: project_id=%s role=%s",
        request.project_id,
        request.role,
    )

    try:
        return run_pilot(
            project_id=str(request.project_id),
            role=request.role,
            text=request.text.strip(),
        )
    except ProjectNotFound:
        raise HTTPException(status_code=404, detail="Project ID was not found.")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        logger.exception("run_graph failed")
        raise HTTPException(
            status_code=500,
            detail="The server failed. Check the Render logs.",
        )import logging
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
    item.strip().rstrip("/")
    for item in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if item.strip()
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
def health():
    return {"status": "ok"}


@app.post("/run_graph", response_model=RunGraphResponse)
def run_graph(request: RunGraphRequest):
    logger.info(
        "run_graph started: project_id=%s role=%s",
        request.project_id,
        request.role,
    )

    try:
        return run_pilot(
            project_id=str(request.project_id),
            role=request.role,
            text=request.text.strip(),
        )
    except ProjectNotFound:
        raise HTTPException(status_code=404, detail="Project ID was not found.")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        logger.exception("run_graph failed")
        raise HTTPException(
            status_code=500,
            detail="The server failed. Check the Render logs.",
        )
