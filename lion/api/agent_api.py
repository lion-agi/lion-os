import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import partial
from typing import Any, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_serializer

from lion.core.generic import Log, LogManager
from lion.core.models import Note
from lion.core.session.session_manager import SessionManager
from lion.libs.parse import to_dict
from lion.operator.agent import Agent
from lion.protocols.operatives.instruct import Instruct
from lion.settings import Settings

# Load environment variables
load_dotenv()

# Get database path from environment or use default
DB_PATH = os.environ.get("DATABASE_URL", "sessions.db").replace("sqlite:///", "")

# Create a single SessionManager instance
session_manager = SessionManager(db_path=DB_PATH)

# Create API logger
api_logger = LogManager(
    persist_dir="./data/logs/api",
    file_prefix="api_",
    capacity=1000,
    auto_save_on_exit=True,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing to do
    yield
    # Shutdown: cleanup
    session_manager.close()
    api_logger.dump()


app = FastAPI(title="Lion Agent API", version="1.0.0", lifespan=lifespan)

# Authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    # Check if key meets minimum length requirement
    if len(api_key) < Settings.API.MIN_KEY_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API key too short"
        )

    # Check if key is valid
    if api_key not in Settings.API.KEYS:
        # In development, allow dev keys if configured
        if Settings.API.ALLOW_DEV_KEYS and api_key.startswith("dev-"):
            return api_key

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )
    return api_key


# Models
class InstructRequest(BaseModel):
    session_id: str | None = Field(
        None, description="Session ID for continuing operations"
    )
    instruction: dict = Field(..., description="Instructions for the agent")
    choices: list[str] | None = Field(
        None, description="Choices for selection operation"
    )


class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime

    @field_serializer("created_at", "last_accessed", "expires_at")
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat()


class OperationHistoryEntry(BaseModel):
    operation: str
    instruct: dict
    result: dict
    timestamp: datetime

    @field_serializer("timestamp")
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat()


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


def create_instruct(data: dict) -> Instruct:
    """Create an Instruct object from request data."""
    if isinstance(data, dict):
        # Ensure required fields exist
        data.setdefault("instruction", "Generate ideas")
        data.setdefault("guidance", data.get("task", ""))
        data.setdefault("context", {})
        return Instruct(**data)
    return data


def log_api_call(
    endpoint: str, request_data: Any, response_data: Any, error: str | None = None
):
    """Log API call with request and response data."""
    content = {
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data,
    }

    loginfo = {"timestamp": datetime.now().isoformat(), "error": error}

    log_entry = Log(content=content, loginfo=loginfo)
    api_logger.log(log_entry)


# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_response = {"error": type(exc).__name__, "detail": str(exc)}
    log_api_call(request.url.path, None, error_response, str(exc))
    return JSONResponse(status_code=500, content=error_response)


# Session management endpoints
@app.post("/sessions", response_model=SessionResponse)
async def create_session(api_key: str = Depends(verify_api_key)):
    """Create a new agent session."""
    try:
        session_id = session_manager.create_session()

        # Get timestamps
        now = datetime.now()
        expires_at = now + timedelta(seconds=session_manager._default_ttl)

        response = SessionResponse(
            session_id=session_id,
            created_at=now,
            last_accessed=now,
            expires_at=expires_at,
        )

        response_dict = {
            "session_id": session_id,
            "created_at": now.isoformat(),
            "last_accessed": now.isoformat(),
            "expires_at": expires_at.isoformat(),
        }

        log_api_call("/sessions", {"api_key": "***"}, response_dict)
        return JSONResponse(content=response_dict)

    except Exception as e:
        log_api_call("/sessions", {"api_key": "***"}, None, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.get("/sessions/{session_id}/history", response_model=list[OperationHistoryEntry])
async def get_session_history(session_id: str, api_key: str = Depends(verify_api_key)):
    """Get operation history for a session."""
    try:
        history = session_manager.get_operation_history(session_id)
        # Convert timestamps to ISO format
        for entry in history:
            if isinstance(entry.get("timestamp"), datetime):
                entry["timestamp"] = entry["timestamp"].isoformat()
        log_api_call(
            f"/sessions/{session_id}/history", {"session_id": session_id}, history
        )
        return JSONResponse(content=history)
    except Exception as e:
        log_api_call(
            f"/sessions/{session_id}/history", {"session_id": session_id}, None, str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, api_key: str = Depends(verify_api_key)):
    """Delete a session."""
    try:
        session_manager.remove_session(session_id)
        response = {"status": "success"}
        log_api_call(f"/sessions/{session_id}", {"session_id": session_id}, response)
        return JSONResponse(content=response)
    except Exception as e:
        log_api_call(
            f"/sessions/{session_id}", {"session_id": session_id}, None, str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# Agent operation endpoints
@app.post("/agent/run")
async def run_agent(request: InstructRequest, api_key: str = Depends(verify_api_key)):
    """Run a complete agent workflow (brainstorm -> plan)."""
    try:
        # Create or get session
        session_id = request.session_id or str(uuid.uuid4())
        agent = Agent(session_id=session_id, session_manager=session_manager)

        # Create Instruct object
        instruct = create_instruct(request.instruction)

        # Execute operation
        result = await agent.run(instruct)

        response = {"session_id": session_id, "result": result}

        log_api_call("/agent/run", request.model_dump(), response)
        return JSONResponse(content=response)

    except Exception as e:
        log_api_call("/agent/run", request.model_dump(), None, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/agent/brainstorm")
async def brainstorm_agent(
    request: InstructRequest, api_key: str = Depends(verify_api_key)
):
    """Execute brainstorm operation."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        agent = Agent(session_id=session_id, session_manager=session_manager)

        # Create Instruct object
        instruct = create_instruct(request.instruction)

        result = await agent.brainstorm(instruct)

        response = {"session_id": session_id, "result": result}

        log_api_call("/agent/brainstorm", request.model_dump(), response)
        return JSONResponse(content=response)

    except Exception as e:
        log_api_call("/agent/brainstorm", request.model_dump(), None, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/agent/plan")
async def plan_agent(request: InstructRequest, api_key: str = Depends(verify_api_key)):
    """Execute plan operation."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        agent = Agent(session_id=session_id, session_manager=session_manager)

        # Create Instruct object
        instruct = create_instruct(request.instruction)

        result = await agent.plan(instruct)

        response = {"session_id": session_id, "result": result}

        log_api_call("/agent/plan", request.model_dump(), response)
        return JSONResponse(content=response)

    except Exception as e:
        log_api_call("/agent/plan", request.model_dump(), None, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/agent/select")
async def select_agent(
    request: InstructRequest, api_key: str = Depends(verify_api_key)
):
    """Execute select operation."""
    if not request.choices:
        error = "Choices are required for selection"
        log_api_call("/agent/select", request.model_dump(), None, error)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error)

    try:
        session_id = request.session_id or str(uuid.uuid4())
        agent = Agent(session_id=session_id, session_manager=session_manager)

        # Create Instruct object
        instruct = create_instruct(request.instruction)

        result = await agent.select(instruct, request.choices)

        response = {"session_id": session_id, "result": result}

        log_api_call("/agent/select", request.model_dump(), response)
        return JSONResponse(content=response)

    except Exception as e:
        log_api_call("/agent/select", request.model_dump(), None, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check API health."""
    response = {"status": "healthy", "timestamp": datetime.now().isoformat()}
    log_api_call("/health", None, response)
    return JSONResponse(content=response)
