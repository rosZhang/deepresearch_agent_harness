"""FastAPI application entry wired to the persistent Harness main chain."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.v1 import router as api_router
from backend.app.bootstrap import build_services
from backend.app.services.message_router import MessageRouter
from deepresearch_agent.config import settings
from deepresearch_agent.harness.errors import AppError, ErrorCode


def _error(request: Request, *, status_code: int, code: str, message: str, retryable: bool = False, details=None):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "retryable": retryable, "details": details or {}}, "request_id": getattr(request.state, "request_id", "unknown")},
    )


def create_app(
    *, database_url: str | None = None, workflow_factory=None,
    artifact_root: str | Path | None = None, skills_root: str | Path | None = None,
    auto_resume: bool | None = None, message_router: MessageRouter | None = None,
) -> FastAPI:
    if not settings.LOCAL_MVP_SINGLE_WORKER:
        raise RuntimeError("本地 MVP 必须设置 FASTAPI_WORKERS=1")
    services = build_services(
        database_url=database_url, workflow_factory=workflow_factory,
        artifact_root=artifact_root, skills_root=skills_root, message_router=message_router,
    )
    database, run_service = services.database, services.run_service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await database.create_schema()
        await run_service.startup_recovery(auto_resume=settings.AUTO_RESUME_RUNS if auto_resume is None else auto_resume)
        yield
        await run_service.shutdown()
        await database.close()

    app = FastAPI(title="DeepResearch HybridRAG Local MVP", version="0.4.0", lifespan=lifespan)
    app.state.database = database
    app.state.run_service = run_service
    app.state.chat_service = services.chat_service
    app.state.event_stream = services.event_stream
    app.state.memory_service = services.memory_service
    app.state.skill_services = services.skill_services
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.FRONTEND_ORIGINS), allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex}"
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        status_code = {
            ErrorCode.NOT_FOUND: 404, ErrorCode.CONFLICT: 409, ErrorCode.IDEMPOTENCY_CONFLICT: 409,
            ErrorCode.INVALID_SOURCE_MODE: 422, ErrorCode.TAVILY_API_KEY_MISSING: 503,
            ErrorCode.SOURCE_UNAVAILABLE: 503, ErrorCode.DATABASE_UNAVAILABLE: 503,
            ErrorCode.MESSAGE_ROUTING_FAILED: 502,
        }.get(exc.code, 400)
        return _error(request, status_code=status_code, code=exc.code.value, message=exc.message, retryable=exc.retryable, details=exc.details)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return _error(request, status_code=422, code="VALIDATION_ERROR", message="请求参数校验失败", details={"errors": exc.errors()})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        return _error(request, status_code=500, code="INTERNAL_ERROR", message="服务内部错误")

    app.include_router(api_router)
    return app


app = create_app()
