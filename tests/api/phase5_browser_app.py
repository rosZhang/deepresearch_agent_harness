"""Disposable local app used by the phase-5 browser acceptance smoke."""

from pathlib import Path

from backend.app.main import create_app
from tests.api.test_api_phase4 import ApiFakeDriver, ApiRouterLlm
from backend.app.services.message_router import MessageRouter

app = create_app(
    database_url="sqlite+aiosqlite:///./.pytest-tmp/phase5-browser.db",
    artifact_root=Path(".pytest-tmp/phase5-browser-artifacts"),
    workflow_factory=ApiFakeDriver,
    message_router=MessageRouter(ApiRouterLlm()),
    auto_resume=False,
)
