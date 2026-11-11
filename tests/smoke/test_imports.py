def test_existing_and_new_modules_import():
    from deepresearch_agent.workflows.deep_research.agent import DeepResearchAgent  # noqa: F401
    from deepresearch_agent.workflows.multi_agent.orchestrator import MultiAgentOrchestrator  # noqa: F401
    from deepresearch_agent.harness import RunStatus, SourceMode  # noqa: F401
    from deepresearch_agent.persistence import ArtifactStore, Database  # noqa: F401


def test_core_can_import_without_http_backend():
    """核心运行、存储与工作流模块不能反向依赖 FastAPI 的 backend 包。"""
    import os
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
import sys
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'backend' or fullname.startswith('backend.'):
            raise AssertionError('核心包反向导入了 HTTP backend: ' + fullname)
sys.meta_path.insert(0, NoBackend())
from deepresearch_agent.persistence.repositories import RunRepository
from deepresearch_agent.persistence.schemas import MessageCreate, RunCreate
from deepresearch_agent.harness.runtime import HarnessRuntime
from deepresearch_agent.context import ContextBuilder, ContextCompactor
from deepresearch_agent.memory import MemoryService
from deepresearch_agent.workflows.deep_research.driver import DeepResearchDriver
from deepresearch_agent.workflows.multi_agent.driver import PlanExecuteReportDriver
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
