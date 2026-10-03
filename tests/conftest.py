from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from agentd import auth
from agentd.app import create_app
from agentd.config import Settings


@pytest.fixture
def workspace_root(tmp_path):
    root = tmp_path / "ws"
    (root / "repo").mkdir(parents=True)
    return root


@pytest.fixture
def settings(tmp_path, workspace_root) -> Settings:
    return Settings(
        instance_id="agentd-test",
        data_dir=tmp_path / "data",
        max_sessions=2,
        workspace_roots=[workspace_root],
        ready_timeout_seconds=3,
        stop_grace_seconds=1,
        cleanup_interval_seconds=0.2,
        max_log_bytes=2000,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:  # runs lifespan: supervisor, cleanup loop
        yield c


@pytest.fixture
def make_agent(client) -> Callable[[str], dict[str, str]]:
    def _make(agent: str) -> dict[str, str]:
        token = auth.create_token(client.app.state.conn, agent)
        return {"Authorization": f"Bearer {token}"}

    return _make


@pytest.fixture
def boostie(make_agent):
    return make_agent("boostie")


@pytest.fixture
def missy(make_agent):
    return make_agent("missy")
