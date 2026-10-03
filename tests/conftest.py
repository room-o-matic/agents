import time
import uuid
from collections.abc import Callable

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

from agentd.app import create_app
from agentd.config import Settings
from agentd.verify import TokenVerifier

ISSUER = "http://lobby.test"
DOMAIN = "test"
BASE_URL = "http://agentd.test"


class FakeLobby:
    """Stands in for lobbyd: signs access tokens with a test key and serves its JWKS."""

    kid = "test-key"

    def __init__(self):
        self.key = Ed25519PrivateKey.generate()

    def jwks(self) -> dict:
        jwk = jwt.algorithms.OKPAlgorithm.to_jwk(self.key.public_key(), as_dict=True)
        return {"keys": [{**jwk, "kid": self.kid, "alg": "EdDSA", "use": "sig"}]}

    def verifier(self) -> TokenVerifier:
        return TokenVerifier(issuer=ISSUER, domain=DOMAIN, audience=BASE_URL, fetch_jwks=self.jwks)

    def token(self, name: str, *, scope="agent", aud=BASE_URL, ttl=900) -> str:
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "sub": f"{name}@{DOMAIN}",
            "aud": aud,
            "scope": scope,
            "iat": now,
            "nbf": now,
            "exp": now + ttl,
            "jti": uuid.uuid4().hex,
        }
        return jwt.encode(claims, self.key, algorithm="EdDSA", headers={"kid": self.kid})

    def headers(self, name: str, **kw) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token(name, **kw)}"}


@pytest.fixture
def lobby() -> FakeLobby:
    return FakeLobby()


@pytest.fixture
def workspace_root(tmp_path):
    root = tmp_path / "ws"
    (root / "repo").mkdir(parents=True)
    return root


@pytest.fixture
def settings(tmp_path, workspace_root) -> Settings:
    return Settings(
        instance_id="agentd-test",
        base_url=BASE_URL,
        lobbyd_url=ISSUER,
        lobbyd_domain=DOMAIN,
        data_dir=tmp_path / "data",
        max_sessions=2,
        workspace_roots=[workspace_root],
        ready_timeout_seconds=3,
        stop_grace_seconds=1,
        cleanup_interval_seconds=0.2,
        max_log_bytes=2000,
    )


@pytest.fixture
def client(settings, lobby):
    with TestClient(
        create_app(settings, verifier=lobby.verifier())
    ) as c:  # runs lifespan: supervisor, cleanup loop
        yield c


@pytest.fixture
def make_agent(lobby) -> Callable[..., dict[str, str]]:
    return lobby.headers


@pytest.fixture
def boostie(make_agent):
    return make_agent("boostie")


@pytest.fixture
def missy(make_agent):
    return make_agent("missy")
