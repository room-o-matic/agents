# Copied from room-o-matic/lobby src/lobbyd/verify.py. Keep the copies in sync.
"""Access-token verification for services that trust lobbyd (roomsd, agentd).

This module is meant to be copied into those services. It depends only on PyJWT and
httpx, not on the rest of lobbyd.

    verifier = TokenVerifier(
        issuer="https://lobby.example", domain="example", audience="https://rooms-a.example"
    )
    claims = verifier.verify(bearer_token)   # raises InvalidToken
    claims.identity, claims.scope            # "boostie@example", "agent"

Federation later: keep one verifier per trusted issuer and dispatch on the unverified `iss`.
A token is accepted only if `sub` ends in that issuer's own domain, so one lobbyd can't
mint identities in another's domain.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import jwt

ALGORITHMS = ["EdDSA"]


class InvalidToken(Exception):
    pass


@dataclass(frozen=True)
class Claims:
    identity: str  # name@domain
    name: str
    scope: str
    expires_at: int
    token_id: str


class TokenVerifier:
    def __init__(
        self,
        *,
        issuer: str,
        domain: str,
        audience: str,
        jwks_url: str | None = None,
        fetch_jwks: Callable[[], dict] | None = None,
        cache_seconds: float = 300,
        min_refresh_seconds: float = 10,
        leeway_seconds: float = 30,
    ):
        self.issuer = issuer.rstrip("/")
        self.domain = domain
        self.audience = audience.rstrip("/")
        url = jwks_url or f"{self.issuer}/.well-known/jwks.json"
        self._fetch = fetch_jwks or (lambda: httpx.get(url, timeout=5).raise_for_status().json())
        self._cache_seconds = cache_seconds
        self._min_refresh = min_refresh_seconds
        self._leeway = leeway_seconds
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at = 0.0
        self._lock = threading.Lock()

    def _refresh(self, force: bool) -> None:
        with self._lock:
            age = time.monotonic() - self._fetched_at
            # Refetching on an unknown kid handles rotation, but never more than once per
            # min_refresh_seconds so bogus kids can't make us hammer lobbyd.
            if age < (self._min_refresh if force else self._cache_seconds):
                return
            try:
                keyset = jwt.PyJWKSet.from_dict(self._fetch())
            except Exception as e:  # noqa: BLE001 - keep serving from the old cache
                if not self._keys:
                    raise InvalidToken(f"cannot fetch issuer keys: {e}") from e
                return
            self._keys = {k.key_id: k for k in keyset.keys if k.key_id}
            self._fetched_at = time.monotonic()

    def _key(self, kid: str) -> jwt.PyJWK:
        self._refresh(force=False)
        if kid not in self._keys:
            self._refresh(force=True)
        if kid not in self._keys:
            raise InvalidToken(f"unknown signing key {kid!r}")
        return self._keys[kid]

    def verify(self, token: str) -> Claims:
        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except jwt.PyJWTError as e:
            raise InvalidToken(f"malformed token: {e}") from e
        if not kid:
            raise InvalidToken("token has no kid")
        try:
            claims = jwt.decode(
                token,
                self._key(kid),
                algorithms=ALGORITHMS,
                audience=self.audience,
                issuer=self.issuer,
                leeway=self._leeway,
                options={"require": ["exp", "iat", "sub", "aud", "iss", "jti"]},
            )
        except jwt.PyJWTError as e:
            raise InvalidToken(str(e)) from e
        name, sep, domain = claims["sub"].rpartition("@")
        if not sep or not name or domain != self.domain:
            raise InvalidToken(f"subject {claims['sub']!r} is not in domain {self.domain!r}")
        return Claims(
            identity=claims["sub"],
            name=name,
            scope=claims.get("scope", ""),
            expires_at=claims["exp"],
            token_id=claims["jti"],
        )
