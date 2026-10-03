# Copied from room-o-matic/lobby src/lobbyd/verify.py. Keep the copies in sync.
"""Access-token verification for services that trust lobbyd (roomsd, agentd).

This module is meant to be copied into those services. It depends only on PyJWT and
httpx, not on the rest of lobbyd.

    verifier = TokenVerifier(
        issuer="https://lobby.example", domain="example", audience="https://rooms-a.example"
    )
    claims = verifier.verify(bearer_token)   # raises InvalidToken
    claims.identity, claims.scope            # "boostie@example", "agent"

Key cache contract (room-o-matic/docs#6):

- A token signed by a key already in the cache never waits on the network. When the cache
  is older than `cache_seconds` it is refreshed in the background, by at most one
  in-flight fetch.
- A token signed by an unknown key triggers at most one synchronous fetch, shared by
  concurrent callers, and only if the last attempt was more than `min_refresh_seconds`
  ago. lobbyd publishes keys before they sign (signing.py), so this is the fallback path.
- Failed fetches back off exponentially (`backoff_base_seconds` doubling up to
  `backoff_max_seconds`) and don't count as success, so an outage doesn't turn every
  request into a network attempt.
- After `max_stale_seconds` without a successful fetch the cache is unusable and
  verification fails closed. That bounds how long a key that lobbyd retired in an emergency
  can still be accepted while lobbyd is unreachable. A successful fetch drops retired
  keys immediately.

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
        min_refresh_seconds: float = 1,
        backoff_base_seconds: float = 1,
        backoff_max_seconds: float = 60,
        max_stale_seconds: float = 3600,
        fetch_timeout_seconds: float = 5,
        leeway_seconds: float = 30,
        clock: Callable[[], float] = time.monotonic,
        background: Callable[[Callable[[], None]], None] | None = None,
    ):
        self.issuer = issuer.rstrip("/")
        self.domain = domain
        self.audience = audience.rstrip("/")
        url = jwks_url or f"{self.issuer}/.well-known/jwks.json"
        self._fetch = fetch_jwks or (
            lambda: httpx.get(url, timeout=fetch_timeout_seconds).raise_for_status().json()
        )
        self.cache_seconds = cache_seconds
        self.min_refresh_seconds = min_refresh_seconds
        self.backoff_base_seconds = backoff_base_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.max_stale_seconds = max_stale_seconds
        self.fetch_timeout_seconds = fetch_timeout_seconds
        self._leeway = leeway_seconds
        self._clock = clock
        self._background = background or (
            lambda fn: threading.Thread(target=fn, name="jwks-refresh", daemon=True).start()
        )
        self._lock = threading.Lock()
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at: float | None = None  # last success
        self._attempted_at: float | None = None  # last attempt, success or not
        self._failures = 0
        self._inflight: threading.Event | None = None
        self.fetches = 0  # attempts made; for tests and metrics

    def health(self) -> dict:
        """For readiness and metrics (docs#24): when keys were last confirmed with lobbyd,
        and whether verification is failing closed right now."""
        with self._lock:
            age = None if self._fetched_at is None else self._clock() - self._fetched_at
            return {
                "fetched": self._fetched_at is not None,
                "age_seconds": age,
                "consecutive_failures": self._failures,
                "fetches": self.fetches,
                "failing_closed": (age is not None and age > self.max_stale_seconds)
                or (age is None and self._failures > 0),
            }

    # ----- cache ---------------------------------------------------------------------

    def _due(self, now: float, *, unknown_kid: bool) -> bool:
        """Whether a new fetch may start now. Called with the lock held."""
        if self._attempted_at is None:
            return True
        since_attempt = now - self._attempted_at
        if self._failures:
            backoff = self.backoff_base_seconds * 2 ** (self._failures - 1)
            return since_attempt >= min(backoff, self.backoff_max_seconds)
        if unknown_kid:
            return since_attempt >= self.min_refresh_seconds
        return now - (self._fetched_at or 0) >= self.cache_seconds

    def _start_fetch(self, *, unknown_kid: bool) -> threading.Event | None:
        """Join the in-flight fetch or start one if due. Returns an event that is set
        when that fetch finishes, or None if no fetch is running or allowed."""
        with self._lock:
            if self._inflight is not None:
                return self._inflight
            if not self._due(self._clock(), unknown_kid=unknown_kid):
                return None
            self._inflight = done = threading.Event()
            self._attempted_at = self._clock()
            self.fetches += 1
        self._background(lambda: self._run_fetch(done))
        return done

    def _run_fetch(self, done: threading.Event) -> None:
        try:
            keyset = jwt.PyJWKSet.from_dict(self._fetch())
            keys = {k.key_id: k for k in keyset.keys if k.key_id}
        except Exception:  # noqa: BLE001 - any fetch failure just backs off
            with self._lock:
                self._failures += 1
        else:
            with self._lock:
                # Replace, don't merge: keys lobbyd stopped publishing are dropped now.
                self._keys = keys
                self._fetched_at = self._clock()
                self._failures = 0
        finally:
            with self._lock:
                self._inflight = None
            done.set()

    def _usable_keys(self) -> dict[str, jwt.PyJWK]:
        with self._lock:
            if self._fetched_at is None:
                return {}
            if self._clock() - self._fetched_at > self.max_stale_seconds:
                return {}  # fail closed: too long without confirming keys with lobbyd
            return self._keys

    def _key(self, kid: str) -> jwt.PyJWK:
        keys = self._usable_keys()
        if kid in keys:
            # Known key: never wait. Refresh in the background if the cache is stale.
            self._start_fetch(unknown_kid=False)
            return keys[kid]
        # Unknown kid (or no usable cache): at most one bounded synchronous fetch.
        done = self._start_fetch(unknown_kid=True)
        if done is not None:
            done.wait(self.fetch_timeout_seconds + 1)
        keys = self._usable_keys()
        if kid in keys:
            return keys[kid]
        if not keys and self._fetched_at is not None:
            raise InvalidToken("issuer keys are stale and lobbyd is unreachable")
        if not keys:
            raise InvalidToken("cannot fetch issuer keys")
        raise InvalidToken(f"unknown signing key {kid!r}")

    # ----- verification --------------------------------------------------------------

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
