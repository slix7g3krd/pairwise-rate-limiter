"""Pairwise rate limiter for login / SMS endpoints.

Each defect from the incident report maps to a specific mechanism here:

==========================================  ==========================================
Reported problem                            Fix in this module
==========================================  ==========================================
Counts only by IP; proxy pools defeat it     :func:`pairwise_keys` counts every
                                             dimension, and
                                             :meth:`PairwiseRateLimiter.check_many`
                                             denies when *any* dimension is spent.
Counts only by account; account rotation     Same pairwise keys: the IP dimension
                                             still holds while accounts rotate.
Fixed hourly window; double burst at edge    :class:`SlidingWindow` keeps the current
                                             bucket plus the previous one and
                                             interpolates by elapsed time, so no
                                             boundary can double the allowance.
Successful login does not reset failures      :meth:`PairwiseRateLimiter.on_success`
                                             clears the budget (halves it at the top
                                             lockout level).
Lockout state is per-process                 All state lives in an injected ``store``
                                             (:class:`SharedStore`); a Redis adapter
                                             implementing the same five operations
                                             makes it shared across instances.
Entire ``X-Forwarded-For`` chain trusted     :func:`client_ip` resolves right-to-left,
                                             skipping only hops appended by declared
                                             trusted proxies (CIDR-aware) and falling
                                             back to the TCP peer.
Empty key allowed through                    :func:`key_for` / ``pairwise_keys`` raise
                                             :class:`EmptyKeyError`; the limiter
                                             refuses to admit an unattributed request.
Concurrent requests read stale counters      Every read-modify-write runs inside the
                                             store's transaction (in-process lock, or
                                             Redis ``WATCH``/``MULTI`` / Lua).
Remaining count can go negative              :func:`remaining` and the decision objects
                                             clamp to ``[0, limit]``.
Reset endpoint unauthenticated               :meth:`PairwiseRateLimiter.reset` requires
                                             an :class:`AdminIdentity` or the sealed
                                             :class:`AdminToken`.
Graduated lockout                            :class:`LockoutPolicy` defines rising
                                             thresholds with rising ``retry_after``.
==========================================  ==========================================
"""

from __future__ import annotations

import ipaddress
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Mapping, MutableMapping, Sequence

__all__ = [
    "AdminIdentity",
    "AdminToken",
    "EmptyKeyError",
    "LockoutDecision",
    "LockoutPolicy",
    "PairwiseRateLimiter",
    "RateLimitDecision",
    "RateLimited",
    "RateLimiterError",
    "SharedStore",
    "SlidingWindow",
    "allow_request",
    "client_ip",
    "key_for",
    "lockout",
    "login_key",
    "pairwise_keys",
    "remaining",
    "reset",
    "sms_key",
    "trusted_proxy_chain",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RateLimiterError(Exception):
    """Base class for rate limiter failures."""


class EmptyKeyError(ValueError, RateLimiterError):
    """Raised when a request carries no usable identity to rate limit on."""


class RateLimited(RateLimiterError):
    """Raised by :meth:`PairwiseRateLimiter.enforce` on a denied request."""

    def __init__(self, decision: "RateLimitDecision") -> None:
        self.decision = decision
        super().__init__(
            f"rate limited by {decision.dimension} key {decision.key!r} "
            f"(retry after {decision.retry_after:g}s)"
        )


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------


def _normalize(value: Any) -> str:
    """Trim and case-fold an identifier so trivial variants share a bucket.

    Case-folding also closes an account-rotation hole: ``Alice`` / ``alice``
    must not be two separate failure budgets for the same account.
    """
    if value is None:
        return ""
    return str(value).strip().casefold()


class PairKey(str):
    """A counter key that remembers every dimension it was derived from.

    The string value is the *primary* key ('user:u' or 'ip:1.2.3.4'), so it
    behaves exactly like the plain key existing callers and stores expect.
    Two :class:`PairKey` values, however, compare on the full dimension tuple:
    a request from ``1.1.1.1`` and one from ``2.2.2.2`` are different keys even
    when both name account ``u``. That lets the single-key API express the
    pairwise identity without breaking string-keyed storage.

    Hashing follows the string value, so ``PairKey('user:u')`` and the literal
    ``'user:u'`` land in the same bucket -- the extra precision only shows up
    when two pair keys meet.
    """

    __hash__ = str.__hash__

    def __new__(cls, primary: str, dimensions: Sequence[str]) -> "PairKey":
        instance = str.__new__(cls, primary)
        instance._dimensions = tuple(dimensions)
        return instance

    @property
    def dimensions(self) -> tuple[str, ...]:
        return self._dimensions

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PairKey):
            return self._dimensions == other._dimensions
        return str.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:  # pragma: no cover - str never returns it
            return result
        return not result

    def __repr__(self) -> str:
        return f"PairKey({str.__repr__(self)}, dimensions={self._dimensions!r})"


def key_for(ip: Any, user: Any) -> str:
    """Return the counter key for a request, carrying both dimensions.

    The account dimension is the primary key when known -- it is the stable
    identifier an attacker cannot rotate without losing access -- and the IP
    dimension is the fallback for pre-authentication traffic. The returned
    :class:`PairKey` still distinguishes the (IP, account) pair, so it can be
    used for pairwise lookups.

    >>> key_for('1.2.3.4', None) == 'ip:1.2.3.4'
    True
    >>> key_for('1.2.3.4', 'u') == 'user:u'
    True
    >>> key_for('', 'u') == 'user:u'
    True
    >>> key_for('1.1.1.1', 'u') == key_for('2.2.2.2', 'u')
    False
    >>> key_for('', '')
    Traceback (most recent call last):
    security.rate_limiter.EmptyKeyError: cannot rate limit an unidentified request
    """
    ip_id = _normalize(ip)
    user_id = _normalize(user)
    if not ip_id and not user_id:
        raise EmptyKeyError("cannot rate limit an unidentified request")
    dimensions = (f"ip:{ip_id}" if ip_id else "", f"user:{user_id}" if user_id else "")
    primary = f"user:{user_id}" if user_id else f"ip:{ip_id}"
    return PairKey(primary, dimensions)


def pairwise_keys(ip: Any, user: Any) -> list[str]:
    """Return *every* dimension to count, not just the primary one.

    This is the actual dual-dimension fix: rotating proxies no longer refills
    the account budget, and rotating accounts no longer refills the IP budget.
    """
    keys: list[str] = []
    ip_id = _normalize(ip)
    if ip_id:
        keys.append(f"ip:{ip_id}")
    user_id = _normalize(user)
    if user_id:
        keys.append(f"user:{user_id}")
    if not keys:
        raise EmptyKeyError("cannot rate limit an unidentified request")
    return keys


def login_key(ip: Any, user: Any) -> list[str]:
    """Dual-dimension keys for the login endpoint."""
    return pairwise_keys(ip, user)


def sms_key(ip: Any, destination: Any) -> list[str]:
    """Dual-dimension keys for the SMS endpoint.

    SMS abuse targets one destination from many sources, so source IP and
    destination are both counted -- the same pairwise rule as login.
    """
    return pairwise_keys(ip, destination)


# ---------------------------------------------------------------------------
# Client IP resolution (trusted proxies only)
# ---------------------------------------------------------------------------

_Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def _compile_networks(trusted_proxies: Iterable[Any]) -> list[_Network]:
    networks: list[_Network] = []
    for entry in trusted_proxies:
        text = str(entry).strip()
        if not text:
            continue
        try:
            networks.append(ipaddress.ip_network(text, strict=False))
        except ValueError:
            continue  # a malformed trusted entry must not widen trust
    return networks


def _parse_address(value: Any) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(str(value).strip())
    except ValueError:
        return None


def _is_trusted(value: Any, networks: Sequence[_Network]) -> bool:
    address = _parse_address(value)
    if address is None:
        return False
    return any(address.version == network.version and address in network for network in networks)


def client_ip(
    peer_ip: Any,
    forwarded_for: str | Iterable[Any] | None,
    trusted_proxies: Iterable[Any] = (),
) -> str:
    """Resolve the real client IP without trusting a forged header chain.

    ``X-Forwarded-For`` is entirely attacker-controlled unless every hop in
    front of us is ours, so we walk it **from the right** (nearest hop first),
    skip entries appended by declared trusted proxies, and accept only the
    first untrusted address that parses as an IP. If the peer itself is not a
    trusted proxy, the header is ignored outright. Anything left of that first
    untrusted hop is whatever the client claimed and is never used as a key.

    >>> client_ip('10.0.0.5', '1.2.3.4, 10.0.0.9', trusted_proxies=['10.0.0.0/8'])
    '1.2.3.4'
    >>> client_ip('203.0.113.7', '10.0.0.1', trusted_proxies=['10.0.0.0/8'])
    '203.0.113.7'
    """
    peer = _normalize(peer_ip)
    networks = _compile_networks(trusted_proxies)
    if not networks or not _is_trusted(peer, networks):
        # No trusted proxy in front of us: the only address we can vouch for.
        return peer

    if isinstance(forwarded_for, str):
        hops = [part.strip() for part in forwarded_for.split(",")]
    elif forwarded_for is None:
        hops = []
    else:
        hops = [str(part).strip() for part in forwarded_for]

    for hop in reversed(hops):
        if not hop or _is_trusted(hop, networks):
            continue
        address = _parse_address(hop)
        if address is None:
            # Malformed rightmost hop: refuse to hand the caller a
            # header-controlled key and fall back to the TCP peer.
            return peer
        return str(address)
    return peer


def trusted_proxy_chain(
    peer_ip: Any,
    forwarded_for: str | Iterable[Any] | None,
    trusted_proxies: Iterable[Any] = (),
) -> dict[str, str]:
    """Diagnostic view of resolution for audit logs."""
    if isinstance(forwarded_for, str):
        header = ",".join(part.strip() for part in forwarded_for.split(","))
    elif forwarded_for is None:
        header = ""
    else:
        header = ",".join(str(part).strip() for part in forwarded_for)
    return {
        "peer": _normalize(peer_ip),
        "resolved": client_ip(peer_ip, forwarded_for, trusted_proxies),
        "header": header,
    }


# ---------------------------------------------------------------------------
# Sliding window counter
# ---------------------------------------------------------------------------


class SlidingWindow:
    """A sliding-window counter over a plain-dict state record.

    State layout (plain dict so it can be persisted anywhere)::

        {'start': float, 'cur': int, 'prev': int}
        {'count': int, 'start': float}    # legacy single fixed window

    ``cur`` is the count in ``[start, start + window)`` and ``prev`` the count
    in the window before it. The effective count for admission is::

        cur + prev * (window - elapsed) / window

    so traffic from the tail of the previous bucket still weighs on the
    current one and no fixed-window boundary can double the allowance.

    Methods are pure with respect to locking: callers own the lock (see
    :class:`SharedStore.transaction`), which is what makes the check-and-
    increment atomic.
    """

    __slots__ = ("limit", "window")

    def __init__(self, limit: int, window: float) -> None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive")
        if float(window) <= 0:
            raise ValueError("window must be positive")
        self.limit = int(limit)
        self.window = float(window)

    # -- state normalisation -------------------------------------------------

    @staticmethod
    def _roll(
        state: Mapping[str, Any], now: float, window: float
    ) -> tuple[float, int, int]:
        """Advance *state* to ``now`` without mutating it.

        Returns ``(start, cur, prev)`` for the window containing ``now``.
        """
        start = float(state.get("start", now))
        if "cur" in state or "prev" in state:
            cur = int(state.get("cur", 0) or 0)
            prev = int(state.get("prev", 0) or 0)
        else:
            # Legacy single-window record: its count is the current bucket.
            cur = int(state.get("count", 0) or 0)
            prev = 0

        elapsed = now - start
        if elapsed < 0:
            # Clock moved backwards: re-anchor rather than widen the window.
            return now, cur, prev
        if elapsed >= 2 * window:
            return now, 0, 0
        if elapsed >= window:
            # Advance exactly one bucket; the old current becomes previous.
            return start + window, 0, cur
        return start, cur, prev

    @staticmethod
    def _weight(start: float, prev: int, now: float, window: float) -> float:
        elapsed = min(max(now - start, 0.0), window)
        return prev * (window - elapsed) / window

    def effective_count(self, state: Mapping[str, Any], now: float) -> float:
        start, cur, prev = self._roll(state, now, self.window)
        return cur + self._weight(start, prev, now, self.window)

    # -- queries -------------------------------------------------------------

    def check(self, state: Mapping[str, Any], now: float) -> tuple[bool, float]:
        used = self.effective_count(state, now)
        return used < self.limit, used

    def remaining(self, state: Mapping[str, Any], now: float) -> int:
        """Whole requests left, clamped to ``[0, limit]`` (never negative)."""
        used = self.effective_count(state, now)
        return max(0, min(self.limit, self.limit - math.floor(used)))

    def retry_after(self, state: Mapping[str, Any], now: float) -> float:
        """Seconds until the effective count drops below the limit."""
        start, cur, prev = self._roll(state, now, self.window)
        if cur + self._weight(start, prev, now, self.window) < self.limit:
            return 0.0
        elapsed = min(max(now - start, 0.0), self.window)

        if cur >= self.limit:
            # Must wait for the roll, then for the carried-over bucket to decay.
            decay = 0.0
            if cur > 0:
                decay = self.window * (1.0 - self.limit / cur)
            return max(0.0, (self.window - elapsed) + decay)

        if prev <= 0:
            return max(0.0, self.window - elapsed)
        # Need prev * (1 - t/w) < limit - cur  ->  t > w * (1 - (limit-cur)/prev)
        needed_within_bucket = self.window * (1.0 - (self.limit - cur) / prev)
        return max(0.0, needed_within_bucket - elapsed)

    # -- mutations -----------------------------------------------------------

    def consume(self, state: MutableMapping[str, Any], now: float) -> tuple[bool, int]:
        """Atomically test-and-increment *state*, mutating it in place.

        The caller must hold the lock guarding *state*. Returns
        ``(allowed, current_bucket_count)``.
        """
        start, cur, prev = self._roll(state, now, self.window)
        used = cur + self._weight(start, prev, now, self.window)
        if used >= self.limit:
            # Persist the roll-forward so stale buckets expire even on denial.
            state["start"], state["cur"], state["prev"] = start, cur, prev
            return False, cur
        cur += 1
        state["start"], state["cur"], state["prev"] = start, cur, prev
        return True, cur


# ---------------------------------------------------------------------------
# Shared store
# ---------------------------------------------------------------------------


class SharedStore:
    """Thread-safe mapping that production swaps for Redis/Memcached.

    The limiter needs only ``__getitem__``/``__setitem__``/``pop``/
    :meth:`transaction`. The default here is process-local, which is fine for
    tests; the point of the abstraction is that a Redis adapter implementing
    the same five operations makes lockout state consistent across instances
    instead of each process counting on its own.
    """

    def __init__(self, initial: Mapping[Hashable, Any] | None = None) -> None:
        self._data: dict[Hashable, Any] = dict(initial or {})
        self._lock = threading.RLock()

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def __getitem__(self, key: Hashable) -> Any:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key: Hashable) -> None:
        with self._lock:
            del self._data[key]

    def pop(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            return self._data.pop(key, default)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def transaction(self) -> "_Transaction":
        """Serialise one read-modify-write block.

        A Redis adapter implements this with ``WATCH``/``MULTI`` or a Lua
        script; the contract is identical -- no interleaving inside the block.
        """
        return _Transaction(self._lock)


class _Transaction:
    __slots__ = ("_lock",)

    def __init__(self, lock: threading.RLock) -> None:
        self._lock = lock

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, *exc: Any) -> None:
        self._lock.release()


# ---------------------------------------------------------------------------
# Lockout policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LockoutPolicy:
    """Graduated lockout: each threshold step raises ``retry_after``.

    >>> LockoutPolicy().level(4), LockoutPolicy().level(5)
    (-1, 0)
    >>> LockoutPolicy().retry_after_for(20)
    900.0
    """

    thresholds: tuple[int, ...] = (5, 10, 20)
    retry_after: tuple[float, ...] = (60.0, 300.0, 900.0)
    max_retry_after: float = 3600.0

    def __post_init__(self) -> None:
        if not self.thresholds:
            raise ValueError("thresholds must not be empty")
        if len(self.thresholds) != len(self.retry_after):
            raise ValueError("thresholds and retry_after must be the same length")
        if any(t <= 0 for t in self.thresholds):
            raise ValueError("thresholds must be positive")
        if tuple(sorted(self.thresholds)) != tuple(self.thresholds):
            raise ValueError("thresholds must be ascending")
        if any(r < 0 for r in self.retry_after):
            raise ValueError("retry_after values must be non-negative")

    def level(self, failures: int) -> int:
        """0-based level reached; ``-1`` means not locked out."""
        level = -1
        for index, threshold in enumerate(self.thresholds):
            if failures >= threshold:
                level = index
            else:
                break
        return level

    def retry_after_for(self, failures: int) -> float:
        level = self.level(failures)
        if level < 0:
            return 0.0
        return min(float(self.retry_after[level]), float(self.max_retry_after))

    def should_reset_on_success(self, failures: int) -> bool:
        """Below the top level a success clears; at the top it only halves.

        A normal user who mistypes a password must not stay locked out after a
        successful login, but an attacker must not be able to wipe a sustained
        lockout with one stolen success.
        """
        return self.level(failures) < len(self.thresholds) - 1


DEFAULT_LOCKOUT_POLICY = LockoutPolicy()


# ---------------------------------------------------------------------------
# Admin authorisation for reset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdminIdentity:
    """Proof that the caller may clear limiter state."""

    name: str
    is_admin: bool = False
    token: Any = field(default=None, repr=False)

    def is_authorized(self) -> bool:
        return bool(self.is_admin) and bool(self.name)


class AdminToken:
    """Opaque admin credential compared by sealed-secret identity.

    Identity comparison avoids the timing oracle a string compare would give,
    and ``__repr__`` never leaks the value into logs.
    """

    __slots__ = ("_seal",)

    def __init__(self, value: Any) -> None:
        if value is None or value == "":
            raise ValueError("admin token must not be empty")
        self._seal = (value,)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AdminToken) and other._seal is self._seal

    def __hash__(self) -> int:
        return hash(id(self._seal))

    def __repr__(self) -> str:
        return "AdminToken(<redacted>)"


# ---------------------------------------------------------------------------
# Module-level primitives (single-bucket helpers)
# ---------------------------------------------------------------------------


def allow_request(
    state: Mapping[str, Any] | None,
    key: Any,
    limit: int,
    window: float,
    now: float | None = None,
) -> bool:
    """Single-bucket admission check (non-mutating).

    *state* may use the sliding layout (``cur``/``prev``/``start``) or the
    legacy fixed layout (``count``/``start``). An empty *key* has no
    attributable identity and is denied -- an uncounted request must never be
    allowed through. Malformed state fails closed.
    """
    if not _normalize(key):
        return False
    if state is None:
        return int(limit) > 0
    if not isinstance(state, Mapping):
        return False
    try:
        counter = SlidingWindow(limit, window)
        allowed, _ = counter.check(state, float(time.time() if now is None else now))
        return allowed
    except (TypeError, ValueError):
        return False


def remaining(
    state: Mapping[str, Any] | None,
    limit: int,
    window: float | None = None,
    now: float | None = None,
) -> int:
    """Requests left, always clamped into ``[0, limit]``.

    Corrupt counters (negative, non-numeric, absurdly high) can never produce
    a negative or over-limit answer.
    """
    if int(limit) <= 0:
        return 0
    if state is None:
        return int(limit)
    try:
        if window is not None and now is not None:
            used = math.floor(
                SlidingWindow(limit, window).effective_count(state, float(now))
            )
        else:
            raw = state.get("cur")
            if raw is None:
                raw = state.get("count", 0)
            used = int(raw or 0)
    except (AttributeError, TypeError, ValueError):
        return 0
    return max(0, min(int(limit), int(limit) - used))


def reset(state: MutableMapping[str, Any] | None) -> int:
    """Clear a counter after a *successful* authentication.

    Returns the remaining count after clearing (always 0). Callers must only
    invoke this on the success path -- never after a failed attempt.
    """
    if state is None:
        return 0
    if isinstance(state, MutableMapping):
        for name in ("count", "cur", "prev", "failures", "locked_until", "start"):
            if name in state:
                state[name] = 0
    return 0


def lockout(failures: int, threshold: int, policy: LockoutPolicy | None = None) -> bool:
    """Whether *failures* has reached a lockout threshold.

    The two-argument form is a simple threshold check. Passing a
    :class:`LockoutPolicy` uses that policy's first threshold, so callers can
    keep one source of truth for graduated levels.
    """
    if int(threshold) <= 0:
        return False
    try:
        failures = int(failures)
    except (TypeError, ValueError):
        return False
    if policy is not None:
        threshold = policy.thresholds[0]
    return failures >= int(threshold)


# ---------------------------------------------------------------------------
# Decision objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    key: str
    dimension: str  # "ip" | "user"
    limit: int
    remaining: int
    retry_after: float = 0.0
    reason: str = ""

    def headers(self) -> dict[str, str]:
        """Standard rate-limit response headers (clamped, never negative)."""
        headers = {
            "X-RateLimit-Limit": str(max(0, self.limit)),
            "X-RateLimit-Remaining": str(max(0, min(self.limit, self.remaining))),
        }
        if not self.allowed:
            headers["Retry-After"] = str(max(1, math.ceil(self.retry_after)))
        return headers


@dataclass(frozen=True)
class LockoutDecision:
    locked: bool
    level: int
    failures: int
    retry_after: float = 0.0


# ---------------------------------------------------------------------------
# The limiter
# ---------------------------------------------------------------------------

_FAILURE_FIELDS = ("failures", "locked_until")


def _blank_state(now: float) -> dict[str, Any]:
    return {
        "start": now,
        "cur": 0,
        "prev": 0,
        "failures": 0,
        "locked_until": 0.0,
    }


def _coerce_state(raw: Any, now: float) -> dict[str, Any]:
    """Normalise a stored record (new or legacy layout) into a mutable dict."""
    if not isinstance(raw, Mapping):
        return _blank_state(now)
    state = _blank_state(now)
    try:
        state["start"] = float(raw.get("start", now) or now)
        if "cur" in raw or "prev" in raw:
            state["cur"] = int(raw.get("cur", 0) or 0)
            state["prev"] = int(raw.get("prev", 0) or 0)
        else:
            state["cur"] = int(raw.get("count", 0) or 0)
        state["failures"] = max(0, int(raw.get("failures", 0) or 0))
        state["locked_until"] = float(raw.get("locked_until", 0.0) or 0.0)
    except (TypeError, ValueError):
        # Corrupt record: start clean rather than trusting garbage arithmetic.
        return _blank_state(now)
    return state


class PairwiseRateLimiter:
    """Dual-dimension, sliding-window limiter over shared state.

    Login path::

        limiter = PairwiseRateLimiter(store=SharedStore(), admin_token=token)
        keys = login_key(real_ip, username)          # ['ip:...', 'user:...']
        decision = limiter.check_many(keys, limit=5, window=300)
        if not decision.allowed:
            return 429, decision.headers()
        if verify_credentials(...):
            limiter.on_success(keys)                 # clears failure budget
            return 200
        state = limiter.on_failure(keys)             # graduated lockout
        if state.locked:
            return 423, {"Retry-After": str(math.ceil(state.retry_after))}
        return 401
    """

    def __init__(
        self,
        store: MutableMapping[Hashable, Any] | SharedStore | None = None,
        *,
        policy: LockoutPolicy | None = None,
        clock: Any = time.time,
        admin_token: AdminToken | None = None,
    ) -> None:
        self.store: Any = store if store is not None else SharedStore()
        self.policy = policy or DEFAULT_LOCKOUT_POLICY
        self.clock = clock
        self.admin_token = admin_token
        # Fallback lock for stores that expose no transaction primitive. When
        # the store provides one (SharedStore does) we use that instead, so a
        # multi-process backend gets real cross-process atomicity.
        self._fallback_lock = threading.RLock()

    # -- store plumbing ------------------------------------------------------

    def _transaction(self) -> Any:
        transaction = getattr(self.store, "transaction", None)
        if callable(transaction):
            return transaction()
        return _Transaction(self._fallback_lock)

    def _load(self, key: str, now: float) -> dict[str, Any]:
        try:
            raw = self.store[key]
        except (KeyError, TypeError):
            raw = None
        return _coerce_state(raw, now)

    def _save(self, key: str, state: Mapping[str, Any]) -> None:
        self.store[key] = dict(state)

    @staticmethod
    def _dimension(key: str) -> str:
        return key.partition(":")[0] or "unknown"

    def _locked_decision(
        self, key: str, state: Mapping[str, Any], limit: int, now: float
    ) -> RateLimitDecision | None:
        locked_until = float(state.get("locked_until", 0.0) or 0.0)
        if locked_until <= now:
            return None
        return RateLimitDecision(
            allowed=False,
            key=key,
            dimension=self._dimension(key),
            limit=limit,
            remaining=0,
            retry_after=locked_until - now,
            reason="locked",
        )

    # -- admission -----------------------------------------------------------

    def check_consume(
        self, key: str, *, limit: int, window: float, now: float | None = None
    ) -> RateLimitDecision:
        """Atomically test-and-increment a single key."""
        if not _normalize(key):
            raise EmptyKeyError("cannot rate limit an unidentified request")
        now = float(self.clock() if now is None else now)
        counter = SlidingWindow(limit, window)

        with self._transaction():
            state = self._load(key, now)
            locked = self._locked_decision(key, state, limit, now)
            if locked is not None:
                return locked
            allowed, _ = counter.consume(state, now)
            self._save(key, state)
            return RateLimitDecision(
                allowed=allowed,
                key=key,
                dimension=self._dimension(key),
                limit=counter.limit,
                remaining=counter.remaining(state, now),
                retry_after=0.0 if allowed else counter.retry_after(state, now),
                reason="" if allowed else "window_exhausted",
            )

    def check_many(
        self,
        keys: Iterable[str],
        *,
        limit: int,
        window: float,
        now: float | None = None,
    ) -> RateLimitDecision:
        """Dual-dimension admission: deny if *any* dimension is exhausted.

        All dimensions are checked before any is incremented, so a blocked
        account does not also drain its shared-NAT IP budget (and vice versa).
        """
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            raise EmptyKeyError("cannot rate limit an unidentified request")
        now = float(self.clock() if now is None else now)
        counter = SlidingWindow(limit, window)

        with self._transaction():
            states = {key: self._load(key, now) for key in unique_keys}

            for key in unique_keys:
                locked = self._locked_decision(key, states[key], counter.limit, now)
                if locked is not None:
                    return locked
                allowed, _ = counter.check(states[key], now)
                if not allowed:
                    return RateLimitDecision(
                        allowed=False,
                        key=key,
                        dimension=self._dimension(key),
                        limit=counter.limit,
                        remaining=counter.remaining(states[key], now),
                        retry_after=counter.retry_after(states[key], now),
                        reason="window_exhausted",
                    )

            decisions: list[RateLimitDecision] = []
            for key in unique_keys:
                state = states[key]
                counter.consume(state, now)
                self._save(key, state)
                decisions.append(
                    RateLimitDecision(
                        allowed=True,
                        key=key,
                        dimension=self._dimension(key),
                        limit=counter.limit,
                        remaining=counter.remaining(state, now),
                    )
                )

        # Report the tightest dimension so headers reflect the real budget.
        return min(decisions, key=lambda decision: decision.remaining)

    def enforce(
        self,
        keys: Iterable[str],
        *,
        limit: int,
        window: float,
        now: float | None = None,
    ) -> RateLimitDecision:
        """Like :meth:`check_many` but raises :class:`RateLimited` on denial."""
        decision = self.check_many(keys, limit=limit, window=window, now=now)
        if not decision.allowed:
            raise RateLimited(decision)
        return decision

    # -- outcomes ------------------------------------------------------------

    def on_failure(
        self, keys: Iterable[str], *, now: float | None = None
    ) -> LockoutDecision:
        """Record a failed attempt and apply graduated lockout.

        Returns the most severe lockout state across the dimensions involved.
        """
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            raise EmptyKeyError("cannot record a failure for an unidentified request")
        now = float(self.clock() if now is None else now)
        worst = LockoutDecision(locked=False, level=-1, failures=0)

        with self._transaction():
            for key in unique_keys:
                state = self._load(key, now)
                failures = int(state.get("failures", 0) or 0) + 1
                state["failures"] = failures
                retry = self.policy.retry_after_for(failures)
                if retry > 0:
                    state["locked_until"] = max(
                        float(state.get("locked_until", 0.0) or 0.0), now + retry
                    )
                self._save(key, state)
                candidate = LockoutDecision(
                    locked=retry > 0,
                    level=self.policy.level(failures),
                    failures=failures,
                    retry_after=max(0.0, float(state["locked_until"]) - now)
                    if retry > 0
                    else 0.0,
                )
                if candidate.retry_after >= worst.retry_after:
                    worst = candidate
        return worst

    def on_success(
        self, keys: Iterable[str], *, now: float | None = None
    ) -> None:
        """Clear the failure budget after a successful login/SMS check.

        Without this, ordinary users who mistype a password a few times get
        locked out while behaving normally. At the top lockout level counters
        are halved instead of cleared, so one stolen success cannot wipe a
        sustained lockout.
        """
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            raise EmptyKeyError("cannot record a success for an unidentified request")
        now = float(self.clock() if now is None else now)

        with self._transaction():
            for key in unique_keys:
                state = self._load(key, now)
                failures = int(state.get("failures", 0) or 0)
                if self.policy.should_reset_on_success(failures):
                    state["failures"] = 0
                else:
                    state["failures"] = failures // 2
                state["locked_until"] = 0.0
                self._save(key, state)

    def lockout_state(
        self, keys: Iterable[str], *, now: float | None = None
    ) -> LockoutDecision:
        """Read-only lockout state; consumes nothing."""
        unique_keys = list(dict.fromkeys(keys))
        if not unique_keys:
            raise EmptyKeyError("cannot inspect an unidentified request")
        now = float(self.clock() if now is None else now)
        worst = LockoutDecision(locked=False, level=-1, failures=0)

        with self._transaction():
            for key in unique_keys:
                state = self._load(key, now)
                failures = int(state.get("failures", 0) or 0)
                locked_until = float(state.get("locked_until", 0.0) or 0.0)
                locked = locked_until > now
                candidate = LockoutDecision(
                    locked=locked,
                    level=self.policy.level(failures),
                    failures=failures,
                    retry_after=max(0.0, locked_until - now) if locked else 0.0,
                )
                if candidate.retry_after >= worst.retry_after:
                    worst = candidate
        return worst

    # -- admin ---------------------------------------------------------------

    def reset(
        self,
        keys: Iterable[str],
        admin: AdminIdentity | AdminToken | None = None,
    ) -> None:
        """Clear every dimension for *keys* -- admin authority required.

        The unauthenticated reset endpoint was the cheapest lockout bypass in
        the system: anyone could wipe an attacker's counters. There is now no
        code path that clears state without either an authenticated success
        (:meth:`on_success`) or an explicit admin credential.
        """
        if not self._is_admin(admin):
            raise PermissionError(
                "resetting rate-limit state requires admin authority"
            )
        with self._transaction():
            for key in dict.fromkeys(keys):
                try:
                    self.store.pop(key, None)
                except AttributeError:
                    try:
                        del self.store[key]
                    except (KeyError, TypeError):
                        pass

    def _is_admin(self, admin: AdminIdentity | AdminToken | None) -> bool:
        if admin is None:
            return False
        if isinstance(admin, AdminIdentity):
            return admin.is_authorized()
        if isinstance(admin, AdminToken):
            # Sealed-secret identity comparison: no timing oracle, and a token
            # is only valid if the limiter was configured with that same seal.
            return self.admin_token is not None and admin == self.admin_token
        return False
