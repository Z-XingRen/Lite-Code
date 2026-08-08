"""Provider retry classification, delay calculation, and cancellation-aware waits."""

import random
import socket
import time
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import RemoteDisconnected

from ..cancellation import CancellationRequested
from .errors import ProviderError


RETRYABLE_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
NON_RETRYABLE_PROVIDER_CODES = frozenset(
    {
        "auth_error",
        "authentication_error",
        "permission_denied",
        "forbidden",
        "invalid_request",
        "validation_error",
        "invalid_json",
        "tool_schema",
        "context_overflow",
        "context_length_exceeded",
        "request_too_large",
    }
)
TRANSIENT_PROVIDER_CODES = frozenset(
    {
        "rate_limited",
        "timeout",
        "network_error",
        "server_error",
        "temporarily_unavailable",
        "overloaded",
        "connection_reset",
    }
)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry budget and exponential backoff parameters."""

    max_retries: int = 2
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 30.0
    jitter_ratio: float = 0.1

    def __post_init__(self):
        if int(self.max_retries) < 0:
            raise ValueError("max_retries must be non-negative")
        if float(self.base_delay_seconds) < 0:
            raise ValueError("base_delay_seconds must be non-negative")
        if float(self.max_delay_seconds) < 0:
            raise ValueError("max_delay_seconds must be non-negative")
        if not 0 <= float(self.jitter_ratio) <= 1:
            raise ValueError("jitter_ratio must be between zero and one")


DEFAULT_RETRY_POLICY = RetryPolicy()


@dataclass(frozen=True)
class RetryResult:
    value: object
    attempts: int
    retry_count: int
    history: tuple[dict, ...] = ()


class RetryExhausted(RuntimeError):
    """A retry budget ended with the original provider failure."""

    def __init__(self, cause, *, attempts, retry_count, history):
        super().__init__(str(cause))
        self.cause = cause
        self.attempts = int(attempts)
        self.retry_count = int(retry_count)
        self.history = tuple(dict(item) for item in history)


def classify_retry(error):
    """Return a stable retry reason, or ``None`` for deterministic failures."""

    status = _status(error)
    if status in {401, 403}:
        return None
    if status in RETRYABLE_HTTP_STATUS:
        if status == 429:
            return "rate_limited"
        if status == 408:
            return "timeout"
        if status >= 500:
            return "server_error"
        return "transient_http"
    code = _error_code(error)
    if code in NON_RETRYABLE_PROVIDER_CODES:
        return None
    if code in TRANSIENT_PROVIDER_CODES:
        return code
    if (
        isinstance(error, ProviderError) and error.retryable
    ) or (isinstance(error, dict) and bool(error.get("retryable"))):
        return "provider_retryable"
    if isinstance(error, (TimeoutError, socket.timeout)):
        return "timeout"
    if isinstance(error, urllib.error.HTTPError):
        return None
    if isinstance(error, (urllib.error.URLError, RemoteDisconnected, ConnectionError)):
        return "network_error"
    return None


def is_retryable(error):
    return classify_retry(error) is not None


def retry_after_seconds(headers, *, now=None):
    """Parse ``retry-after-ms``/``Retry-After`` as a non-negative delay.

    Numeric values are seconds except for the explicitly millisecond header.
    HTTP-date values are evaluated against the injectable ``now`` clock.
    Invalid and negative values return ``None`` so exponential backoff applies.
    """

    milliseconds = _header_value(headers, "retry-after-ms")
    if milliseconds is not None:
        try:
            value = float(milliseconds) / 1000.0
        except (TypeError, ValueError):
            value = None
        if value is not None and value >= 0:
            return value

    value = _header_value(headers, "retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = None
    if seconds is not None:
        return seconds if seconds >= 0 else None
    try:
        parsed = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    current = time.time() if now is None else now
    if isinstance(current, datetime):
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.timestamp()
    delay = parsed.timestamp() - float(current)
    return delay if delay >= 0 else 0.0


def calculate_retry_delay(
    attempt,
    headers=None,
    *,
    policy=None,
    now=None,
    random_fn=None,
):
    """Calculate one bounded exponential delay with optional jitter."""

    policy = policy or DEFAULT_RETRY_POLICY
    retry_after = retry_after_seconds(headers, now=now)
    if retry_after is not None:
        return min(float(retry_after), float(policy.max_delay_seconds))
    exponent = max(0, int(attempt))
    delay = min(
        float(policy.max_delay_seconds),
        float(policy.base_delay_seconds) * (2**exponent),
    )
    if delay <= 0 or policy.jitter_ratio <= 0:
        return delay
    random_fn = random_fn or random.random
    spread = (float(random_fn()) * 2.0) - 1.0
    return max(
        0.0,
        min(
            float(policy.max_delay_seconds),
            delay * (1.0 + spread * float(policy.jitter_ratio)),
        ),
    )


def cancellable_backoff(delay, cancellation_token=None, *, sleep_fn=None):
    """Wait for a retry delay, allowing a run token to interrupt it."""

    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
        if delay > 0:
            cancellation_token.wait(float(delay))
        cancellation_token.raise_if_cancelled()
        return
    if delay > 0:
        (sleep_fn or time.sleep)(float(delay))


def run_with_retries(
    operation,
    *,
    policy=None,
    cancellation_token=None,
    sleep_fn=None,
    clock=None,
    random_fn=None,
    on_retry=None,
):
    """Run an operation under one bounded, observable retry policy."""

    policy = policy or DEFAULT_RETRY_POLICY
    retry_count = 0
    attempts = 0
    history = []
    clock = clock or time.time
    while True:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        attempts += 1
        try:
            value = operation(attempts)
        except CancellationRequested:
            raise
        except Exception as exc:
            reason = classify_retry(exc)
            if reason is None or retry_count >= int(policy.max_retries):
                raise RetryExhausted(
                    exc,
                    attempts=attempts,
                    retry_count=retry_count,
                    history=history,
                ) from exc
            delay = calculate_retry_delay(
                retry_count,
                _headers(exc),
                policy=policy,
                now=clock(),
                random_fn=random_fn,
            )
            retry_count += 1
            event = {
                "attempt": attempts,
                "retry_count": retry_count,
                "reason": reason,
                "delay_seconds": delay,
            }
            history.append(event)
            if on_retry is not None:
                on_retry(dict(event))
            try:
                cancellable_backoff(
                    delay,
                    cancellation_token,
                    sleep_fn=sleep_fn,
                )
            except CancellationRequested:
                raise
            finally:
                if hasattr(exc, "close") and callable(exc.close):
                    exc.close()
            continue
        return RetryResult(value, attempts, retry_count, tuple(history))


def _header_value(headers, name):
    if not headers:
        return None
    try:
        value = headers.get(name)
    except AttributeError:
        value = None
    if value is not None:
        return value
    wanted = str(name).casefold()
    try:
        for key, value in headers.items():
            if str(key).casefold() == wanted:
                return value
    except AttributeError:
        return None
    return None


def _headers(error):
    return getattr(error, "headers", None)


def _status(error):
    if isinstance(error, dict):
        value = error.get("http_status", error.get("status"))
    else:
        value = getattr(error, "code", None)
        if not isinstance(value, int):
            value = getattr(error, "http_status", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _error_code(error):
    if isinstance(error, dict):
        value = error.get("code", error.get("type"))
        if not value and isinstance(error.get("error"), dict):
            nested = error["error"]
            value = nested.get("code", nested.get("type"))
    else:
        value = getattr(error, "code", "")
    code = str(value or "").strip().lower().replace("-", "_")
    if "rate" in code or "throttl" in code:
        return "rate_limited"
    if "overload" in code or "unavailable" in code:
        return "overloaded"
    if "timeout" in code:
        return "timeout"
    if "auth" in code or "permission" in code or "forbidden" in code:
        return "auth_error"
    if "context" in code or "length" in code or "token" in code:
        return "context_overflow"
    if "invalid" in code or "validation" in code or "schema" in code:
        return "invalid_request"
    if "server" in code or "internal" in code:
        return "server_error"
    return code
