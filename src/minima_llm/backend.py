# backend.py
"""OpenAI-compatible async LLM backend with caching, retries, and batch execution."""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import random
import sqlite3
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypeVar, Union, TYPE_CHECKING
import contextvars

from contextlib import asynccontextmanager

from .config import BatchConfig, MinimaLlmConfig
from .protocol import AsyncMinimaLlmBackend, BatchPendingResponse, Json, MinimaLlmFailure, MinimaLlmRequest, MinimaLlmResponse, MinimaLlmResult

if TYPE_CHECKING:
    from .batch import BatchCollector


# ----------------------------
# Backend Pulse (Diagnostics)
# ----------------------------

@dataclass
class BackendPulse:
    """Real-time diagnostics from backend operations.

    Collects all stats that can be measured at the backend level:
    - Request counts (cache hits, LLM calls)
    - Token usage (input/output)
    - Latency (min/max/avg)

    Auto-reset at the start of each batch (run_batched/run_batched_callable).
    The heartbeat polls this for display, computing rates from deltas.
    """
    # Request counts
    cache_hits: int = 0
    llm_calls: int = 0

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Latency tracking (in seconds)
    total_latency_s: float = 0.0
    min_latency_s: float = float('inf')
    max_latency_s: float = 0.0

    @property
    def done(self) -> int:
        """Total completed requests (cache hits + LLM calls)."""
        return self.cache_hits + self.llm_calls

    @property
    def avg_latency_s(self) -> float:
        """Average latency per LLM call in seconds."""
        return self.total_latency_s / self.llm_calls if self.llm_calls > 0 else 0.0


# Backwards compatibility alias
BackendStats = BackendPulse


# ----------------------------
# Rate Limit Info
# ----------------------------

@dataclass
class RateLimitInfo:
    """Parsed rate limit information from HTTP response.

    Extracted from headers (Retry-After, x-ratelimit-*) and response body.
    Used to configure cooldown/backpressure and display warnings.
    """
    # When to retry (seconds from now)
    retry_after_s: Optional[float] = None

    # Request limits
    limit_requests: Optional[int] = None
    remaining_requests: Optional[int] = None
    reset_requests_s: Optional[float] = None  # seconds until reset

    # Token limits
    limit_tokens: Optional[int] = None
    remaining_tokens: Optional[int] = None
    reset_tokens_s: Optional[float] = None  # seconds until reset

    # Error message from body
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    def summary(self) -> str:
        """Human-readable summary for warning messages."""
        parts = []
        if self.retry_after_s is not None:
            parts.append(f"retry_after={self.retry_after_s:.1f}s")
        if self.remaining_requests is not None and self.limit_requests is not None:
            parts.append(f"requests={self.remaining_requests}/{self.limit_requests}")
        if self.remaining_tokens is not None and self.limit_tokens is not None:
            parts.append(f"tokens={self.remaining_tokens}/{self.limit_tokens}")
        if self.reset_requests_s is not None:
            parts.append(f"reset_in={self.reset_requests_s:.0f}s")
        if self.error_message:
            # Truncate long messages
            msg = self.error_message[:80] + "..." if len(self.error_message) > 80 else self.error_message
            parts.append(f'msg="{msg}"')
        return " | ".join(parts) if parts else "(no rate limit info)"

    @property
    def suggested_delay_s(self) -> Optional[float]:
        """Best guess for how long to wait before retrying."""
        # Priority: explicit retry_after > reset time > None
        if self.retry_after_s is not None:
            return self.retry_after_s
        # Use the longer of request/token reset times
        resets = [r for r in [self.reset_requests_s, self.reset_tokens_s] if r is not None]
        return max(resets) if resets else None


def _parse_retry_after_value(value: str) -> Optional[float]:
    """Parse Retry-After header value (seconds or HTTP date)."""
    if not value:
        return None
    value = value.strip()

    # Try numeric first (most common)
    try:
        return float(value.rstrip('s'))  # Handle "30" or "30s"
    except ValueError:
        pass

    # Try HTTP date format (RFC 7231)
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(value)
        delay = (dt.timestamp() - time.time())
        return max(0.0, delay)
    except (ValueError, TypeError):
        pass

    return None


def _parse_reset_timestamp(value: str) -> Optional[float]:
    """Parse reset timestamp (Unix epoch or relative seconds)."""
    if not value:
        return None
    try:
        ts = float(value)
        # If it looks like a Unix timestamp (> year 2000), compute delta
        if ts > 946684800:  # 2000-01-01
            return max(0.0, ts - time.time())
        else:
            # Treat as relative seconds
            return ts
    except ValueError:
        return None


def _parse_rate_limit_info(
    status: int,
    headers: Dict[str, str],
    body: bytes,
) -> RateLimitInfo:
    """Extract rate limit info from HTTP response.

    Parses:
    - Retry-After header (seconds or HTTP date)
    - x-ratelimit-* headers (OpenAI/Anthropic style)
    - Error message from JSON body
    """
    info = RateLimitInfo()

    # Parse Retry-After header
    if ra := headers.get("retry-after"):
        info.retry_after_s = _parse_retry_after_value(ra)

    # Parse x-ratelimit-* headers (case-insensitive, already lowercased)
    if v := headers.get("x-ratelimit-limit-requests"):
        try:
            info.limit_requests = int(v)
        except ValueError:
            pass
    if v := headers.get("x-ratelimit-remaining-requests"):
        try:
            info.remaining_requests = int(v)
        except ValueError:
            pass
    if v := headers.get("x-ratelimit-reset-requests"):
        info.reset_requests_s = _parse_reset_timestamp(v)

    if v := headers.get("x-ratelimit-limit-tokens"):
        try:
            info.limit_tokens = int(v)
        except ValueError:
            pass
    if v := headers.get("x-ratelimit-remaining-tokens"):
        try:
            info.remaining_tokens = int(v)
        except ValueError:
            pass
    if v := headers.get("x-ratelimit-reset-tokens"):
        info.reset_tokens_s = _parse_reset_timestamp(v)

    # Parse error body for message
    if body and status >= 400:
        try:
            body_text = body.decode("utf-8", errors="replace")
            data = json.loads(body_text)
            if "error" in data:
                err = data["error"]
                if isinstance(err, dict):
                    info.error_message = err.get("message")
                    info.error_type = err.get("type") or err.get("code")
                elif isinstance(err, str):
                    info.error_message = err
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    return info


# Task-local flags for cache bypass and telemetry (safe for parallel async execution)
_force_refresh_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar('force_refresh', default=False)
_last_cached_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar('last_cached', default=False)


# Public API for adapter authors
def set_last_cached(cached: bool) -> None:
    """Call after generate() to record cache status for heartbeat tracking.

    Adapter authors should call this when their adapter unwraps MinimaLlmResponse
    and loses the cached attribute.
    """
    _last_cached_ctx.set(cached)


def get_last_cached() -> bool:
    """Get cache status from most recent generate() in this async task."""
    return _last_cached_ctx.get()

def set_force_refresh(force_refresh: bool)->contextvars.Token[bool]:
    """Call in generate() to force re-issuing the prompt (e.g. when response parsing failed)

    Adapter authors should call this when `acall` does not pass the `force_refresh` flag to the LLM backend.
    """
    return _force_refresh_ctx.set(force_refresh)

def reset_force_refresh(token:contextvars.Token[bool]):
    _force_refresh_ctx.reset(token)

def get_force_refresh() -> bool:
    """Get force_refresh requests for recent generate() in this async task."""
    return _force_refresh_ctx.get()


T = TypeVar("T")
R = TypeVar("R")


# ----------------------------
# Helpers: sleep + backoff
# ----------------------------

def _sleep_s(seconds: float) -> asyncio.Future:
    return asyncio.sleep(seconds)


def _jittered(base: float, jitter: float) -> float:
    if jitter <= 0:
        return base
    lo = max(0.0, 1.0 - jitter)
    hi = 1.0 + jitter
    return base * random.uniform(lo, hi)


# ----------------------------
# Pacing gate: simple rpm limiter
# ----------------------------

class RpmGate:
    """RPM limiter with dynamic adjustment based on rate limit headers.

    Supports both static configuration and dynamic updates from server
    rate limit responses (x-ratelimit-* headers).
    """

    def __init__(self, rpm: int):
        self._configured_rpm = rpm  # User-configured RPM (0 = unlimited)
        self._server_rpm: Optional[float] = None  # Learned from rate limit headers
        self._lock: Optional[asyncio.Lock] = None
        self._bound_loop: Optional[asyncio.AbstractEventLoop] = None
        self._next_ok = 0.0

    @property
    def effective_rpm(self) -> float:
        """Current effective RPM (min of configured and server-learned)."""
        if self._configured_rpm <= 0:
            # No configured limit - use server limit if known
            return self._server_rpm or 0
        if self._server_rpm is None:
            return float(self._configured_rpm)
        # Use the more restrictive of the two
        return min(float(self._configured_rpm), self._server_rpm)

    def _ensure_lock(self) -> asyncio.Lock:
        """Get or create lock for current event loop."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._bound_loop is not loop:
            self._lock = asyncio.Lock()
            self._bound_loop = loop
        return self._lock

    def update_from_rate_limit(self, rate_info: "RateLimitInfo") -> Optional[float]:
        """Update RPM based on rate limit info from server response.

        Computes effective RPM from x-ratelimit-limit-requests and reset time.
        Returns the new effective RPM if updated, None otherwise.
        """
        # Compute RPM from request limits
        if rate_info.limit_requests and rate_info.reset_requests_s:
            # requests per reset_period -> requests per minute
            new_rpm = (rate_info.limit_requests / rate_info.reset_requests_s) * 60.0

            # Only update if this is more restrictive or we don't have a value yet
            if self._server_rpm is None or new_rpm < self._server_rpm:
                old_rpm = self.effective_rpm
                self._server_rpm = new_rpm
                if old_rpm != self.effective_rpm:
                    return self.effective_rpm
        return None

    async def wait_turn(self) -> None:
        rpm = self.effective_rpm
        if rpm <= 0:
            return
        spacing = 60.0 / rpm
        async with self._ensure_lock():
            now = time.monotonic()
            if now < self._next_ok:
                await asyncio.sleep(self._next_ok - now)
                now = time.monotonic()
            self._next_ok = now + spacing


# ----------------------------
# Cooldown gate (global)
# ----------------------------

class Cooldown:
    """Cooldown gate with lazy per-loop lock initialization."""

    def __init__(self, floor_s: float, cap_s: float, halflife_s: float):
        self._floor = max(0.0, floor_s)
        self._cap = max(self._floor, cap_s)
        self._halflife = max(1e-6, halflife_s)

        self._lock: Optional[asyncio.Lock] = None
        self._bound_loop: Optional[asyncio.AbstractEventLoop] = None
        self._cooldown_s = 0.0
        self._last = time.monotonic()

    def _ensure_lock(self) -> asyncio.Lock:
        """Get or create lock for current event loop."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._bound_loop is not loop:
            self._lock = asyncio.Lock()
            self._bound_loop = loop
        return self._lock

    def _decay(self) -> None:
        now = time.monotonic()
        dt = max(0.0, now - self._last)
        self._last = now
        if self._cooldown_s <= 0.0:
            return
        # exponential decay with half-life
        decay = 0.5 ** (dt / self._halflife)
        self._cooldown_s *= decay
        if self._cooldown_s < self._floor:
            self._cooldown_s = 0.0

    async def wait_if_needed(self) -> None:
        async with self._ensure_lock():
            self._decay()
            cd = self._cooldown_s
        if cd > 0.0:
            await asyncio.sleep(cd)

    async def bump(self, suggested_s: float) -> None:
        async with self._ensure_lock():
            self._decay()
            new_cd = max(self._floor, suggested_s)
            self._cooldown_s = min(self._cap, max(self._cooldown_s, new_cd))


# ----------------------------
# HTTP helpers
# ----------------------------

def _json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _is_retriable_status(status: int) -> bool:
    return status in (408, 409, 425, 429, 500, 502, 503, 504)


def _is_overload_status(status: int) -> bool:
    return status in (429, 502, 503, 504)


# ----------------------------
# Prompt cache (SQLite-backed)
# ----------------------------

class PromptCache:
    """
    SQLite-backed prompt cache. Multi-process safe via WAL mode.

    This cache stores LLM responses keyed by a hash of the request parameters.
    Multiple processes can safely read/write concurrently.
    """

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response_text TEXT NOT NULL,
                response_raw TEXT,
                created_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def get(self, key: str) -> Optional[Tuple[str, Optional[Json]]]:
        """Retrieve cached response by key. Returns (text, raw_json) or None."""
        row = self._conn.execute(
            "SELECT response_text, response_raw FROM cache WHERE key = ?",
            (key,)
        ).fetchone()
        if row is None:
            return None
        raw = json.loads(row[1]) if row[1] else None
        return (row[0], raw)

    def put(self, key: str, text: str, raw: Optional[Json]) -> None:
        """Store response in cache."""
        raw_json = json.dumps(raw) if raw else None
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, response_text, response_raw, created_at) VALUES (?, ?, ?, ?)",
            (key, text, raw_json, time.time())
        )
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()


# ----------------------------
# Batch runner helpers
# ----------------------------

class _FailureCollector:
    def __init__(self, *, print_first_n: int, keep_last_n: int) -> None:
        self._print_first = max(0, int(print_first_n))
        self._keep = max(0, int(keep_last_n))
        self._seen = 0
        self._summaries: List[str] = []

    def record(self, f: MinimaLlmFailure) -> None:
        self._seen += 1
        # Include attempts and timeout in summary
        timeout_info = f", timeout={f.timeout_s}s" if f.timeout_s else ""
        msg = f"{f.request_id}: {f.error_type}: {f.message} (attempts={f.attempts}{timeout_info})"
        if self._seen <= self._print_first:
            ts = f.format_attempts()
            print(f"Failure {self._seen}: {f.request_id}")
            print(f"    {f.error_type}: {f.message}")
            print(f"    attempts={f.attempts}, timeout={f.timeout_s}s {ts}")
            if f.body_snippet:
                print(f"    body={f.body_snippet[:100]}")
        if self._keep > 0:
            self._summaries.append(msg)
            if len(self._summaries) > self._keep:
                self._summaries = self._summaries[-self._keep :]

    @property
    def count(self) -> int:
        return self._seen

    def summary_lines(self) -> List[str]:
        return list(self._summaries)

class _Heartbeat:
    def __init__(
        self,
        *,
        interval_seconds: float,
        stall_timeout_seconds: float,
        num_workers: int = 0,
        pulse_provider: Optional[Callable[[], BackendPulse]] = None,
    ) -> None:
        self._every_s = float(interval_seconds)
        self._stall_s = float(stall_timeout_seconds)
        self._num_workers = num_workers
        self._start = time.monotonic()
        self._last_done = self._start
        self._last_print = self._start

        # Per-interval LLM counters (excludes cache hits)
        self._interval_llm_sent = 0      # LLM requests sent this interval
        self._interval_llm_received = 0  # LLM responses received this interval

        # Item completion counter - tracks logical items, not raw operations.
        self._done = 0

        # Pulse provider for stats (polled from backend)
        self._pulse_provider = pulse_provider
        self._last_pulse: Optional[BackendPulse] = None

    @property
    def done(self) -> int:
        """Total completed items (logical completions, not raw operations)."""
        return self._done

    def mark_start(self) -> None:
        """Mark that a request has been sent (only counts LLM calls, not cache hits)."""
        self._interval_llm_sent += 1

    def mark_done(self, *, cached: bool = False) -> None:
        """Mark a request as done. Updates stall detection and rate counters."""
        self._done += 1
        self._last_done = time.monotonic()
        if cached:
            # It was cached, so it wasn't really sent - correct the mark_start() assumption
            self._interval_llm_sent -= 1
        else:
            self._interval_llm_received += 1

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        if seconds < 0 or not seconds < float("inf"):
            return "?"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:d}h{m:02d}m"
        if m > 0:
            return f"{m:d}m{s:02d}s"
        return f"{s:d}s"

    @staticmethod
    def _fmt_tokens(count: int) -> str:
        """Format token count with k/M suffix."""
        if count >= 1_000_000:
            return f"{count / 1_000_000:.1f}M"
        if count >= 1_000:
            return f"{count / 1_000:.1f}k"
        return str(count)

    def maybe_print(self, *, total: int, failed: int) -> None:
        now = time.monotonic()

        if self._every_s > 0 and (now - self._last_print) >= self._every_s:
            interval_s = now - self._last_print
            elapsed = now - self._start

            # Per-interval rates (from local counters)
            sent_rate = self._interval_llm_sent / interval_s if interval_s > 0 else 0
            recv_rate = self._interval_llm_received / interval_s if interval_s > 0 else 0

            # Reset interval counters
            self._interval_llm_sent = 0
            self._interval_llm_received = 0

            # Poll backend pulse for tokens, latency, cache hits, llm calls
            pulse: Optional[BackendPulse] = None
            if self._pulse_provider:
                pulse = self._pulse_provider()

            # Compute token rates from delta between polls
            in_tok_rate = 0.0
            out_tok_rate = 0.0
            if pulse and self._last_pulse:
                delta_in = pulse.input_tokens - self._last_pulse.input_tokens
                delta_out = pulse.output_tokens - self._last_pulse.output_tokens
                in_tok_rate = delta_in / interval_s if interval_s > 0 else 0
                out_tok_rate = delta_out / interval_s if interval_s > 0 else 0

            # Snapshot pulse for next interval
            if pulse:
                self._last_pulse = BackendPulse(
                    cache_hits=pulse.cache_hits,
                    llm_calls=pulse.llm_calls,
                    input_tokens=pulse.input_tokens,
                    output_tokens=pulse.output_tokens,
                    total_latency_s=pulse.total_latency_s,
                    min_latency_s=pulse.min_latency_s,
                    max_latency_s=pulse.max_latency_s,
                )

            # ETA estimation based on LLM calls only (not cache hits)
            llm_done = pulse.llm_calls if pulse else 0
            remaining = max(0, total - self.done)

            if llm_done > 0 and elapsed > 0:
                llm_rate = llm_done / elapsed  # Overall LLM rate
                eta_s = remaining / llm_rate if llm_rate > 0 else float("inf")
                eta_str = self._fmt_eta(eta_s)
            else:
                eta_str = "?"

            # Build token stats string (only if we have pulse provider with token data)
            tok_str = ""
            if pulse and (pulse.input_tokens > 0 or pulse.output_tokens > 0):
                lat_str = f" lat={pulse.avg_latency_s:.1f}s" if pulse.llm_calls > 0 else ""
                tok_str = (
                    f" | tok: in={self._fmt_tokens(int(in_tok_rate))}/s "
                    f"out={self._fmt_tokens(int(out_tok_rate))}/s "
                    f"({self._fmt_tokens(pulse.input_tokens)}/{self._fmt_tokens(pulse.output_tokens)})"
                    f"{lat_str}"
                )

            # Get cached count from pulse if available
            cached_count = pulse.cache_hits if pulse else 0

            print(
                f"[{elapsed:7.1f}s] "
                f"done={self.done}/{total} "
                f"sent={sent_rate:.1f}/s recv={recv_rate:.1f}/s "
                f"cached={cached_count} "
                f"failed={failed} "
                f"eta={eta_str}"
                f"{tok_str}"
            )
            self._last_print = now

        if self._stall_s > 0 and (now - self._last_done) >= self._stall_s:
            elapsed = now - self._start
            print(
                f"[{elapsed:7.1f}s] WARNING: "
                f"no completions for {now - self._last_done:.1f}s"
            )
            self._last_done = now  # avoid spamming



# ----------------------------
# Generic batch executor
# ----------------------------

async def run_batched_callable(
    items: List[T],
    async_callable: Callable[[T], Awaitable[R]],
    batch_config: Optional[BatchConfig] = None,
    pulse_provider: Optional[Callable[[], BackendPulse]] = None,
) -> List[Union[R, MinimaLlmFailure]]:
    """
    Execute a batch of async calls using the worker pool pattern.

    This is a generic async batch executor that works with any async callable.
    It maintains batching infrastructure: worker pool, queue, heartbeat, and
    failure tracking.

    Parameters
    ----------
    items : List[T]
        List of items to process
    async_callable : Callable[[T], Awaitable[R]]
        Async function to call for each item
    batch_config : BatchConfig
        Configuration for batch execution (num_workers, max_failures, etc.)
    pulse_provider : Optional[Callable[[], BackendPulse]]
        Optional callback to get backend pulse for heartbeat display

    Returns
    -------
    List[Union[R, MinimaLlmFailure]]
        Results in input order (success values or MinimaLlmFailure)
    """

    if batch_config is None:
        batch_config = BatchConfig.from_env()

    num_workers = max(1, int(batch_config.num_workers))
    hb = _Heartbeat(
        interval_seconds=batch_config.heartbeat_s,
        stall_timeout_seconds=batch_config.stall_s,
        num_workers=num_workers,
        pulse_provider=pulse_provider,
    )
    fc = _FailureCollector(
        print_first_n=batch_config.print_first_failures,
        keep_last_n=batch_config.keep_failure_summaries,
    )
    abort_event = asyncio.Event()  # Shared flag for early abort

    total = len(items)
    results: List[Optional[Union[R, MinimaLlmFailure]]] = [None] * total
    q: asyncio.Queue[Tuple[int, T]] = asyncio.Queue()

    for i, item in enumerate(items):
        q.put_nowait((i, item))

    async def worker() -> None:
        while True:
            # Check for early abort before taking next item
            if abort_event.is_set():
                return

            try:
                i, item = q.get_nowait()
            except asyncio.QueueEmpty:
                return

            # Reset contextvar for cache status
            set_last_cached(False)
            hb.mark_start()
            cached = False
            try:
                result = await async_callable(item)
                # Check if result is a failure (generate() returns Result, not raises)
                if isinstance(result, MinimaLlmFailure):
                    fc.record(result)
                # Check if result was from cache (MinimaLlmResponse has cached attr)
                else:
                    cached = bool(getattr(result, "cached", False)) or get_last_cached()
                results[i] = result
            except Exception as e:
                # Code errors (NameError, TypeError, etc.) propagate immediately
                if isinstance(e, (NameError, TypeError, AttributeError, SyntaxError, ImportError)):
                    raise
                # LLM and transport errors are recorded as failures
                f = MinimaLlmFailure(
                    request_id=f"input {i}/{item.request_id}" if hasattr(item, "request_id") else  f"input {i}",
                    error_type=type(e).__name__,
                    message=str(e),
                    attempts=1,
                )
                results[i] = f
                fc.record(f)

            hb.mark_done(cached=cached)
            q.task_done()

            # Check if we should trigger early abort after recording failure
            if batch_config.max_failures is not None and fc.count > batch_config.max_failures:
                abort_event.set()
                return

    async def heartbeat_loop() -> None:
        while True:
            hb.maybe_print(total=total, failed=fc.count)
            if hb.done >= total or abort_event.is_set():
                return
            await asyncio.sleep(hb._every_s)

    workers = [asyncio.create_task(worker()) for _ in range(max(1, int(batch_config.num_workers)))]
    hb_task = asyncio.create_task(heartbeat_loop())

    try:
        await asyncio.gather(*workers)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nInterrupted. Syncing cache and cleaning up...")
        for w in workers:
            w.cancel()
        raise
    finally:
        # Always cancel heartbeat cleanly after workers finish
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass

    # Early-abort policy (raised after cleanup)
    if abort_event.is_set():
        lines = fc.summary_lines()
        tail = "\n".join(f"  - {s}" for s in lines)
        raise RuntimeError(f"Aborting batch: {fc.count} failures\nRecent failures:\n{tail}")

    # All results are filled
    return [r for r in results if r is not None]


# ----------------------------
# Main backend
# ----------------------------

class OpenAIMinimaLlm(AsyncMinimaLlmBackend):
    """
    OpenAI-compatible backend using stdlib urllib.

    Dependency-light, provides caching, retries, error handling, etc.

    This backend supports being reused across multiple asyncio.run() calls.
    Async primitives (semaphore, locks) are lazily created per event loop.
    The cache database is automatically reopened if previously closed.
    """

    def __init__(self, cfg: MinimaLlmConfig):
        self.cfg = cfg

        # Semaphore and cache lock are lazy-init (bound per event loop)
        self._sem: Optional[asyncio.Semaphore] = None
        self._cache_lock: Optional[asyncio.Lock] = None
        self._bound_loop: Optional[asyncio.AbstractEventLoop] = None

        # RpmGate and Cooldown manage their own per-loop lock binding,
        # so we create them once and let them persist across event loops.
        self._rpm = RpmGate(cfg.rpm)
        self._cooldown = Cooldown(
            cfg.cooldown_floor_s,
            cfg.cooldown_cap_s,
            cfg.cooldown_halflife_s
        )

        self._closed = False

        b = cfg._normalize_base_url(cfg.base_url)
        self._has_v1 = b.endswith("/v1")
        self._base = b

        # Cache path (database opened lazily/reopened as needed)
        self._cache: Optional[PromptCache] = None
        self._cache_path: Optional[str] = None
        if cfg.cache_dir:
            os.makedirs(cfg.cache_dir, exist_ok=True)
            self._cache_path = os.path.join(cfg.cache_dir, "minima_llm.db")

        # Backend pulse - diagnostics that persist for lifetime of backend
        self._pulse = BackendPulse()

        # Overload tracking - shared across concurrent requests
        self._overload_warned = -1

        # Thread pool for HTTP requests
        self._executor: Optional[ThreadPoolExecutor] = None

        # Parasail batch mode state
        self._batch_collector: Optional["BatchCollector"] = None
        self._batch_prefix: Optional[str] = None

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Lazily create/recreate executor (e.g., after aclose())."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.cfg.max_outstanding)
        return self._executor

    def get_pulse(self) -> BackendPulse:
        """Return backend diagnostics pulse."""
        return self._pulse

    def reset_pulse(self) -> None:
        """Reset diagnostics (e.g., between batches if desired)."""
        self._pulse = BackendPulse()

    # Backwards compatibility
    get_stats = get_pulse
    reset_stats = reset_pulse

    def _ensure_async_resources(self) -> None:
        """Lazily create/recreate Semaphore and cache lock for current event loop."""
        loop = asyncio.get_running_loop()
        if self._bound_loop is not loop:
            self._sem = asyncio.Semaphore(self.cfg.max_outstanding)
            self._cache_lock = asyncio.Lock()
            self._bound_loop = loop
            self._closed = False

    def _ensure_cache(self) -> Optional[PromptCache]:
        """Ensure cache is open, reopen if previously closed."""
        if self._cache_path is None:
            return None
        if self._cache is None:
            self._cache = PromptCache(self._cache_path)
        return self._cache

    @classmethod
    def from_env(cls) -> "OpenAIMinimaLlm":
        """Construct backend from environment variables via MinimaLlmConfig."""
        return cls(MinimaLlmConfig.from_env())

    async def aclose(self) -> None:
        """Close cache database and executor. Backend can be reused - cache reopens automatically."""
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        if self._cache is not None:
            print(f"Synchronizing cache at {self.cfg.cache_dir}.")
            self._cache.close()
            self._cache = None
            print("Cache synched.")

    def batch_mode(self, prefix: str):
        """
        Async context manager for Parasail batch mode.

        When active, generate() queues requests and returns BatchPendingResponse
        sentinels instead of making HTTP calls. On context exit, the batch is
        submitted to Parasail, polled until complete, and results are written
        to cache.

        Usage:
            async with backend.batch_mode("my-judge-v1"):
                # Phase 1: Collection - generate() returns sentinels
                await run_dspy_batch(...)  # All requests queued

            # Phase 2: Submission happens here (context exit)
            # Batch uploaded, polled, results cached

            # Phase 3: Retrieval - run again, everything hits cache
            await run_dspy_batch(...)

        Args:
            prefix: Identifier for batch state files (for resumption)
        """
        # Import here to avoid circular import
        from .batch import BatchCollector

        @asynccontextmanager
        async def _batch_mode_context():
            self._batch_prefix = prefix
            self._batch_collector = BatchCollector(self.cfg, prefix, backend=self)

            # Check for completed batch BEFORE collection starts
            await self._batch_collector.populate_cache_if_completed()

            try:
                yield
            finally:
                # Submit batch on context exit if there are pending requests
                if self._batch_collector and self._batch_collector.has_pending():
                    print(f"\nSubmitting batch '{prefix}' ({self._batch_collector.pending_count} requests)...")
                    await self._batch_collector.submit_and_wait()
                    print(f"\nBatch complete. Results in cache.\n")

                self._batch_prefix = None
                self._batch_collector = None

        return _batch_mode_context()

    def _endpoint(self, path: str) -> str:
        if self._has_v1 and path.startswith("/v1/"):
            path = path[len("/v1") :]
        return self._base.rstrip("/") + path

    def _headers(self, *, body_is_gzip: bool) -> Dict[str, str]:
        h: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "minima-llm"
        }
        if self.cfg.api_key is not None:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        if body_is_gzip:
            h["Content-Encoding"] = "gzip"
        return h

    @staticmethod
    def _parse_retry_after(headers: Dict[str, str]) -> Optional[float]:
        """Extract Retry-After header as seconds (float)."""
        ra = headers.get("retry-after")
        if not ra:
            return None
        try:
            return float(ra)
        except ValueError:
            return None

    def _make_cache_key(self, req: MinimaLlmRequest) -> str:
        """Generate cache key from request parameters."""
        obj: Dict[str, Any] = {"model": self.cfg.model, "messages": req.messages}
        if req.temperature is not None:
            obj["temperature"] = req.temperature
        if req.max_tokens is not None:
            obj["max_tokens"] = req.max_tokens
        if req.extra:
            obj["extra"] = req.extra
        canonical = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    async def _post_json(self, url: str, payload: Json) -> Tuple[int, Dict[str, str], bytes]:
        """Perform a blocking urllib POST in a thread, to keep async-friendly."""
        body = _json_dumps(payload)
        body_is_gzip = bool(self.cfg.compress_gzip)
        if body_is_gzip:
            body = gzip.compress(body)

        req = urllib.request.Request(
            url=url,
            data=body,
            headers=self._headers(body_is_gzip=body_is_gzip),
            method="POST",
        )

        def _do() -> Tuple[int, Dict[str, str], bytes]:
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    headers = {k.lower(): v for k, v in resp.headers.items()}
                    return int(resp.status), headers, resp.read()
            except urllib.error.HTTPError as e:
                headers = {k.lower(): v for k, v in e.headers.items()} if e.headers else {}
                data = e.read() if e.fp is not None else b""
                return int(e.code), headers, data
            except urllib.error.URLError as e:
                return 408, {}, f"URLError: {e.reason}".encode()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._ensure_executor(), _do)

    async def generate(self, req: MinimaLlmRequest, *, force_refresh: bool = False) -> MinimaLlmResult:
        """
        Generate a response for the given request.

        Parameters
        ----------
        req : MinimaLlmRequest
            The request to process
        force_refresh : bool
            If True, bypass cache lookup and make a fresh LLM call.

        Returns MinimaLlmResponse on success, MinimaLlmFailure on error.
        """
        self._ensure_async_resources()

        # Check cache first (unless force_refresh)
        cache = self._ensure_cache()
        cache_key: Optional[str] = None
        if cache is not None:
            cache_key = self._make_cache_key(req)
            if not (force_refresh or self.cfg.force_refresh):
                async with self._cache_lock:  # type: ignore[union-attr]
                    cached = cache.get(cache_key)
                if cached is not None:
                    self._pulse.cache_hits += 1
                    set_last_cached(True)
                    return MinimaLlmResponse(request_id=req.request_id, text=cached[0], raw=cached[1], cached=True)

        # Batch collection mode: queue request and return sentinel
        if self._batch_collector is not None:
            if cache_key is None:
                cache_key = self._make_cache_key(req)
            self._batch_collector.add_request(req, cache_key)
            return BatchPendingResponse(  # type: ignore[return-value]
                request_id=req.request_id,
                cache_key=cache_key,
            )

        payload: Json = {
            "model": self.cfg.model,
            "messages": req.messages,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.extra:
            payload.update(req.extra)

        url = self._endpoint("/v1/chat/completions")

        attempt = 0
        attempt_timestamps: List[float] = []
        last_body: Optional[str] = None
        this_request_saw_overload = False

        while True:
            attempt += 1
            await self._cooldown.wait_if_needed()
            await self._rpm.wait_turn()

            async with self._sem:
                attempt_timestamps.append(time.monotonic())
                call_start = time.monotonic()
                status, headers, raw = await self._post_json(url, payload)
                call_latency = time.monotonic() - call_start

            body_text = raw.decode("utf-8", errors="replace")
            last_body = body_text[:300]

            if os.environ.get("MINIMA_DEBUG"):
                print(f"TRACE generate:\n - payload: {payload}\n - response: {body_text}")

            if 200 <= status < 300:
                try:
                    data = json.loads(body_text)
                except Exception as e:
                    return MinimaLlmFailure(
                        request_id=req.request_id,
                        error_type="JSONDecodeError",
                        message=f"non-JSON response: {e}",
                        attempts=attempt,
                        status=status,
                        body_snippet=last_body,
                        timeout_s=self.cfg.timeout_s,
                        attempt_timestamps=tuple(attempt_timestamps),
                    )

                try:
                    text = data["choices"][0]["message"]["content"]
                except Exception as e:
                    return MinimaLlmFailure(
                        request_id=req.request_id,
                        error_type="MalformedResponse",
                        message=f"missing expected fields: {e}",
                        attempts=attempt,
                        status=status,
                        body_snippet=last_body,
                        timeout_s=self.cfg.timeout_s,
                        attempt_timestamps=tuple(attempt_timestamps),
                    )

                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0) or 0
                output_tokens = usage.get("completion_tokens", 0) or 0

                if cache is not None and cache_key is not None:
                    async with self._cache_lock:  # type: ignore[union-attr]
                        cache.put(cache_key, str(text), data)

                if this_request_saw_overload and self._overload_warned >= 0:
                    count = self._overload_warned
                    self._overload_warned = -1
                    print(f"Server recovered after {count} retries. Resuming normal operation.")

                self._pulse.llm_calls += 1
                self._pulse.input_tokens += input_tokens
                self._pulse.output_tokens += output_tokens
                self._pulse.total_latency_s += call_latency
                self._pulse.min_latency_s = min(self._pulse.min_latency_s, call_latency)
                self._pulse.max_latency_s = max(self._pulse.max_latency_s, call_latency)

                set_last_cached(False)
                return MinimaLlmResponse(
                    request_id=req.request_id,
                    text=str(text),
                    raw=data,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            # non-2xx: parse rate limit info
            rate_info = _parse_rate_limit_info(status, headers, raw)
            retry_after = rate_info.suggested_delay_s

            new_rpm = self._rpm.update_from_rate_limit(rate_info)

            if _is_overload_status(status) or status == 408:
                cooldown_s = retry_after or self.cfg.cooldown_floor_s or 1.0
                await self._cooldown.bump(cooldown_s)
                this_request_saw_overload = True
                if self._overload_warned < 0:
                    if os.environ.get("MINIMA_DEBUG"):
                        print(f"DEBUG 429 headers: {headers}")
                        print(f"DEBUG 429 body: {body_text[:500]}")
                    info_str = rate_info.summary()
                    print(f"Server overload (HTTP {status}). {info_str}")
                    rpm_str = f" Adjusted RPM to {new_rpm:.0f}." if new_rpm else ""
                    print(f"  Retrying with cooldown={cooldown_s:.1f}s.{rpm_str} Press Ctrl-C to abort.")
                    self._overload_warned = 0
                self._overload_warned += 1

            if (self.cfg.max_attempts > 0 and attempt >= self.cfg.max_attempts) or not _is_retriable_status(status):
                error_type = "TimeoutError" if status == 408 else "HTTPError"
                msg = f"status={status}"
                if rate_info.error_message:
                    msg += f" | {rate_info.error_message}"
                return MinimaLlmFailure(
                    request_id=req.request_id,
                    error_type=error_type,
                    message=msg,
                    attempts=attempt,
                    status=status,
                    body_snippet=last_body,
                    timeout_s=self.cfg.timeout_s,
                    attempt_timestamps=tuple(attempt_timestamps),
                )

            if retry_after is not None:
                await asyncio.sleep(retry_after)
            else:
                backoff = min(self.cfg.max_backoff_s, self.cfg.base_backoff_s * (2 ** (attempt - 1)))
                await asyncio.sleep(_jittered(backoff, self.cfg.jitter))


    # ----------------------------
    # Batch runner
    # ----------------------------

    async def run_batched(self, requests: List[MinimaLlmRequest]) -> List[MinimaLlmResult]:
        """Execute a batch using the config's batch policy and return results."""
        self._print_batch_start(len(requests))
        self.reset_pulse()
        return await run_batched_callable(
            requests, self.generate, self.cfg.batch, pulse_provider=self.get_pulse
        )

    def _print_batch_start(self, total: int) -> None:
        """Print batch configuration at start and ensure executor is ready."""
        self._ensure_executor()
        workers = self.cfg.batch.num_workers
        outstanding = self.cfg.max_outstanding
        rpm = self.cfg.rpm
        rpm_str = f"rpm={rpm}" if rpm > 0 else "rpm=unlimited"
        print(f"Starting batch: {total} items | workers={workers} max_outstanding={outstanding} {rpm_str}")

    async def run_batched_callable(
        self,
        items: List[T],
        async_callable: Callable[[T], Awaitable[R]],
    ) -> List[Union[R, MinimaLlmFailure]]:
        """Execute a batch of async calls using the worker pool pattern."""
        self._print_batch_start(len(items))
        self.reset_pulse()
        return await run_batched_callable(
            items, async_callable, self.cfg.batch, pulse_provider=self.get_pulse
        )
