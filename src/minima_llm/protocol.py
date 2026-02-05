# protocol.py
"""Protocol definitions and data models for MinimaLlm."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

Json = Dict[str, Any]


@dataclass(frozen=True)
class MinimaLlmRequest:
    """
    Minimal request shape for a single LLM call.

    This stays intentionally stdlib-only so that participants can use their own
    frameworks (DSPy, LangChain, LiteLLM, raw HTTP, etc.) without dependency
    conflicts.

    Parameters
    ----------
    request_id:
        Stable identifier for this request (used for logging and error reporting).

    messages:
        OpenAI-compatible chat message list. Each message is a dict with keys:
        - role: "system" | "user" | "assistant" | ...
        - content: string

    temperature, max_tokens:
        Standard generation knobs. If None, the endpoint default is used.

    extra:
        Additional OpenAI-compatible parameters to forward verbatim (e.g., stop,
        top_p). Keep this policy-free and backend-agnostic.
    """

    request_id: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra: Optional[Json] = None


@dataclass(frozen=True)
class MinimaLlmResponse:
    """Result of a successful LLM call."""

    request_id: str
    text: str
    raw: Optional[Json] = None
    cached: bool = False  # True if returned from prompt cache
    input_tokens: int = 0  # Prompt tokens (from usage.prompt_tokens)
    output_tokens: int = 0  # Completion tokens (from usage.completion_tokens)



@dataclass(frozen=True)
class MinimaLlmFailure:
    request_id: str
    error_type: str
    message: str
    attempts: int
    status: Optional[int] = None
    body_snippet: Optional[str] = None
    timeout_s: Optional[float] = None
    attempt_timestamps: Tuple[float, ...] = ()

    def format_attempts(self) -> str:
        """Format attempt timestamps relative to first attempt."""
        if not self.attempt_timestamps:
            return ""
        t0 = self.attempt_timestamps[0]
        times = [f"+{t - t0:.1f}s" for t in self.attempt_timestamps]
        return f"[{', '.join(times)}]"


MinimaLlmResult = Union[MinimaLlmResponse, MinimaLlmFailure]


@dataclass(frozen=True)
class BatchPendingResponse:
    """
    Sentinel returned during batch collection phase.

    When Parasail batch mode is enabled, generate() returns this during the
    collection phase instead of blocking for an HTTP response. This signals
    that the request was queued for batch submission.

    Callers should NOT try to extract text from this - it's a marker that
    the real response will be available from cache after batch completion.

    Used in two-phase batch execution:
    1. Collection phase: generate() queues requests, returns BatchPendingResponse
    2. Submission phase: batch_mode() context exit submits and polls
    3. Retrieval phase: generate() returns from cache (100% hits)
    """
    request_id: str
    cache_key: str


@runtime_checkable
class AsyncMinimaLlmBackend(Protocol):
    """
    Minimal async backend interface.

    The harness and DSPy adapter only rely on these methods. Backends can
    implement rate limiting, retries, and backpressure internally.
    """

    async def generate(self, req: MinimaLlmRequest) -> MinimaLlmResult:
        """Perform one LLM call and return the generated text."""
        ...


    async def aclose(self) -> None:
        """Release any backend resources (sessions, files)."""
        ...
