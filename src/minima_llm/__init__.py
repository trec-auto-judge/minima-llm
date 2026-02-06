"""MinimaLLM: Minimal async LLM backend with batching support.

Core package has zero dependencies (stdlib only).
Optional DSPy integration available with `pip install minima-llm[dspy]`.
"""

from .config import BatchConfig, MinimaLlmConfig, ParasailBatchConfig
from .protocol import (
    AsyncMinimaLlmBackend,
    BatchPendingResponse,
    MinimaLlmRequest,
    MinimaLlmResponse,
    MinimaLlmFailure,
    MinimaLlmResult,
)
from .backend import (
    OpenAIMinimaLlm,
    run_batched_callable,
    set_last_cached,
    get_last_cached,
    set_force_refresh,
    get_force_refresh,
    reset_force_refresh,
    BackendPulse,
    BackendStats,  # Backwards compatibility alias
    PromptCache,
    RpmGate,
    Cooldown,
)
from .batch import (
    batch_status_overview,
    cancel_batch,
    cancel_all_batches,
    cancel_all_local_batches,
)

__all__ = [
    # Config
    "MinimaLlmConfig",
    "BatchConfig",
    "ParasailBatchConfig",
    # Protocol and data types
    "AsyncMinimaLlmBackend",
    "MinimaLlmRequest",
    "MinimaLlmResponse",
    "MinimaLlmFailure",
    "MinimaLlmResult",
    "BatchPendingResponse",
    # Backend
    "OpenAIMinimaLlm",
    "run_batched_callable",
    # Diagnostics
    "BackendPulse",
    "BackendStats",
    # Cache utilities
    "set_last_cached",
    "get_last_cached",
    "set_force_refresh",
    "get_force_refresh",
    "reset_force_refresh",
    # Internal (for advanced use)
    "PromptCache",
    "RpmGate",
    "Cooldown",
    # Batch management CLI
    "batch_status_overview",
    "cancel_batch",
    "cancel_all_batches",
    "cancel_all_local_batches",
]

__version__ = "0.1.0"
