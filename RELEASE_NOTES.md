# Release Notes

## 0.2.3

### New: Proxy Mode (`minimallm-proxy`)

OpenAI-compatible HTTP proxy server that exposes minima-llm's caching, rate limiting, backpressure, and retry logic to any application.

- Start with `minimallm-proxy --port 8990`
- Any OpenAI-compatible client (DSPy, LangChain, litellm, curl) can point to `http://localhost:8990/v1` and benefit from prompt caching and rate limiting
- `--force-model` flag overrides client model with configured `OPENAI_MODEL`
- Non-streaming only; `GET /v1/models` also supported

### Per-request model override

`MinimaLlmRequest` now accepts an optional `model` field. When set, it overrides the configured model for that request in both the cache key and the upstream API call. This is used by the proxy to pass through client model selections.

## 0.2.2

Initial public release with core features:

- SQLite-backed prompt caching (WAL mode, multi-process safe)
- Async batch execution with worker pool, heartbeat, failure tracking
- RPM pacing with dynamic server-learned limits
- Exponential backoff with jitter and cooldown after overload
- Parasail batch API support
- DSPy adapter (optional extra)
- YAML config support (optional extra)
- Cache debug tracing via `MINIMA_TRACE_FILE`