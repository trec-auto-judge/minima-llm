# Release Notes

## 0.2.7

### Cache lookup across model synonyms and lookup caches

Caches populated under different model names for the same model (for example `GPT-OSS-120b` and `openai/gpt-oss-120b`), or spread over several cache directories from endpoint switches, can be consolidated during a replay.

- `model_synonyms` (env `MODEL_SYNONYMS`, comma-separated; yaml list): model names treated as the same model as `model`. Lookup tries the key under every synonym.
- `cache_lookup_dirs` (env `CACHE_LOOKUP_DIRS`; yaml list): additional caches consulted read-only. Writes go to `cache_dir` only. A missing lookup database fails at backend construction.
- Across all synonym keys and all caches the row with the newest `created_at` wins. A hit that did not come from `cache_dir` under the effective model name is copied into `cache_dir` under the effective name with the row's original timestamp, so after one replay `cache_dir` holds the newest state and the synonyms and lookup dirs can be dropped. This copy replaces an older row under the effective name when one exists.
- `MinimaLlmResponse.cache_source` names where a cached reply came from (`primary`, `synonym:<name>`, `lookup:<dir>`, `lookup:<dir>:synonym:<name>`). The proxy prints it on every hit, `BackendPulse` counts `cache_hits_synonym` and `cache_hits_lookup`, and the `MINIMA_TRACE_FILE` line carries a `source` field (`miss` for a live call). During a replay a `llm call` line in the proxy output marks a cache hole.

### Cache rows store what was hashed

The sqlite table gains `model` and `canonical` columns (the exact JSON string hashed into `key`). Existing files are migrated once with `ALTER TABLE`; older rows keep NULL there. Rows written by the Parasail batch path do not fill the columns yet.

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