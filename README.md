# minima-llm

Minimal async LLM backend with caching and batch execution.

## Features

- **Zero Dependencies**: Core package uses only Python stdlib (asyncio, urllib, sqlite3)
- **SQLite Cache**: Automatic prompt caching with WAL mode for multi-process safety
- **Batch Execution**: Worker pool pattern with heartbeat, failure tracking, and early abort
- **Rate Limiting**: RPM pacing with server-learned limits from rate limit headers
- **Retry Logic**: Exponential backoff with jitter, cooldown after overload
- **OpenAI Compatible**: Works with any OpenAI-compatible endpoint
- **DSPy Integration**: Optional adapter for DSPy framework (requires `[dspy]` extra)

## Installation

```bash
# Core only (no dependencies)
pip install minima-llm

# With DSPy support
pip install minima-llm[dspy]

# With YAML config support
pip install minima-llm[yaml]

# Development
pip install minima-llm[dev]
```

## Quick Start

### Basic Usage

```python
import asyncio
from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm, MinimaLlmRequest

async def main():
    # Configure from environment or explicit values
    config = MinimaLlmConfig(
        base_url="https://api.openai.com/v1",
        model="gpt-4",
        api_key="sk-...",
        cache_dir="./cache",
    )

    backend = OpenAIMinimaLlm(config)

    # Single request
    request = MinimaLlmRequest(
        request_id="q1",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0.0,
    )

    result = await backend.generate(request)
    print(result.text)

    await backend.aclose()

asyncio.run(main())
```

### Batch Execution

```python
import asyncio
from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm, MinimaLlmRequest

async def main():
    config = MinimaLlmConfig.from_env()
    backend = OpenAIMinimaLlm(config)

    requests = [
        MinimaLlmRequest(
            request_id=f"q{i}",
            messages=[{"role": "user", "content": f"Question {i}"}],
        )
        for i in range(100)
    ]

    # Run batch with progress heartbeat
    results = await backend.run_batched(requests)

    for r in results:
        if hasattr(r, 'text'):
            print(f"{r.request_id}: {r.text[:50]}...")

    await backend.aclose()

asyncio.run(main())
```

### With DSPy

```python
import asyncio
import dspy
from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm
from minima_llm.dspy_adapter import MinimaLlmDSPyLM

class QA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

async def main():
    config = MinimaLlmConfig.from_env()
    backend = OpenAIMinimaLlm(config)
    lm = MinimaLlmDSPyLM(backend)

    dspy.configure(lm=lm)

    predictor = dspy.ChainOfThought(QA)
    result = await predictor.acall(question="What is the capital of France?")
    print(result.answer)

    await backend.aclose()

asyncio.run(main())
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_BASE_URL` | API endpoint URL | (required) |
| `OPENAI_MODEL` | Model identifier | (required) |
| `OPENAI_API_KEY` | API key | None |
| `CACHE_DIR` | SQLite cache directory | None (disabled) |
| `BATCH_NUM_WORKERS` | Concurrent workers | 64 |
| `MAX_OUTSTANDING` | Max in-flight HTTP requests | 32 |
| `RPM` | Requests per minute (0=unlimited) | 600 |
| `TIMEOUT_S` | Per-request timeout | 60.0 |
| `MAX_ATTEMPTS` | Max retry attempts (0=infinite) | 6 |

### YAML Configuration

```yaml
base_url: "https://api.openai.com/v1"
model: "gpt-4"
api_key: "sk-..."
cache_dir: "./cache"

# Optional batch settings
batch:
  num_workers: 64
  max_failures: 25
  heartbeat_s: 10.0
```

Load with:

```python
config = MinimaLlmConfig.from_yaml("config.yml")
```

## Architecture

```
minima_llm/
├── protocol.py      # AsyncMinimaLlmBackend protocol, Request/Response types
├── config.py        # MinimaLlmConfig, BatchConfig, ParasailBatchConfig
├── backend.py       # OpenAIMinimaLlm - full async backend with cache
├── batch.py         # run_batched_callable, Parasail batch support
└── dspy_adapter.py  # MinimaLlmDSPyLM, TolerantChatAdapter (optional)
```

## Multi-Loop Support

The backend is designed to be reused across multiple `asyncio.run()` calls:

```python
backend = OpenAIMinimaLlm(config)

# First asyncio.run()
asyncio.run(batch1(backend))

# Second asyncio.run() - works correctly
asyncio.run(batch2(backend))
```

This is achieved through lazy per-loop initialization of async primitives.

## License

MIT
