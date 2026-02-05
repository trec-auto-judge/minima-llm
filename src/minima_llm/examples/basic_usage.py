#!/usr/bin/env python3
"""Basic usage example for minima-llm."""
import asyncio
from typing import List

from minima_llm import (
    MinimaLlmConfig,
    MinimaLlmRequest,
    MinimaLlmResponse,
    MinimaLlmFailure,
    OpenAIMinimaLlm,
)


# --------------------
# Build a batch of requests
# --------------------

requests = [
    MinimaLlmRequest(
        request_id=f"q{i}",
        messages=[
            {"role": "system", "content": "You are a fair and careful evaluator."},
            {"role": "user", "content": f"Evaluate answer #{i} for correctness."},
        ],
        temperature=0.0,
    )
    for i in range(100)
]


# --------------------
# Run the batch
# --------------------

async def main() -> None:
    backend = OpenAIMinimaLlm(MinimaLlmConfig.from_env())

    # Always log the effective configuration for long runs
    print(backend.cfg.describe())
    print("-" * 60)

    try:
        results = await backend.run_batched(requests)
    except RuntimeError as e:
        # Batch-level abort (e.g., too many failures)
        print(f"Batch aborted: {e}")
        return

    # Separate successes from failures
    ok: List[MinimaLlmResponse] = []
    failed: List[MinimaLlmFailure] = []

    for r in results:
        if isinstance(r, MinimaLlmResponse):
            ok.append(r)
        else:
            failed.append(r)

    print(f"Completed: {len(ok)}")
    print(f"Failed:    {len(failed)}")

    # Example: print first few judgments
    for r in ok[:3]:
        print(r.request_id, ": ", r.text[:80])


if __name__ == "__main__":
    asyncio.run(main())
