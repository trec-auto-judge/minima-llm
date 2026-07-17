"""Tests for the deterministic retry temperature ramp."""

import asyncio
import json
import os

import pytest

os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:9/v1")
os.environ.setdefault("OPENAI_MODEL", "stub")

from minima_llm import MinimaLlmConfig, retry_temperature, set_retry_seed, reset_retry_seed
from minima_llm.backend import OpenAIMinimaLlm
from minima_llm.protocol import MinimaLlmRequest


def test_ramp_is_strictly_increasing_and_bounded():
    values = [retry_temperature(n, 0.5, 5.0) for n in range(1, 200)]
    assert all(b > a for a, b in zip(values, values[1:]))  # strictly increasing
    assert all(v < 0.5 for v in values)  # never reaches the asymptote
    assert values[0] == pytest.approx(0.5 / 6)  # 0.083...
    assert retry_temperature(5, 0.5, 5.0) == pytest.approx(0.25)  # half-ramp


def test_ramp_disabled_and_edge_cases():
    assert retry_temperature(0, 0.5, 5.0) == 0.0
    assert retry_temperature(-1, 0.5, 5.0) == 0.0
    assert retry_temperature(10, 0.0, 5.0) == 0.0  # max=0 disables


def _make_backend(**cfg_overrides):
    cfg = MinimaLlmConfig.from_dict({
        "base_url": "http://localhost:9/v1",
        "model": "stub",
        "cache_dir": None,
        "max_attempts": 1,
        **cfg_overrides,
    })
    return OpenAIMinimaLlm(cfg)


def _run_generate(backend, req, force_refresh=False, seed_attempt=None):
    """Run one generate() call against a stubbed transport; return the payload sent."""
    captured = {}

    async def fake_post(url, payload):
        captured.update(payload)
        body = json.dumps({
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        }).encode()
        return 200, {}, body

    backend._post_json = fake_post

    async def run():
        token = set_retry_seed(seed_attempt) if seed_attempt is not None else None
        try:
            return await backend.generate(req, force_refresh=force_refresh)
        finally:
            if token is not None:
                reset_retry_seed(token)

    asyncio.run(run())
    return captured


def test_first_attempt_stays_greedy():
    backend = _make_backend()
    payload = _run_generate(backend, MinimaLlmRequest(request_id="r", messages=[{"role": "user", "content": "hi"}]))
    assert "temperature" not in payload
    assert "seed" not in payload


def test_retry_gets_ramped_temperature():
    backend = _make_backend()
    payload = _run_generate(
        backend,
        MinimaLlmRequest(request_id="r", messages=[{"role": "user", "content": "hi"}]),
        force_refresh=True, seed_attempt=3,
    )
    assert payload["seed"] == 3
    assert payload["temperature"] == pytest.approx(retry_temperature(3, 0.5, 5.0))


def test_explicit_temperature_never_overridden():
    backend = _make_backend()
    payload = _run_generate(
        backend,
        MinimaLlmRequest(request_id="r", messages=[{"role": "user", "content": "hi"}], temperature=0.9),
        force_refresh=True, seed_attempt=3,
    )
    assert payload["temperature"] == 0.9


def test_disabled_via_config():
    backend = _make_backend(retry_temperature_max=0.0)
    payload = _run_generate(
        backend,
        MinimaLlmRequest(request_id="r", messages=[{"role": "user", "content": "hi"}]),
        force_refresh=True, seed_attempt=3,
    )
    assert "temperature" not in payload


def test_env_force_refresh_also_resamples():
    backend = _make_backend(force_refresh=True)
    payload = _run_generate(
        backend, MinimaLlmRequest(request_id="r", messages=[{"role": "user", "content": "hi"}]),
    )
    # no retry context: seed increments from 0 -> 1, ramp applies with n=1
    assert payload["seed"] == 1
    assert payload["temperature"] == pytest.approx(retry_temperature(1, 0.5, 5.0))
