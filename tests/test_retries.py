# tests/test_retries.py
"""
Tests for retry behavior with deterministic seeds.

Tests verify that:
1. Seeds are injected on force_refresh (1, 2, 3, ...)
2. retry_seed context variable is honored
3. HTTP retries reuse the same seed
"""
import asyncio
import json
from dataclasses import replace
from typing import Any, Dict, List
from unittest.mock import patch
import pytest

from minima_llm import (
    OpenAIMinimaLlm,
    MinimaLlmConfig,
    MinimaLlmRequest,
    set_force_refresh,
    reset_force_refresh,
    set_retry_seed,
    reset_retry_seed,
    get_retry_seed,
)


@pytest.fixture
def base_config():
    """Minimal config for testing (no real LLM calls)."""
    return MinimaLlmConfig(
        base_url="http://localhost:9999",
        api_key="test-key",
        model="test-model",
        max_outstanding=1,
        timeout_s=1.0,
        base_backoff_s=0.01,
        max_backoff_s=0.02,
        max_attempts=3,
        cache_dir=None,
    )


class TestBackendSeedInjection:
    """Test that backend injects seeds correctly on force_refresh."""

    def test_no_seed_without_force_refresh(self, base_config):
        """Without force_refresh, no seed should be injected."""
        backend = OpenAIMinimaLlm(base_config)
        payloads_sent: List[Dict[str, Any]] = []

        async def mock_post_json(url, payload):
            payloads_sent.append(payload.copy())
            return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )
                return await backend.generate(req, force_refresh=False)

        asyncio.run(run_test())

        assert len(payloads_sent) == 1
        assert "seed" not in payloads_sent[0]

    def test_seed_injected_with_force_refresh(self, base_config):
        """With force_refresh=True and no context seed, seed=1 should be injected."""
        backend = OpenAIMinimaLlm(base_config)
        payloads_sent: List[Dict[str, Any]] = []

        async def mock_post_json(url, payload):
            payloads_sent.append(payload.copy())
            return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )
                return await backend.generate(req, force_refresh=True)

        asyncio.run(run_test())

        assert len(payloads_sent) == 1
        assert payloads_sent[0].get("seed") == 1

    def test_seed_from_retry_seed_context(self, base_config):
        """With retry_seed context set, that seed should be used."""
        backend = OpenAIMinimaLlm(base_config)
        payloads_sent: List[Dict[str, Any]] = []

        async def mock_post_json(url, payload):
            payloads_sent.append(payload.copy())
            return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )
                token = set_retry_seed(5)
                try:
                    return await backend.generate(req, force_refresh=True)
                finally:
                    reset_retry_seed(token)

        asyncio.run(run_test())

        assert len(payloads_sent) == 1
        assert payloads_sent[0].get("seed") == 5

    def test_seed_increment_from_request_extra(self, base_config):
        """With seed in request.extra and no context seed, increment existing seed."""
        backend = OpenAIMinimaLlm(base_config)
        payloads_sent: List[Dict[str, Any]] = []

        async def mock_post_json(url, payload):
            payloads_sent.append(payload.copy())
            return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                    extra={"seed": 7},
                )
                return await backend.generate(req, force_refresh=True)

        asyncio.run(run_test())

        assert len(payloads_sent) == 1
        assert payloads_sent[0].get("seed") == 8

    def test_context_seed_overrides_request_seed(self, base_config):
        """Context seed takes precedence over request.extra seed."""
        backend = OpenAIMinimaLlm(base_config)
        payloads_sent: List[Dict[str, Any]] = []

        async def mock_post_json(url, payload):
            payloads_sent.append(payload.copy())
            return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                    extra={"seed": 100},
                )
                token = set_retry_seed(3)
                try:
                    return await backend.generate(req, force_refresh=True)
                finally:
                    reset_retry_seed(token)

        asyncio.run(run_test())

        assert len(payloads_sent) == 1
        assert payloads_sent[0].get("seed") == 3


class TestHttpRetries:
    """Test that HTTP retries (502, timeout) reuse the same seed."""

    def test_http_retries_keep_same_seed(self, base_config):
        """HTTP retries should send the same seed on each attempt."""
        config = replace(base_config, max_attempts=3)
        backend = OpenAIMinimaLlm(config)
        payloads_sent: List[Dict[str, Any]] = []
        call_count = 0

        async def mock_post_json(url, payload):
            nonlocal call_count
            call_count += 1
            payloads_sent.append(payload.copy())
            if call_count < 3:
                return (502, {}, b"Bad Gateway")
            return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )
                token = set_retry_seed(2)
                try:
                    return await backend.generate(req, force_refresh=True)
                finally:
                    reset_retry_seed(token)

        asyncio.run(run_test())

        assert call_count == 3
        seeds = [p.get("seed") for p in payloads_sent]
        assert seeds == [2, 2, 2], f"All HTTP retries should use same seed, got {seeds}"

    def test_http_retry_without_force_refresh_no_seed(self, base_config):
        """HTTP retries without force_refresh should not inject seed."""
        config = replace(base_config, max_attempts=3)
        backend = OpenAIMinimaLlm(config)
        payloads_sent: List[Dict[str, Any]] = []
        call_count = 0

        async def mock_post_json(url, payload):
            nonlocal call_count
            call_count += 1
            payloads_sent.append(payload.copy())
            if call_count < 3:
                return (502, {}, b"Bad Gateway")
            return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )
                return await backend.generate(req, force_refresh=False)

        asyncio.run(run_test())

        assert call_count == 3
        for p in payloads_sent:
            assert "seed" not in p


class TestRetryLoopSeeds:
    """Test that retry loops set incrementing seeds correctly."""

    def test_simulated_retry_loop_sets_incrementing_seeds(self, base_config):
        """Simulate a retry loop and verify seeds increment."""
        backend = OpenAIMinimaLlm(base_config)
        payloads_sent: List[Dict[str, Any]] = []

        async def mock_post_json(url, payload):
            payloads_sent.append(payload.copy())
            return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(backend, '_post_json', side_effect=mock_post_json):
                req = MinimaLlmRequest(
                    request_id="test-1",
                    messages=[{"role": "user", "content": "test"}],
                )

                # attempt=0: no force_refresh, no seed
                await backend.generate(req, force_refresh=False)

                # attempt=1: force_refresh + retry_seed=1
                token1 = set_retry_seed(1)
                fr_token1 = set_force_refresh(True)
                try:
                    await backend.generate(req, force_refresh=True)
                finally:
                    reset_retry_seed(token1)
                    reset_force_refresh(fr_token1)

                # attempt=2: force_refresh + retry_seed=2
                token2 = set_retry_seed(2)
                fr_token2 = set_force_refresh(True)
                try:
                    await backend.generate(req, force_refresh=True)
                finally:
                    reset_retry_seed(token2)
                    reset_force_refresh(fr_token2)

        asyncio.run(run_test())

        assert len(payloads_sent) == 3
        seeds = [p.get("seed") for p in payloads_sent]
        assert seeds == [None, 1, 2]


class TestContextIsolation:
    """Test context variable isolation between async tasks."""

    def test_retry_seed_context_isolation(self):
        """Verify retry_seed context is isolated per async task."""
        results: Dict[str, int] = {}

        async def task_with_seed(task_id: str, seed: int):
            token = set_retry_seed(seed)
            try:
                await asyncio.sleep(0.01)
                results[task_id] = get_retry_seed()
            finally:
                reset_retry_seed(token)

        async def run_test():
            await asyncio.gather(
                task_with_seed("a", 10),
                task_with_seed("b", 20),
                task_with_seed("c", 30),
            )

        asyncio.run(run_test())

        assert results == {"a": 10, "b": 20, "c": 30}

    def test_retry_seed_reset_after_exception(self):
        """Verify retry_seed is reset even when exception occurs."""

        async def run_test():
            token = set_retry_seed(99)
            try:
                raise ValueError("test error")
            finally:
                reset_retry_seed(token)

        with pytest.raises(ValueError):
            asyncio.run(run_test())

        assert get_retry_seed() == 0


class TestProxyEndToEnd:
    """End-to-end tests with a real proxy server."""

    def test_proxy_passes_seeds_end_to_end(self, base_config):
        """Start a real proxy and verify seeds pass through correctly."""
        import socket
        import urllib.request
        from unittest.mock import patch
        from minima_llm.proxy import ProxyServer

        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]

        proxy = ProxyServer(base_config, host="127.0.0.1", port=port)
        seeds_received: List[Any] = []

        async def mock_post_json(url, payload):
            seeds_received.append(payload.get("seed"))
            return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            # Patch the backend's _post_json to capture what proxy sends upstream
            with patch.object(proxy.backend, '_post_json', side_effect=mock_post_json):
                # Start proxy server
                server = await asyncio.start_server(
                    proxy._handle_connection,
                    host="127.0.0.1",
                    port=port,
                )

                async with server:
                    # Make requests with different seeds using raw HTTP
                    for seed in [None, 1, 2, 3]:
                        reader, writer = await asyncio.open_connection('127.0.0.1', port)

                        body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
                        if seed is not None:
                            body["seed"] = seed

                        body_bytes = json.dumps(body).encode()
                        request = (
                            f"POST /v1/chat/completions HTTP/1.1\r\n"
                            f"Host: 127.0.0.1:{port}\r\n"
                            f"Content-Type: application/json\r\n"
                            f"Content-Length: {len(body_bytes)}\r\n"
                            f"\r\n"
                        ).encode() + body_bytes

                        writer.write(request)
                        await writer.drain()

                        # Read response
                        response = await reader.read(4096)
                        writer.close()
                        await writer.wait_closed()

                        assert b"200" in response, f"Expected 200, got: {response[:100]}"

        asyncio.run(run_test())

        # Verify seeds were passed through
        assert seeds_received == [None, 1, 2, 3], f"Expected [None, 1, 2, 3], got {seeds_received}"

    def test_proxy_increments_seed_with_no_cache(self, base_config):
        """Verify proxy increments seed when no-cache pragma is set."""
        import socket
        from unittest.mock import patch
        from minima_llm.proxy import ProxyServer

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]

        proxy = ProxyServer(base_config, host="127.0.0.1", port=port)
        seeds_received: List[Any] = []

        async def mock_post_json(url, payload):
            seeds_received.append(payload.get("seed"))
            return (200, {}, b'{"choices":[{"message":{"content":"ok"}}]}')

        async def run_test():
            with patch.object(proxy.backend, '_post_json', side_effect=mock_post_json):
                server = await asyncio.start_server(
                    proxy._handle_connection,
                    host="127.0.0.1",
                    port=port,
                )

                async with server:
                    # Request with seed=5 and no-cache pragma
                    reader, writer = await asyncio.open_connection('127.0.0.1', port)

                    body = {
                        "model": "test",
                        "messages": [{"role": "user", "content": "hi"}],
                        "seed": 5,
                        "cache": {"no-cache": True},
                    }
                    body_bytes = json.dumps(body).encode()
                    request = (
                        f"POST /v1/chat/completions HTTP/1.1\r\n"
                        f"Host: 127.0.0.1:{port}\r\n"
                        f"Content-Type: application/json\r\n"
                        f"Content-Length: {len(body_bytes)}\r\n"
                        f"\r\n"
                    ).encode() + body_bytes

                    writer.write(request)
                    await writer.drain()

                    response = await reader.read(4096)
                    writer.close()
                    await writer.wait_closed()

                    assert b"200" in response

        asyncio.run(run_test())

        # With no-cache, seed should be incremented from 5 to 6
        assert seeds_received == [6], f"Expected [6], got {seeds_received}"


class TestDspyEndToEnd:
    """End-to-end tests with DSPy run_dspy_batch_generic."""

    @pytest.fixture
    def dspy_available(self):
        """Skip if DSPy not installed."""
        pytest.importorskip("dspy")
        return True

    def test_dspy_adapter_parse_failure(self, base_config, dspy_available):
        """Test DSPy internal adapter retry when response can't be parsed.

        When DSPy's TolerantChatAdapter can't find expected fields, DSPy tries
        another adapter format. This is DSPy's internal retry (2 calls per attempt),
        NOT our retry loop with incrementing seeds.
        """
        import dspy
        from pydantic import BaseModel
        from typing import Optional
        from minima_llm.dspy_adapter import run_dspy_batch_generic

        class GradeSignature(dspy.Signature):
            """Grade the answer on a scale of 0-5."""
            question: str = dspy.InputField(desc="The question")
            passage: str = dspy.InputField(desc="The passage to grade")
            grade: int = dspy.OutputField(desc="Grade from 0-5")

        class GradeData(BaseModel):
            question: str
            passage: str
            grade: int = 0
            reasoning: Optional[str] = None

        def convert_output(prediction: dspy.Prediction, data: GradeData) -> None:
            data.grade = int(prediction.grade)
            data.reasoning = getattr(prediction, 'reasoning', None)

        config = replace(base_config, max_attempts=5)

        payloads_sent: List[Dict[str, Any]] = []
        call_count = 0

        async def mock_post_json(self, url, payload):
            nonlocal call_count
            call_count += 1
            payloads_sent.append(payload.copy())

            # First 2 calls: unparseable response (triggers DSPy adapter switch)
            if call_count <= 2:
                return (200, {}, b'{"choices":[{"message":{"content":"thinking about it..."}}]}')

            # After that: valid response
            valid_content = (
                "[[ ## reasoning ## ]]\n"
                "Simple arithmetic: 2+2=4\n\n"
                "[[ ## grade ## ]]\n"
                "4\n\n"
                "[[ ## completed ## ]]"
            )
            response = {"choices": [{"message": {"content": valid_content}}]}
            return (200, {}, json.dumps(response).encode())

        test_data = [GradeData(question="What is 2+2?", passage="The answer is 4.")]

        with patch.object(OpenAIMinimaLlm, '_post_json', mock_post_json):
            results = run_dspy_batch_generic(
                test_data,
                GradeSignature,
                convert_output,
                config,
            )

        assert results[0].grade == 4
        # DSPy tries 2 adapters, so at least 2 calls on first attempt
        assert call_count >= 2, f"Expected at least 2 calls, got {call_count}"

    def test_converter_valueerror_triggers_retry_with_seeds(self, base_config, dspy_available):
        """Test that converter ValueError triggers our retry loop with incrementing seeds.

        When DSPy parses successfully but the local converter raises ValueError
        (e.g., grade out of range), our retry loop kicks in with force_refresh
        and incrementing seeds (1, 2, 3, ...).
        """
        import dspy
        from pydantic import BaseModel
        from typing import Optional
        from minima_llm.dspy_adapter import run_dspy_batch_generic

        class GradeSignature(dspy.Signature):
            """Grade the answer on a scale of 0-5."""
            question: str = dspy.InputField(desc="The question")
            passage: str = dspy.InputField(desc="The passage to grade")
            grade: int = dspy.OutputField(desc="Grade from 0-5")

        class GradeData(BaseModel):
            question: str
            passage: str
            grade: int = 0
            reasoning: Optional[str] = None

        def convert_output(prediction: dspy.Prediction, data: GradeData) -> None:
            grade_val = int(prediction.grade)
            # Validate range - raises ValueError if out of bounds
            if not (0 <= grade_val <= 5):
                raise ValueError(f"Grade {grade_val} out of range 0-5")
            data.grade = grade_val
            data.reasoning = getattr(prediction, 'reasoning', None)

        config = replace(base_config, max_attempts=5)

        payloads_sent: List[Dict[str, Any]] = []
        call_count = 0

        async def mock_post_json(self, url, payload):
            nonlocal call_count
            call_count += 1
            payloads_sent.append(payload.copy())

            # First few calls: valid int but out of range (converter raises ValueError)
            if call_count <= 4:
                invalid_content = (
                    "[[ ## reasoning ## ]]\n"
                    "I think the answer is excellent\n\n"
                    "[[ ## grade ## ]]\n"
                    "6\n\n"  # Valid int, but out of 0-5 range
                    "[[ ## completed ## ]]"
                )
                response = {"choices": [{"message": {"content": invalid_content}}]}
                return (200, {}, json.dumps(response).encode())

            # After retries: valid response with in-range grade
            valid_content = (
                "[[ ## reasoning ## ]]\n"
                "Simple arithmetic: 2+2=4\n\n"
                "[[ ## grade ## ]]\n"
                "4\n\n"
                "[[ ## completed ## ]]"
            )
            response = {"choices": [{"message": {"content": valid_content}}]}
            return (200, {}, json.dumps(response).encode())

        test_data = [GradeData(question="What is 2+2?", passage="The answer is 4.")]

        with patch.object(OpenAIMinimaLlm, '_post_json', mock_post_json):
            results = run_dspy_batch_generic(
                test_data,
                GradeSignature,
                convert_output,
                config,
            )

        assert results[0].grade == 4

        # Verify our retry loop was triggered with incrementing seeds
        seeds = [p.get("seed") for p in payloads_sent]

        # First attempt has no seed
        assert seeds[0] is None, f"First call should have no seed, got {seeds[0]}"

        # Subsequent retries should have incrementing seeds
        non_none_seeds = [s for s in seeds if s is not None]
        assert len(non_none_seeds) >= 2, f"Expected at least 2 retries with seeds, got {non_none_seeds}"
        assert 1 in non_none_seeds, f"Expected seed=1 in retries, got {non_none_seeds}"
        assert 2 in non_none_seeds, f"Expected seed=2 in retries, got {non_none_seeds}"

    def test_converter_valueerror_via_proxy_with_seeds(self, base_config, dspy_available):
        """Test that converter ValueError triggers retries with seeds through the proxy.

        Full end-to-end: DSPy -> Proxy -> mocked upstream
        Verifies incrementing seeds flow correctly through the proxy layer.
        """
        import socket
        import dspy
        from pydantic import BaseModel
        from typing import Optional
        from minima_llm.proxy import ProxyServer
        from minima_llm.dspy_adapter import run_dspy_batch_generic

        class GradeSignature(dspy.Signature):
            """Grade the answer on a scale of 0-5."""
            question: str = dspy.InputField(desc="The question")
            passage: str = dspy.InputField(desc="The passage to grade")
            grade: int = dspy.OutputField(desc="Grade from 0-5")

        class GradeData(BaseModel):
            question: str
            passage: str
            grade: int = 0
            reasoning: Optional[str] = None

        def convert_output(prediction: dspy.Prediction, data: GradeData) -> None:
            grade_val = int(prediction.grade)
            if not (0 <= grade_val <= 5):
                raise ValueError(f"Grade {grade_val} out of range 0-5")
            data.grade = grade_val
            data.reasoning = getattr(prediction, 'reasoning', None)

        # Find a free port for the proxy
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            proxy_port = s.getsockname()[1]

        # Proxy config (upstream backend)
        proxy_backend_config = replace(base_config, max_attempts=5)
        proxy = ProxyServer(proxy_backend_config, host="127.0.0.1", port=proxy_port)

        # Client config points to proxy
        client_config = replace(
            base_config,
            base_url=f"http://127.0.0.1:{proxy_port}",
            max_attempts=5,
        )

        seeds_at_upstream: List[Any] = []
        call_count = 0

        async def mock_upstream(url, payload):
            nonlocal call_count
            call_count += 1
            seeds_at_upstream.append(payload.get("seed"))

            # First few calls: valid int but out of range
            if call_count <= 4:
                invalid_content = (
                    "[[ ## reasoning ## ]]\n"
                    "Excellent work\n\n"
                    "[[ ## grade ## ]]\n"
                    "6\n\n"
                    "[[ ## completed ## ]]"
                )
                response = {"choices": [{"message": {"content": invalid_content}}]}
                return (200, {}, json.dumps(response).encode())

            # After retries: valid response
            valid_content = (
                "[[ ## reasoning ## ]]\n"
                "Good answer\n\n"
                "[[ ## grade ## ]]\n"
                "4\n\n"
                "[[ ## completed ## ]]"
            )
            response = {"choices": [{"message": {"content": valid_content}}]}
            return (200, {}, json.dumps(response).encode())

        test_data = [GradeData(question="What is 2+2?", passage="The answer is 4.")]

        async def run_test():
            # Patch the proxy's backend to capture what reaches upstream
            with patch.object(proxy.backend, '_post_json', side_effect=mock_upstream):
                server = await asyncio.start_server(
                    proxy._handle_connection,
                    host="127.0.0.1",
                    port=proxy_port,
                )

                async with server:
                    # Run DSPy batch through the proxy
                    # Need to run in executor since run_dspy_batch_generic is sync
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None,
                        lambda: run_dspy_batch_generic(
                            test_data,
                            GradeSignature,
                            convert_output,
                            client_config,
                        )
                    )
                    return results

        results = asyncio.run(run_test())

        assert results[0].grade == 4

        # Verify seeds reached upstream with incrementing values
        assert seeds_at_upstream[0] is None, f"First call should have no seed, got {seeds_at_upstream[0]}"

        non_none_seeds = [s for s in seeds_at_upstream if s is not None]
        assert len(non_none_seeds) >= 2, f"Expected at least 2 retries with seeds via proxy, got {non_none_seeds}"
        assert 1 in non_none_seeds, f"Expected seed=1 via proxy, got {non_none_seeds}"
        assert 2 in non_none_seeds, f"Expected seed=2 via proxy, got {non_none_seeds}"