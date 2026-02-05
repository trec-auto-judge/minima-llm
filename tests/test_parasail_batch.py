"""Tests for Parasail batch mode - sentinel behavior and two-phase execution."""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from minima_llm import (
    MinimaLlmConfig,
    ParasailBatchConfig,
    MinimaLlmRequest,
    MinimaLlmResponse,
    BatchPendingResponse,
    OpenAIMinimaLlm,
)


class TestBatchPendingResponse:
    """Test the BatchPendingResponse sentinel type."""

    def test_sentinel_has_required_fields(self):
        """BatchPendingResponse should have request_id and cache_key."""
        sentinel = BatchPendingResponse(
            request_id="test-123",
            cache_key="abc123hash",
        )
        assert sentinel.request_id == "test-123"
        assert sentinel.cache_key == "abc123hash"

    def test_sentinel_is_frozen(self):
        """BatchPendingResponse should be immutable."""
        sentinel = BatchPendingResponse(
            request_id="test-123",
            cache_key="abc123hash",
        )
        with pytest.raises(AttributeError):
            sentinel.request_id = "changed"


class TestGenerateSentinelBehavior:
    """Test that generate() returns sentinel when batch collector is active."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create a minimal config with cache enabled."""
        return MinimaLlmConfig(
            base_url="http://localhost:8000/v1",
            model="test-model",
            cache_dir=str(tmp_path / "cache"),
        )

    @pytest.fixture
    def backend(self, config):
        """Create backend instance."""
        return OpenAIMinimaLlm(config)

    @pytest.fixture
    def llm_request(self):
        """Create a test request."""
        return MinimaLlmRequest(
            request_id="test-req-1",
            messages=[{"role": "user", "content": "Hello"}],
        )

    def test_generate_returns_sentinel_when_collector_active(self, backend, llm_request):
        """When _batch_collector is set, generate() should return BatchPendingResponse."""
        # Setup mock collector
        mock_collector = Mock()
        mock_collector.add_request = Mock()
        backend._batch_collector = mock_collector

        # Run generate
        result = asyncio.run(backend.generate(llm_request))

        # Should return sentinel, not MinimaLlmResponse
        assert isinstance(result, BatchPendingResponse)
        assert result.request_id == "test-req-1"
        assert result.cache_key is not None  # Should have computed cache key

    def test_generate_queues_request_in_collector(self, backend, llm_request):
        """generate() should call collector.add_request() with request and cache_key."""
        mock_collector = Mock()
        mock_collector.add_request = Mock()
        backend._batch_collector = mock_collector

        result = asyncio.run(backend.generate(llm_request))

        # Verify add_request was called
        mock_collector.add_request.assert_called_once()
        call_args = mock_collector.add_request.call_args
        assert call_args[0][0] == llm_request  # First arg is request
        assert isinstance(call_args[0][1], str)  # Second arg is cache_key

    def test_generate_does_not_make_http_call_in_batch_mode(self, backend, llm_request):
        """generate() should NOT make HTTP request when collector is active."""
        mock_collector = Mock()
        mock_collector.add_request = Mock()
        backend._batch_collector = mock_collector

        # Mock the HTTP method to track if it's called
        backend._post_json = AsyncMock()

        asyncio.run(backend.generate(llm_request))

        # HTTP should not be called
        backend._post_json.assert_not_called()

    def test_generate_returns_cached_before_checking_collector(self, backend, llm_request):
        """Cache hits should return before batch collection check."""
        # Pre-populate cache
        cache = backend._ensure_cache()
        cache_key = backend._make_cache_key(llm_request)
        cache.put(cache_key, "cached response", {"choices": []})

        # Setup collector
        mock_collector = Mock()
        mock_collector.add_request = Mock()
        backend._batch_collector = mock_collector

        # Run generate
        result = asyncio.run(backend.generate(llm_request))

        # Should return cached response, not sentinel
        assert isinstance(result, MinimaLlmResponse)
        assert result.text == "cached response"
        assert result.cached is True

        # Collector should NOT have been called
        mock_collector.add_request.assert_not_called()

    def test_generate_normal_mode_when_no_collector(self, backend, llm_request):
        """Without collector, generate() should make normal HTTP call."""
        # No collector set (normal mode)
        assert backend._batch_collector is None

        # Mock HTTP to avoid real network call
        mock_response = b'{"choices": [{"message": {"content": "Hello back"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}'

        with patch.object(backend, '_post_json', new_callable=AsyncMock) as mock_http:
            mock_http.return_value = (200, {}, mock_response)
            result = asyncio.run(backend.generate(llm_request))

        # Should return real response
        assert isinstance(result, MinimaLlmResponse)
        assert result.text == "Hello back"
        assert result.cached is False


class TestBatchModeContextManager:
    """Test the batch_mode() async context manager."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create config with parasail settings."""
        return MinimaLlmConfig(
            base_url="http://localhost:8000/v1",
            model="test-model",
            cache_dir=str(tmp_path / "cache"),
            parasail=ParasailBatchConfig(
                prefix="test-batch",
                state_dir=str(tmp_path / "state"),
            ),
        )

    @pytest.fixture
    def backend(self, config):
        """Create backend instance."""
        return OpenAIMinimaLlm(config)

    def test_batch_mode_sets_collector(self, backend):
        """Entering batch_mode should set _batch_collector."""
        async def _test():
            assert backend._batch_collector is None

            async with backend.batch_mode("test-prefix"):
                assert backend._batch_collector is not None
                assert backend._batch_prefix == "test-prefix"

            # After exit, should be cleared
            assert backend._batch_collector is None
            assert backend._batch_prefix is None

        asyncio.run(_test())

    def test_batch_mode_calls_submit_on_exit(self, backend):
        """Exiting batch_mode should call submit_and_wait if pending."""
        async def _test():
            with patch('minima_llm.batch.BatchCollector') as MockCollector:
                mock_instance = AsyncMock()
                MockCollector.return_value = mock_instance
                # has_pending is a sync method, not async
                mock_instance.has_pending = Mock(return_value=True)
                mock_instance.pending_count = 5

                async with backend.batch_mode("test-prefix"):
                    # Manually set the mock (since we patched the import)
                    backend._batch_collector = mock_instance

                # submit_and_wait should have been called
                mock_instance.submit_and_wait.assert_called_once()

        asyncio.run(_test())

    def test_batch_mode_skips_submit_if_no_pending(self, backend):
        """Exiting batch_mode should NOT call submit if no pending requests."""
        async def _test():
            with patch('minima_llm.batch.BatchCollector') as MockCollector:
                mock_instance = AsyncMock()
                MockCollector.return_value = mock_instance
                # has_pending is a sync method, not async
                mock_instance.has_pending = Mock(return_value=False)

                async with backend.batch_mode("test-prefix"):
                    backend._batch_collector = mock_instance

                # submit_and_wait should NOT have been called
                mock_instance.submit_and_wait.assert_not_called()

        asyncio.run(_test())


class TestBatchCollectorDownloadAndResolve:
    """Test BatchCollector._download_and_resolve actually populates cache."""

    @pytest.fixture
    def config(self, tmp_path):
        """Create config with cache enabled and directories created."""
        cache_dir = tmp_path / "cache"
        state_dir = tmp_path / "state"
        cache_dir.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)

        return MinimaLlmConfig(
            base_url="http://localhost:8000/v1",
            model="test-model",
            cache_dir=str(cache_dir),
            parasail=ParasailBatchConfig(
                prefix="test-batch",
                state_dir=str(state_dir),
            ),
        )

    def test_download_and_resolve_populates_cache(self, config, tmp_path):
        """Verify _download_and_resolve writes responses to backend's cache."""
        import json
        from minima_llm.batch import BatchCollector, BatchState

        # Create a backend to provide the cache
        backend = OpenAIMinimaLlm(config)

        # Mock JSONL response from Parasail
        jsonl_response = "\n".join([
            json.dumps({
                "custom_id": "test-batch-0",
                "response": {"body": {"choices": [{"message": {"content": "Response 1"}}]}}
            }),
            json.dumps({
                "custom_id": "test-batch-1",
                "response": {"body": {"choices": [{"message": {"content": "Response 2"}}]}}
            }),
        ])

        async def _test():
            # Create collector WITH backend
            collector = BatchCollector(config, "test-batch", backend=backend)

            # Add pending requests (must be inside async context)
            req1 = MinimaLlmRequest(request_id="req-1", messages=[{"role": "user", "content": "Hello"}])
            req2 = MinimaLlmRequest(request_id="req-2", messages=[{"role": "user", "content": "World"}])

            future1 = collector.add_request(req1, "cache-key-1")
            future2 = collector.add_request(req2, "cache-key-2")

            # Create mock batch state
            state = BatchState(
                prefix="test-batch",
                batch_id="batch-123",
                input_file_id="file-in",
                output_file_id="file-out",
                status="completed",
                created_at=0,
                custom_id_to_cache_key={"test-batch-0": "cache-key-1", "test-batch-1": "cache-key-2"},
            )

            # Mock _download_file to return our JSONL
            collector._download_file = AsyncMock(return_value=jsonl_response)

            result = await collector._download_and_resolve(state)

            assert result.completed_count == 2
            assert result.failed_requests == []

            # Verify cache was populated via backend's cache
            cache = backend._ensure_cache()

            cached1 = cache.get("cache-key-1")
            cached2 = cache.get("cache-key-2")

            assert cached1 is not None, "cache-key-1 not found in backend cache"
            assert cached1[0] == "Response 1"

            assert cached2 is not None, "cache-key-2 not found in backend cache"
            assert cached2[0] == "Response 2"

        asyncio.run(_test())

    def test_download_and_resolve_handles_errors(self, config, tmp_path):
        """Verify _download_and_resolve handles error responses."""
        import json
        from minima_llm.batch import BatchCollector, BatchState

        # Response with error
        jsonl_response = json.dumps({
            "custom_id": "test-batch-0",
            "error": {"code": "content_filter", "message": "Content filtered"}
        })

        async def _test():
            collector = BatchCollector(config, "test-batch")

            req1 = MinimaLlmRequest(request_id="req-1", messages=[{"role": "user", "content": "Hello"}])
            future1 = collector.add_request(req1, "cache-key-1")

            state = BatchState(
                prefix="test-batch",
                batch_id="batch-123",
                input_file_id="file-in",
                output_file_id="file-out",
                status="completed",
                created_at=0,
                custom_id_to_cache_key={"test-batch-0": "cache-key-1"},
            )

            collector._download_file = AsyncMock(return_value=jsonl_response)
            result = await collector._download_and_resolve(state)

            assert result.completed_count == 0
            assert len(result.failed_requests) == 1
            assert result.failed_requests[0][0] == "test-batch-0"

        asyncio.run(_test())

    def test_download_and_resolve_uses_backend_cache(self, config, tmp_path):
        """Verify _download_and_resolve uses backend's cache when provided."""
        import json
        from minima_llm.batch import BatchCollector, BatchState

        # Create a backend to get its cache
        backend = OpenAIMinimaLlm(config)

        # Mock JSONL response from Parasail
        jsonl_response = json.dumps({
            "custom_id": "test-batch-0",
            "response": {"body": {"choices": [{"message": {"content": "Backend cached"}}]}}
        })

        async def _test():
            # Create collector WITH backend
            collector = BatchCollector(config, "test-batch", backend=backend)

            req1 = MinimaLlmRequest(request_id="req-1", messages=[{"role": "user", "content": "Hello"}])
            future1 = collector.add_request(req1, "cache-key-backend")

            state = BatchState(
                prefix="test-batch",
                batch_id="batch-123",
                input_file_id="file-in",
                output_file_id="file-out",
                status="completed",
                created_at=0,
                custom_id_to_cache_key={"test-batch-0": "cache-key-backend"},
            )

            collector._download_file = AsyncMock(return_value=jsonl_response)
            result = await collector._download_and_resolve(state)

            assert result.completed_count == 1

            # Verify cache was populated via backend's cache
            cache = backend._ensure_cache()
            cached = cache.get("cache-key-backend")

            assert cached is not None, "cache-key-backend not found in backend cache"
            assert cached[0] == "Backend cached"

        asyncio.run(_test())


class TestParasailBatchConfig:
    """Test ParasailBatchConfig parsing."""

    def test_default_values(self):
        """Default config should have batch mode disabled."""
        config = ParasailBatchConfig()
        assert config.prefix is None  # Disabled
        assert config.state_dir is None
        assert config.poll_interval_s == 30.0
        assert config.max_poll_hours == 24.0

    def test_from_dict_with_prefix(self):
        """from_dict should parse prefix to enable batch mode."""
        config = ParasailBatchConfig.from_dict({
            "prefix": "my-batch-job",
            "poll_interval_s": 60,
        })
        assert config.prefix == "my-batch-job"
        assert config.poll_interval_s == 60.0

    def test_from_dict_empty(self):
        """from_dict with empty dict should return defaults."""
        config = ParasailBatchConfig.from_dict({})
        assert config.prefix is None

    def test_from_dict_none(self):
        """from_dict with None should return defaults."""
        config = ParasailBatchConfig.from_dict(None)
        assert config.prefix is None

    def test_minima_config_includes_parasail(self):
        """MinimaLlmConfig should have parasail field."""
        config = MinimaLlmConfig(
            base_url="http://localhost:8000/v1",
            model="test",
            parasail=ParasailBatchConfig(prefix="test"),
        )
        assert config.parasail.prefix == "test"