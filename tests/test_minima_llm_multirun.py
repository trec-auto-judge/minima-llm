# tests/test_minima_llm_multirun.py
"""
Tests for MinimaLLM behavior across multiple asyncio.run() calls.

These tests ensure that the lazy per-loop initialization of async primitives
(Semaphore, Lock) and the cache auto-reopen behavior work correctly when
the backend is reused across separate event loops.

NOTE: These tests do NOT require a real LLM endpoint.
"""

import asyncio
import pytest

from minima_llm import (
    MinimaLlmConfig,
    OpenAIMinimaLlm,
)
from minima_llm.backend import (
    RpmGate,
    Cooldown,
    PromptCache,
)


class TestRpmGateMultiLoop:
    """Test RpmGate lazy lock initialization across event loops."""

    def test_rpm_gate_works_across_asyncio_runs(self):
        """RpmGate should work when used in multiple asyncio.run() calls."""
        gate = RpmGate(rpm=60)

        async def use_gate():
            await gate.wait_turn()
            return True

        result1 = asyncio.run(use_gate())
        assert result1 is True

        result2 = asyncio.run(use_gate())
        assert result2 is True

    def test_rpm_gate_lock_recreated_for_new_loop(self):
        """RpmGate should recreate its lock for a new event loop."""
        gate = RpmGate(rpm=60)

        async def get_lock_id():
            await gate.wait_turn()
            return id(gate._lock)

        lock_id_1 = asyncio.run(get_lock_id())
        lock_id_2 = asyncio.run(get_lock_id())

        assert lock_id_1 != lock_id_2

    def test_rpm_gate_disabled_works_across_runs(self):
        """RpmGate with rpm=0 (disabled) should work across runs."""
        gate = RpmGate(rpm=0)

        async def use_gate():
            await gate.wait_turn()
            return True

        result1 = asyncio.run(use_gate())
        result2 = asyncio.run(use_gate())
        assert result1 is True
        assert result2 is True


class TestCooldownMultiLoop:
    """Test Cooldown lazy lock initialization across event loops."""

    def test_cooldown_works_across_asyncio_runs(self):
        """Cooldown should work when used in multiple asyncio.run() calls."""
        cooldown = Cooldown(floor_s=0.0, cap_s=10.0, halflife_s=5.0)

        async def use_cooldown():
            await cooldown.wait_if_needed()
            await cooldown.bump(0.1)
            return True

        result1 = asyncio.run(use_cooldown())
        assert result1 is True

        result2 = asyncio.run(use_cooldown())
        assert result2 is True

    def test_cooldown_lock_recreated_for_new_loop(self):
        """Cooldown should recreate its lock for a new event loop."""
        cooldown = Cooldown(floor_s=0.0, cap_s=10.0, halflife_s=5.0)

        async def get_lock_id():
            await cooldown.wait_if_needed()
            return id(cooldown._lock)

        lock_id_1 = asyncio.run(get_lock_id())
        lock_id_2 = asyncio.run(get_lock_id())

        assert lock_id_1 != lock_id_2


class TestOpenAIMinimaLlmMultiLoop:
    """Test OpenAIMinimaLlm lazy initialization across event loops."""

    @pytest.fixture
    def backend_config(self, tmp_path):
        """Create a config with a temporary cache directory."""
        return MinimaLlmConfig(
            base_url="http://localhost:9999/v1",
            api_key="test-key-not-used",
            model="test-model",
            cache_dir=str(tmp_path / "cache"),
            max_outstanding=10,
            rpm=0,
        )

    def test_backend_async_resources_recreated_for_new_loop(self, backend_config):
        """Backend should recreate Semaphore for new loop."""
        backend = OpenAIMinimaLlm(backend_config)

        async def get_resource_ids():
            backend._ensure_async_resources()
            backend._rpm._ensure_lock()
            backend._cooldown._ensure_lock()
            return {
                "sem": id(backend._sem),
                "rpm": id(backend._rpm),
                "cooldown": id(backend._cooldown),
                "rpm_lock": id(backend._rpm._lock),
                "cooldown_lock": id(backend._cooldown._lock),
                "loop": id(backend._bound_loop),
            }

        ids_1 = asyncio.run(get_resource_ids())
        ids_2 = asyncio.run(get_resource_ids())

        assert ids_1["sem"] != ids_2["sem"], "Semaphore should be recreated"
        assert ids_1["rpm"] == ids_2["rpm"], "RpmGate should persist across loops"
        assert ids_1["cooldown"] == ids_2["cooldown"], "Cooldown should persist across loops"
        assert ids_1["rpm_lock"] != ids_2["rpm_lock"], "RpmGate lock should be recreated"
        assert ids_1["cooldown_lock"] != ids_2["cooldown_lock"], "Cooldown lock should be recreated"
        assert ids_1["loop"] != ids_2["loop"], "Loop reference should change"

    def test_backend_cache_reopens_after_close(self, backend_config):
        """Backend cache should reopen automatically after being closed."""
        backend = OpenAIMinimaLlm(backend_config)

        async def use_close_and_verify():
            cache = backend._ensure_cache()
            assert cache is not None
            cache.put("test_key", "test_value", None)

            await backend.aclose()
            assert backend._cache is None, "Cache should be cleared after aclose"

            cache2 = backend._ensure_cache()
            assert cache2 is not None, "Cache should reopen"
            result = cache2.get("test_key")
            assert result is not None, "Data should persist across reopen"
            assert result[0] == "test_value"

            return True

        result = asyncio.run(use_close_and_verify())
        assert result is True

    def test_backend_works_across_asyncio_runs_without_close(self, backend_config):
        """Backend should work across asyncio.run() without explicit close."""
        backend = OpenAIMinimaLlm(backend_config)

        async def ensure_resources():
            backend._ensure_async_resources()
            cache = backend._ensure_cache()
            return cache is not None

        result1 = asyncio.run(ensure_resources())
        assert result1 is True

        result2 = asyncio.run(ensure_resources())
        assert result2 is True

        result3 = asyncio.run(ensure_resources())
        assert result3 is True

    def test_semaphore_usable_after_loop_change(self, backend_config):
        """Semaphore should be usable in new event loop without deadlock."""
        backend = OpenAIMinimaLlm(backend_config)

        async def use_semaphore():
            backend._ensure_async_resources()
            async with backend._sem:
                return True

        result1 = asyncio.run(use_semaphore())
        assert result1 is True

        result2 = asyncio.run(use_semaphore())
        assert result2 is True


class TestPromptCacheReopen:
    """Test PromptCache behavior with close and reopen pattern."""

    def test_cache_operations_after_reopen(self, tmp_path):
        """Cache should work correctly after being closed and reopened."""
        db_path = str(tmp_path / "test_cache.db")

        cache1 = PromptCache(db_path)
        cache1.put("key1", "value1", {"raw": "data1"})
        cache1.close()

        cache2 = PromptCache(db_path)
        result = cache2.get("key1")
        cache2.close()

        assert result is not None
        assert result[0] == "value1"
        assert result[1] == {"raw": "data1"}

    def test_cache_data_persists_across_reopens(self, tmp_path):
        """Data written before close should be readable after reopen."""
        db_path = str(tmp_path / "persist_cache.db")

        cache1 = PromptCache(db_path)
        cache1.put("key_a", "value_a", None)
        cache1.put("key_b", "value_b", {"nested": "data"})
        cache1.close()

        cache2 = PromptCache(db_path)
        result_a = cache2.get("key_a")
        result_b = cache2.get("key_b")
        result_missing = cache2.get("nonexistent")
        cache2.close()

        assert result_a == ("value_a", None)
        assert result_b == ("value_b", {"nested": "data"})
        assert result_missing is None


class TestIntegrationMultipleAsyncioRuns:
    """Integration test simulating reuse across multiple asyncio.run() calls."""

    @pytest.fixture
    def backend(self, tmp_path):
        """Create a backend with temporary cache."""
        cfg = MinimaLlmConfig(
            base_url="http://localhost:9999/v1",
            api_key="test-key",
            model="test-model",
            cache_dir=str(tmp_path / "cache"),
            max_outstanding=5,
            rpm=0,
        )
        return OpenAIMinimaLlm(cfg)

    def test_simulated_nuggify_then_judge_workflow(self, backend):
        """Simulate a workflow that calls asyncio.run() twice."""

        async def simulate_nuggify():
            backend._ensure_async_resources()
            cache = backend._ensure_cache()

            await backend._cooldown.wait_if_needed()
            await backend._rpm.wait_turn()

            async with backend._sem:
                pass

            if cache:
                cache.put("nuggify_key", "nuggify_result", None)

            return "nuggify_done"

        async def simulate_judge():
            backend._ensure_async_resources()
            cache = backend._ensure_cache()

            await backend._cooldown.wait_if_needed()
            await backend._rpm.wait_turn()

            async with backend._sem:
                pass

            if cache:
                cache.put("judge_key", "judge_result", None)

            return "judge_done"

        result1 = asyncio.run(simulate_nuggify())
        assert result1 == "nuggify_done"

        result2 = asyncio.run(simulate_judge())
        assert result2 == "judge_done"

    def test_workflow_with_aclose_between_phases(self, backend):
        """Test workflow where aclose() is called between phases."""

        async def phase1_with_close():
            backend._ensure_async_resources()
            cache = backend._ensure_cache()
            if cache:
                cache.put("phase1", "data1", None)
            await backend.aclose()
            return "phase1_done"

        async def phase2_after_close():
            backend._ensure_async_resources()
            cache = backend._ensure_cache()
            if cache:
                result = cache.get("phase1")
                cache.put("phase2", "data2", None)
                return result
            return None

        result1 = asyncio.run(phase1_with_close())
        assert result1 == "phase1_done"

        result2 = asyncio.run(phase2_after_close())
        assert result2 is not None
        assert result2[0] == "data1"

    def test_three_phase_workflow(self, backend):
        """Test three consecutive asyncio.run() calls."""

        async def phase(n: int):
            backend._ensure_async_resources()
            cache = backend._ensure_cache()
            async with backend._sem:
                if cache:
                    cache.put(f"phase{n}", f"data{n}", None)
            return f"phase{n}_done"

        result1 = asyncio.run(phase(1))
        result2 = asyncio.run(phase(2))
        result3 = asyncio.run(phase(3))

        assert result1 == "phase1_done"
        assert result2 == "phase2_done"
        assert result3 == "phase3_done"

        async def verify():
            cache = backend._ensure_cache()
            return [cache.get(f"phase{n}") for n in [1, 2, 3]]

        results = asyncio.run(verify())
        assert all(r is not None for r in results)
        assert [r[0] for r in results] == ["data1", "data2", "data3"]


class TestBatchRunnerMultiLoop:
    """Test run_batched_callable across multiple asyncio.run() calls."""

    def test_run_batched_callable_across_asyncio_runs(self, tmp_path):
        """run_batched_callable should work across multiple asyncio.run() calls."""
        from minima_llm import run_batched_callable, BatchConfig

        batch_config = BatchConfig(
            num_workers=2,
            heartbeat_s=0,
            stall_s=0,
            max_failures=None,
        )

        async def dummy_callable(item: int) -> int:
            await asyncio.sleep(0.001)
            return item * 2

        async def run_batch(items):
            return await run_batched_callable(items, dummy_callable, batch_config)

        results1 = asyncio.run(run_batch([1, 2, 3]))
        assert results1 == [2, 4, 6]

        results2 = asyncio.run(run_batch([10, 20, 30]))
        assert results2 == [20, 40, 60]

    def test_backend_run_batched_callable_across_runs(self, tmp_path):
        """Backend's run_batched_callable method should work across runs."""
        cfg = MinimaLlmConfig(
            base_url="http://localhost:9999/v1",
            api_key="test",
            model="test",
            cache_dir=str(tmp_path / "cache"),
            max_outstanding=5,
            rpm=0,
        )
        backend = OpenAIMinimaLlm(cfg)

        async def dummy_callable(item: str) -> str:
            backend._ensure_async_resources()
            async with backend._sem:
                await asyncio.sleep(0.001)
            return f"processed_{item}"

        async def run_batch(items):
            return await backend.run_batched_callable(items, dummy_callable)

        results1 = asyncio.run(run_batch(["a", "b"]))
        assert set(results1) == {"processed_a", "processed_b"}

        results2 = asyncio.run(run_batch(["x", "y", "z"]))
        assert set(results2) == {"processed_x", "processed_y", "processed_z"}
