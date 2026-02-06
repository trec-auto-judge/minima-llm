# batch.py
"""
Parasail Batch API support for MinimaLlm.

Provides async batch submission with:
- Automatic cache filtering (skip already-cached requests)
- Resumption support via state files
- Polling with progress updates
- Cache population from batch results
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .config import MinimaLlmConfig
from .protocol import MinimaLlmRequest

if TYPE_CHECKING:
    from .backend import OpenAIMinimaLlm

# Type alias for JSON-serializable data
Json = Dict[str, Any]


# ----------------------------
# Data classes
# ----------------------------

@dataclass
class BatchState:
    """
    Persistent state for a Parasail batch job.

    Stored as JSON in state_dir for resumption after interruption.
    """
    prefix: str  # User-provided batch prefix
    batch_id: Optional[str] = None  # Parasail batch ID (assigned after upload)
    input_file_id: Optional[str] = None  # Parasail file ID for uploaded .jsonl
    output_file_id: Optional[str] = None  # Parasail file ID for results
    status: str = "pending"  # pending, validating, in_progress, finalizing, completed, failed, expired, cancelled
    created_at: float = field(default_factory=time.time)
    custom_id_to_cache_key: Dict[str, str] = field(default_factory=dict)

    def with_status(self, status: str) -> "BatchState":
        """Return new state with updated status."""
        return BatchState(
            prefix=self.prefix,
            batch_id=self.batch_id,
            input_file_id=self.input_file_id,
            output_file_id=self.output_file_id,
            status=status,
            created_at=self.created_at,
            custom_id_to_cache_key=self.custom_id_to_cache_key,
        )

    def with_output_file_id(self, output_file_id: str) -> "BatchState":
        """Return new state with output_file_id set."""
        return BatchState(
            prefix=self.prefix,
            batch_id=self.batch_id,
            input_file_id=self.input_file_id,
            output_file_id=output_file_id,
            status=self.status,
            created_at=self.created_at,
            custom_id_to_cache_key=self.custom_id_to_cache_key,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "prefix": self.prefix,
            "batch_id": self.batch_id,
            "input_file_id": self.input_file_id,
            "output_file_id": self.output_file_id,
            "status": self.status,
            "created_at": self.created_at,
            "custom_id_to_cache_key": self.custom_id_to_cache_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchState":
        """Deserialize from dict."""
        return cls(
            prefix=data["prefix"],
            batch_id=data.get("batch_id"),
            input_file_id=data.get("input_file_id"),
            output_file_id=data.get("output_file_id"),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", time.time()),
            custom_id_to_cache_key=data.get("custom_id_to_cache_key", {}),
        )


@dataclass
class BatchResult:
    """Result summary from a batch submission."""
    cached_count: int  # Requests skipped (already in cache)
    submitted_count: int  # Requests uploaded to Parasail
    completed_count: int  # Successful results from batch
    failed_requests: List[Tuple[str, Any]]  # (custom_id, error) for failures


# ----------------------------
# Batch Collector
# ----------------------------

class BatchCollector:
    """
    Collects requests during batch mode, handles submission and polling.

    Usage:
        collector = BatchCollector(config, "my-batch-prefix")
        future1 = collector.add_request(req1, cache_key1)
        future2 = collector.add_request(req2, cache_key2)
        # ... add more requests ...
        await collector.submit_and_wait()
        # All futures now resolved, results in cache
    """

    def __init__(self, config: MinimaLlmConfig, prefix: str, backend: "Optional[OpenAIMinimaLlm]" = None):
        self.config = config
        self.prefix = prefix
        self._backend = backend  # Use backend's cache if provided
        self._pending: List[Tuple[MinimaLlmRequest, str, asyncio.Future[bool]]] = []
        self._state_dir = self._get_state_dir()
        self._state_file = self._state_dir / f"parasail_batch_{prefix}.json"

    def _get_state_dir(self) -> Path:
        """Get directory for state files."""
        state_dir = self.config.parasail.state_dir or self.config.cache_dir
        if not state_dir:
            raise ValueError("Either parasail.state_dir or cache_dir must be set for batch mode")
        path = Path(state_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_cache_db_path(self) -> Optional[str]:
        """Get path to cache database."""
        if self.config.cache_dir:
            return os.path.join(self.config.cache_dir, "prompt_cache.db")
        return None

    def add_request(self, req: MinimaLlmRequest, cache_key: str) -> "asyncio.Future[bool]":
        """
        Add request to pending batch, return Future that resolves when result available.

        The Future resolves to True on success, or raises an exception on failure.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending.append((req, cache_key, future))
        return future

    def has_pending(self) -> bool:
        """Check if there are pending requests."""
        return len(self._pending) > 0

    @property
    def pending_count(self) -> int:
        """Number of pending requests."""
        return len(self._pending)

    async def populate_cache_if_completed(self) -> bool:
        """Populate cache from completed batch state if it exists.

        Called at context ENTRY (before collection) so that items hit cache
        during collection and aren't added to _pending.

        Returns True if cache was populated from a completed batch.
        """
        state = self._load_state()
        if state is None or state.status != "completed":
            return False

        # Batch is completed - populate cache from it
        print(f"Found completed batch '{state.batch_id}'. Populating cache...")
        result = await self._populate_cache_from_completed_batch(state)
        print(f"Cache populated: {result.completed_count} items. Collection will skip cached items.")

        # Delete state file so submit_and_wait() creates NEW batch for any remaining cache misses
        if self._state_file.exists():
            self._state_file.unlink()
            print(f"Cleared old batch state. New requests will create a fresh batch.")

        return True

    def _load_state(self) -> Optional[BatchState]:
        """Load existing batch state from file."""
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    return BatchState.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load state file {self._state_file}: {e}")
        return None

    def _save_state(self, state: BatchState) -> None:
        """Save batch state to file."""
        with open(self._state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _make_custom_id(self, idx: int) -> str:
        """Generate custom_id for a request."""
        return f"{self.prefix}-{idx}"

    def _make_cache_key(self, req: MinimaLlmRequest) -> str:
        """Generate cache key from request parameters (same as OpenAIMinimaLlm)."""
        obj: Dict[str, Any] = {"model": self.config.model, "messages": req.messages}
        if req.temperature is not None:
            obj["temperature"] = req.temperature
        if req.max_tokens is not None:
            obj["max_tokens"] = req.max_tokens
        if req.extra:
            obj["extra"] = req.extra
        canonical = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    async def submit_and_wait(self) -> BatchResult:
        """
        Submit batch to Parasail, poll until done, populate cache, resolve futures.

        If the number of pending requests exceeds max_batch_size, splits into
        multiple batches with suffixed prefixes.

        Returns:
            BatchResult with summary of the operation.
        """
        if not self._pending:
            return BatchResult(cached_count=0, submitted_count=0, completed_count=0, failed_requests=[])

        max_size = self.config.parasail.max_batch_size

        # Check if we need to split into multiple batches
        if len(self._pending) > max_size:
            return await self._submit_and_wait_chunked(max_size)

        # Single batch path (original logic)
        return await self._submit_and_wait_single()

    async def _submit_and_wait_single(self) -> BatchResult:
        """Submit a single batch (original logic)."""
        # Check for existing batch (resumption case)
        state = self._load_state()

        if state and state.status in ("validating", "in_progress", "finalizing"):
            print(f"Resuming existing batch {state.batch_id} (status: {state.status})")
        elif state and state.status == "completed":
            print(f"Batch {state.batch_id} already completed, fetching results...")
            result = await self._populate_cache_from_completed_batch(state)
            self._clear_pending_futures()
            return result
        else:
            # New batch - upload
            state = await self._upload_batch()

        # Poll until completion
        state = await self._poll_until_done(state)

        # Download results and resolve futures
        return await self._download_and_resolve(state)

    async def _submit_and_wait_chunked(self, max_size: int) -> BatchResult:
        """Submit multiple batches when pending exceeds max_batch_size."""
        num_chunks = (len(self._pending) + max_size - 1) // max_size
        print(f"Splitting {len(self._pending)} requests into {num_chunks} batches (max {max_size} each)")

        # Split pending into chunks
        chunks: List[List[Tuple[MinimaLlmRequest, str, asyncio.Future[bool]]]] = []
        for i in range(num_chunks):
            start = i * max_size
            end = min(start + max_size, len(self._pending))
            chunks.append(self._pending[start:end])

        # Upload all chunks (or resume existing)
        batch_states: List[Tuple[BatchState, int]] = []
        for i, chunk in enumerate(chunks):
            chunk_prefix = f"{self.prefix}-{i}"
            state_file = self._state_dir / f"parasail_batch_{chunk_prefix}.json"

            state = self._load_state_from_file(state_file)
            if state and state.status in ("validating", "in_progress", "finalizing"):
                print(f"Resuming chunk {i}: {state.batch_id} (status: {state.status})")
            elif state and state.status == "completed":
                print(f"Chunk {i} already completed: {state.batch_id}")
                await self._populate_cache_from_completed_batch(state)
                continue
            else:
                state = await self._upload_batch_chunk(chunk, chunk_prefix, state_file)

            batch_states.append((state, i))

        if not batch_states:
            self._clear_pending_futures()
            return BatchResult(cached_count=0, submitted_count=len(self._pending),
                             completed_count=len(self._pending), failed_requests=[])

        # Poll all batches in parallel
        print(f"Polling {len(batch_states)} batch(es) in parallel...")
        await self._poll_all_until_done(batch_states)

        # Download and populate cache for each completed batch
        total_completed = 0
        all_failed: List[Tuple[str, Any]] = []

        for state, chunk_idx in batch_states:
            chunk_prefix = f"{self.prefix}-{chunk_idx}"
            state_file = self._state_dir / f"parasail_batch_{chunk_prefix}.json"

            state = self._load_state_from_file(state_file)
            if state and state.status == "completed":
                result = await self._populate_cache_from_completed_batch(state)
                total_completed += result.completed_count
                all_failed.extend(result.failed_requests)

        self._clear_pending_futures()

        return BatchResult(
            cached_count=0,
            submitted_count=len(self._pending),
            completed_count=total_completed,
            failed_requests=all_failed,
        )

    def _load_state_from_file(self, state_file: Path) -> Optional[BatchState]:
        """Load batch state from a specific file."""
        if state_file.exists():
            try:
                with open(state_file) as f:
                    return BatchState.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load state file {state_file}: {e}")
        return None

    def _save_state_to_file(self, state: BatchState, state_file: Path) -> None:
        """Save batch state to a specific file."""
        with open(state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    async def _upload_batch_chunk(
        self,
        chunk: List[Tuple[MinimaLlmRequest, str, "asyncio.Future[bool]"]],
        chunk_prefix: str,
        state_file: Path,
    ) -> BatchState:
        """Upload a single chunk as a batch."""
        lines = []
        custom_id_to_cache_key = {}

        for idx, (req, cache_key, _future) in enumerate(chunk):
            custom_id = f"{chunk_prefix}-{idx}"
            custom_id_to_cache_key[custom_id] = cache_key

            body: Dict[str, Any] = {
                "model": self.config.model,
                "messages": req.messages,
            }
            if req.temperature is not None:
                body["temperature"] = req.temperature
            if req.max_tokens is not None:
                body["max_tokens"] = req.max_tokens
            if req.extra:
                body.update(req.extra)

            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            lines.append(json.dumps(line))

        jsonl_content = "\n".join(lines)

        print(f"Uploading batch chunk '{chunk_prefix}' ({len(lines)} requests)...")
        input_file_id = await self._upload_file(jsonl_content)
        print(f"  File uploaded: {input_file_id}")

        batch_response = await self._create_batch(input_file_id)
        batch_id = batch_response["id"]
        print(f"  Batch created: {batch_id}")

        state = BatchState(
            prefix=chunk_prefix,
            batch_id=batch_id,
            input_file_id=input_file_id,
            status=batch_response["status"],
            created_at=time.time(),
            custom_id_to_cache_key=custom_id_to_cache_key,
        )
        self._save_state_to_file(state, state_file)

        return state

    async def _poll_all_until_done(self, batch_states: List[Tuple[BatchState, int]]) -> None:
        """Poll multiple batches in parallel until all complete."""
        start_time = time.time()
        max_poll_s = self.config.parasail.max_poll_hours * 3600
        poll_interval = self.config.parasail.poll_interval_s

        active = {i: state for state, i in batch_states}

        while active:
            elapsed = time.time() - start_time
            if elapsed > max_poll_s:
                raise TimeoutError(f"Batches not completed after {elapsed/3600:.1f}h")

            poll_tasks = []
            for chunk_idx, state in active.items():
                poll_tasks.append(self._poll_single_batch(state, chunk_idx))

            results = await asyncio.gather(*poll_tasks, return_exceptions=True)

            completed_chunks = []
            for (chunk_idx, state), result in zip(list(active.items()), results):
                if isinstance(result, Exception):
                    print(f"  Chunk {chunk_idx}: poll error: {result}")
                    continue

                new_state = result
                chunk_prefix = f"{self.prefix}-{chunk_idx}"
                state_file = self._state_dir / f"parasail_batch_{chunk_prefix}.json"
                self._save_state_to_file(new_state, state_file)

                if new_state.status == "completed":
                    print(f"  Chunk {chunk_idx} ({new_state.batch_id}) completed!")
                    completed_chunks.append(chunk_idx)
                elif new_state.status in ("failed", "expired", "cancelled"):
                    print(f"  Chunk {chunk_idx} ({new_state.batch_id}) {new_state.status}")
                    completed_chunks.append(chunk_idx)
                else:
                    active[chunk_idx] = new_state

            for chunk_idx in completed_chunks:
                del active[chunk_idx]

            if active:
                counts_str = ", ".join(
                    f"chunk-{i}: {s.status}" for i, s in sorted(active.items())
                )
                print(f"Polling: {len(active)} active ({elapsed/60:.0f}m elapsed) - {counts_str}")
                await asyncio.sleep(poll_interval)

    async def _poll_single_batch(self, state: BatchState, chunk_idx: int) -> BatchState:
        """Poll a single batch and return updated state."""
        batch_info = await self._get_batch_status(state.batch_id)  # type: ignore[arg-type]
        new_state = state.with_status(batch_info["status"])

        if batch_info["status"] == "completed":
            output_file_id = batch_info.get("output_file_id")
            if output_file_id:
                new_state = new_state.with_output_file_id(output_file_id)

        return new_state

    def _clear_pending_futures(self) -> None:
        """Clear pending list and mark all futures as done."""
        for _req, _cache_key, future in self._pending:
            if not future.done():
                future.set_result(True)
        self._pending.clear()

    async def _populate_cache_from_completed_batch(self, state: BatchState) -> BatchResult:
        """Populate cache from an already-completed batch using saved state."""
        if not state.output_file_id:
            raise RuntimeError(f"Batch {state.batch_id} has no output_file_id")

        print(f"Downloading results from {state.output_file_id}...")
        output_content = await self._download_file(state.output_file_id)

        results: Dict[str, Dict[str, Any]] = {}
        for line in output_content.strip().split("\n"):
            if line:
                result = json.loads(line)
                results[result["custom_id"]] = result

        cache = self._backend._ensure_cache() if self._backend else None

        completed = 0
        failed_requests: List[Tuple[str, Any]] = []

        for custom_id, cache_key in state.custom_id_to_cache_key.items():
            if custom_id not in results:
                error = f"Request {custom_id} not in batch results"
                failed_requests.append((custom_id, error))
                continue

            result = results[custom_id]

            if result.get("error"):
                error = result["error"]
                failed_requests.append((custom_id, error))
                continue

            response = result.get("response", {})
            response_body = response.get("body", {})
            choices = response_body.get("choices", [])

            if not choices:
                error = "No choices in response"
                failed_requests.append((custom_id, error))
                continue

            text = choices[0].get("message", {}).get("content", "")

            if cache:
                cache.put(cache_key, text, response_body)

            completed += 1

        print(f"Batch results: {completed} completed, {len(failed_requests)} failed")

        return BatchResult(
            cached_count=0,
            submitted_count=len(state.custom_id_to_cache_key),
            completed_count=completed,
            failed_requests=failed_requests,
        )

    async def _upload_batch(self) -> BatchState:
        """Create .jsonl file, upload to Parasail, create batch job."""
        lines = []
        custom_id_to_cache_key = {}
        seen_cache_keys: set[str] = set()
        duplicate_count = 0

        for idx, (req, cache_key, _future) in enumerate(self._pending):
            if cache_key in seen_cache_keys:
                duplicate_count += 1
                continue
            seen_cache_keys.add(cache_key)

            custom_id = self._make_custom_id(idx)
            custom_id_to_cache_key[custom_id] = cache_key

            body: Dict[str, Any] = {
                "model": self.config.model,
                "messages": req.messages,
            }
            if req.temperature is not None:
                body["temperature"] = req.temperature
            if req.max_tokens is not None:
                body["max_tokens"] = req.max_tokens
            if req.extra:
                body.update(req.extra)

            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            lines.append(json.dumps(line))

        if duplicate_count > 0:
            print(f"Deduplicated: {duplicate_count} duplicate prompts skipped ({len(lines)} unique)")

        jsonl_content = "\n".join(lines)

        print(f"Uploading batch file ({len(lines)} requests)...")
        input_file_id = await self._upload_file(jsonl_content)
        print(f"  File uploaded: {input_file_id}")

        print(f"Creating batch job...")
        batch_response = await self._create_batch(input_file_id)
        batch_id = batch_response["id"]
        print(f"  Batch created: {batch_id}")

        state = BatchState(
            prefix=self.prefix,
            batch_id=batch_id,
            input_file_id=input_file_id,
            status=batch_response["status"],
            created_at=time.time(),
            custom_id_to_cache_key=custom_id_to_cache_key,
        )
        self._save_state(state)

        return state

    async def _poll_until_done(self, state: BatchState) -> BatchState:
        """Poll batch status until completed/failed/expired."""
        start_time = time.time()
        max_poll_s = self.config.parasail.max_poll_hours * 3600
        poll_interval = self.config.parasail.poll_interval_s

        while True:
            elapsed = time.time() - start_time
            if elapsed > max_poll_s:
                raise TimeoutError(f"Batch {state.batch_id} not completed after {elapsed/3600:.1f}h")

            batch_info = await self._get_batch_status(state.batch_id)  # type: ignore[arg-type]
            state = state.with_status(batch_info["status"])
            self._save_state(state)

            status = batch_info["status"]
            if status == "completed":
                output_file_id = batch_info.get("output_file_id")
                if output_file_id:
                    state = state.with_output_file_id(output_file_id)
                    self._save_state(state)
                print(f"Batch {state.batch_id} completed!")
                return state
            elif status in ("failed", "expired", "cancelled"):
                errors = batch_info.get("errors")
                raise RuntimeError(f"Batch {state.batch_id} {status}: {errors}")

            counts = batch_info.get("request_counts", {})
            total = counts.get("total", "?")
            completed = counts.get("completed", 0)
            failed = counts.get("failed", 0)
            print(f"Batch {state.batch_id}: {status} "
                  f"({completed}/{total} done, {failed} failed, "
                  f"{elapsed/60:.0f}m elapsed)")

            await asyncio.sleep(poll_interval)

    async def _download_and_resolve(self, state: BatchState) -> BatchResult:
        """Download results, populate cache, resolve all pending futures."""
        if not state.output_file_id:
            raise RuntimeError(f"Batch {state.batch_id} has no output_file_id")

        print(f"Downloading results from {state.output_file_id}...")
        output_content = await self._download_file(state.output_file_id)

        results: Dict[str, Dict[str, Any]] = {}
        for line in output_content.strip().split("\n"):
            if line:
                result = json.loads(line)
                results[result["custom_id"]] = result

        cache = self._backend._ensure_cache() if self._backend else None

        completed = 0
        failed_requests: List[Tuple[str, Any]] = []

        for idx, (_req, cache_key, future) in enumerate(self._pending):
            custom_id = self._make_custom_id(idx)

            if custom_id not in results:
                error = f"Request {custom_id} not in batch results"
                failed_requests.append((custom_id, error))
                if not future.done():
                    future.set_result(True)
                continue

            result = results[custom_id]

            if result.get("error"):
                error = result["error"]
                failed_requests.append((custom_id, error))
                if not future.done():
                    future.set_result(True)
                continue

            response = result.get("response", {})
            response_body = response.get("body", {})
            choices = response_body.get("choices", [])

            if not choices:
                error = "No choices in response"
                failed_requests.append((custom_id, error))
                if not future.done():
                    future.set_result(True)
                continue

            text = choices[0].get("message", {}).get("content", "")

            if cache:
                cache.put(cache_key, text, response_body)

            completed += 1
            if not future.done():
                future.set_result(True)

        print(f"Batch results: {completed} completed, {len(failed_requests)} failed")

        return BatchResult(
            cached_count=0,
            submitted_count=len(self._pending),
            completed_count=completed,
            failed_requests=failed_requests,
        )

    # ----------------------------
    # HTTP helpers for Parasail API
    # ----------------------------

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Parasail API calls."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def _upload_file(self, content: str) -> str:
        """Upload .jsonl file to Parasail Files API."""
        url = f"{self.config.base_url}/files"

        boundary = "----BatchUploadBoundary" + hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
        body_parts = []

        body_parts.append(f"--{boundary}".encode())
        body_parts.append(b'Content-Disposition: form-data; name="purpose"')
        body_parts.append(b"")
        body_parts.append(b"batch")

        body_parts.append(f"--{boundary}".encode())
        body_parts.append(b'Content-Disposition: form-data; name="file"; filename="batch_input.jsonl"')
        body_parts.append(b"Content-Type: application/jsonl")
        body_parts.append(b"")
        body_parts.append(content.encode("utf-8"))

        body_parts.append(f"--{boundary}--".encode())

        body = b"\r\n".join(body_parts)

        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._do_request("POST", url, body, headers)
        )

        return response["id"]

    async def _create_batch(self, input_file_id: str) -> Dict[str, Any]:
        """Create batch job via POST /v1/batches."""
        url = f"{self.config.base_url}/batches"
        payload = {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "1h",
        }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._do_request("POST", url, json.dumps(payload).encode(), self._get_headers())
        )

    async def _get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get batch status via GET /v1/batches/{batch_id}."""
        url = f"{self.config.base_url}/batches/{batch_id}"

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._do_request("GET", url, None, self._get_headers())
        )

    async def _cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """Cancel batch via POST /v1/batches/{batch_id}/cancel."""
        url = f"{self.config.base_url}/batches/{batch_id}/cancel"

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._do_request("POST", url, None, self._get_headers())
        )

    async def _list_batches(self, limit: int = 100) -> Dict[str, Any]:
        """List batches via GET /v1/batches."""
        url = f"{self.config.base_url}/batches?limit={limit}"

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._do_request("GET", url, None, self._get_headers())
        )

    async def _download_file(self, file_id: str) -> str:
        """Download file content via GET /v1/files/{file_id}/content."""
        url = f"{self.config.base_url}/files/{file_id}/content"

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._do_request_text("GET", url, self._get_headers())
        )

    def _do_request(self, method: str, url: str, body: Optional[bytes], headers: Dict[str, str]) -> Dict[str, Any]:
        """Perform HTTP request and return JSON response."""
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {url}: {body_text[:500]}")

    def _do_request_text(self, method: str, url: str, headers: Dict[str, str]) -> str:
        """Perform HTTP request and return text response."""
        class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, _req, _fp, _code, _msg, _hdrs, _newurl):
                return None

        opener = urllib.request.build_opener(NoRedirectHandler)
        req = urllib.request.Request(url, headers=headers, method=method)

        try:
            with opener.open(req, timeout=120.0) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                redirect_url = e.headers.get("Location")
                if redirect_url:
                    redirect_req = urllib.request.Request(redirect_url, method="GET")
                    try:
                        with urllib.request.urlopen(redirect_req, timeout=120.0) as resp:
                            return resp.read().decode("utf-8")
                    except urllib.error.HTTPError as e2:
                        body_text = e2.read().decode("utf-8", errors="replace")
                        raise RuntimeError(f"HTTP {e2.code} from redirect {redirect_url}: {body_text[:500]}")
            body_text = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {url}: {body_text[:500]}")


# ----------------------------
# CLI Utility Functions
# ----------------------------

def check_and_populate_cache(prefix: str, config: MinimaLlmConfig) -> None:
    """
    Check batch status and populate cache if completed.

    Used for resumption after process interruption.
    """
    from .backend import OpenAIMinimaLlm

    backend = OpenAIMinimaLlm(config)
    collector = BatchCollector(config, prefix, backend=backend)
    state = collector._load_state()

    if state is None:
        print(f"No batch state found for prefix '{prefix}'")
        return

    print(f"Batch: {state.batch_id}")
    print(f"Status: {state.status}")
    print(f"Created: {time.ctime(state.created_at)}")
    print(f"Requests: {len(state.custom_id_to_cache_key)}")

    if state.status == "completed":
        print("\nBatch already completed. Populating cache...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if not state.output_file_id:
                print("Error: No output_file_id in state")
                return

            output_content = loop.run_until_complete(collector._download_file(state.output_file_id))

            cache = backend._ensure_cache()
            if not cache:
                print("Error: No cache_dir configured")
                return

            completed = 0
            failed = 0

            for line in output_content.strip().split("\n"):
                if not line:
                    continue
                result = json.loads(line)
                custom_id = result["custom_id"]
                cache_key = state.custom_id_to_cache_key.get(custom_id)

                if not cache_key:
                    print(f"  Warning: Unknown custom_id {custom_id}")
                    continue

                if result.get("error"):
                    failed += 1
                    continue

                response = result.get("response", {})
                response_body = response.get("body", {})
                choices = response_body.get("choices", [])

                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    cache.put(cache_key, text, response_body)
                    completed += 1
                else:
                    failed += 1

            print(f"Results added to cache: {completed} success, {failed} failed")

        finally:
            loop.close()
            loop2 = asyncio.new_event_loop()
            asyncio.set_event_loop(loop2)
            try:
                loop2.run_until_complete(backend.aclose())
            finally:
                loop2.close()

    elif state.status in ("validating", "in_progress", "finalizing"):
        print(f"\nBatch still processing. Checking current status...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            batch_info = loop.run_until_complete(collector._get_batch_status(state.batch_id))
            remote_status = batch_info.get("status", "unknown")
            counts = batch_info.get("request_counts", {})
            print(f"Current status: {remote_status}")
            print(f"Progress: {counts.get('completed', 0)}/{counts.get('total', '?')} done, "
                  f"{counts.get('failed', 0)} failed")

            if remote_status == "completed":
                output_file_id = batch_info.get("output_file_id")
                if output_file_id:
                    state = state.with_status("completed").with_output_file_id(output_file_id)
                    collector._save_state(state)
                    print(f"\nBatch completed! Downloading results...")

                    output_content = loop.run_until_complete(collector._download_file(output_file_id))

                    cache = backend._ensure_cache()
                    if not cache:
                        print("Error: No cache_dir configured")
                        return

                    completed = 0
                    failed = 0

                    for line in output_content.strip().split("\n"):
                        if not line:
                            continue
                        result = json.loads(line)
                        custom_id = result["custom_id"]
                        cache_key = state.custom_id_to_cache_key.get(custom_id)

                        if not cache_key:
                            print(f"  Warning: Unknown custom_id {custom_id}")
                            continue

                        if result.get("error"):
                            failed += 1
                            continue

                        response = result.get("response", {})
                        response_body = response.get("body", {})
                        choices = response_body.get("choices", [])

                        if choices:
                            text = choices[0].get("message", {}).get("content", "")
                            cache.put(cache_key, text, response_body)
                            completed += 1
                        else:
                            failed += 1

                    print(f"Results added to cache: {completed} success, {failed} failed")
                else:
                    print("Error: Batch completed but no output_file_id available")
        finally:
            loop.close()
            loop2 = asyncio.new_event_loop()
            asyncio.set_event_loop(loop2)
            try:
                loop2.run_until_complete(backend.aclose())
            finally:
                loop2.close()

    elif state.status in ("failed", "expired", "cancelled"):
        print(f"\nBatch {state.status}. Cannot retrieve results.")

    else:
        print(f"\nUnknown status: {state.status}")


# ----------------------------
# CLI Batch Management Functions
# ----------------------------

def _get_state_dir(config: MinimaLlmConfig) -> Optional[Path]:
    """Get the state directory from config."""
    state_dir = config.parasail.state_dir or config.cache_dir
    if state_dir:
        return Path(state_dir)
    return None


def _list_batch_states(config: MinimaLlmConfig) -> List[Tuple[Path, BatchState]]:
    """List all batch state files and their parsed states."""
    state_dir = _get_state_dir(config)
    if not state_dir or not state_dir.exists():
        return []

    results = []
    for state_file in state_dir.glob("parasail_batch_*.json"):
        try:
            with open(state_file) as f:
                data = json.load(f)
            state = BatchState(
                prefix=data["prefix"],
                batch_id=data.get("batch_id"),
                input_file_id=data.get("input_file_id"),
                output_file_id=data.get("output_file_id"),
                status=data.get("status", "unknown"),
                created_at=data.get("created_at", 0),
                custom_id_to_cache_key=data.get("custom_id_to_cache_key", {}),
            )
            results.append((state_file, state))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {state_file}: {e}")

    return sorted(results, key=lambda x: x[1].created_at, reverse=True)


def batch_status_overview(config: MinimaLlmConfig) -> None:
    """Show status overview of all local batch state files."""
    from .backend import OpenAIMinimaLlm
    import datetime

    states = _list_batch_states(config)

    if not states:
        print("No batch state files found.")
        state_dir = _get_state_dir(config)
        if state_dir:
            print(f"State directory: {state_dir}")
        else:
            print("No state directory configured (set parasail.state_dir or cache_dir)")
        return

    print(f"Found {len(states)} batch state file(s):\n")

    for state_file, state in states:
        created = datetime.datetime.fromtimestamp(state.created_at).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Prefix: {state.prefix}")
        print(f"  Status: {state.status}")
        print(f"  Created: {created}")
        print(f"  Batch ID: {state.batch_id or '<not submitted>'}")
        print(f"  Requests: {len(state.custom_id_to_cache_key)}")
        print(f"  State file: {state_file}")

        # Check remote status for active batches
        if state.batch_id and state.status in ("validating", "in_progress", "finalizing"):
            print("  Checking remote status...")
            backend = OpenAIMinimaLlm(config)
            collector = ParasailBatchCollector(backend, state.prefix)
            loop = asyncio.new_event_loop()
            try:
                batch_info = loop.run_until_complete(collector._get_batch_status(state.batch_id))
                remote_status = batch_info.get("status", "unknown")
                counts = batch_info.get("request_counts", {})
                print(f"  Remote status: {remote_status}")
                print(f"  Progress: {counts.get('completed', 0)}/{counts.get('total', '?')} done")
            except Exception as e:
                print(f"  Could not check remote status: {e}")
            finally:
                loop.close()
                loop2 = asyncio.new_event_loop()
                asyncio.set_event_loop(loop2)
                try:
                    loop2.run_until_complete(backend.aclose())
                finally:
                    loop2.close()

        print()


def cancel_batch(batch_id: str, config: MinimaLlmConfig) -> None:
    """Cancel a specific batch by its remote batch ID."""
    from .backend import OpenAIMinimaLlm

    backend = OpenAIMinimaLlm(config)

    # Find state file with this batch_id
    states = _list_batch_states(config)
    state_file = None
    for sf, state in states:
        if state.batch_id == batch_id:
            state_file = sf
            break

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Create a dummy collector to use _cancel_batch
        collector = ParasailBatchCollector(backend, "cancel")
        result = loop.run_until_complete(collector._cancel_batch(batch_id))
        print(f"Cancelled batch {batch_id}")
        print(f"Result: {result}")

        # Update local state if found
        if state_file and state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
            data["status"] = "cancelled"
            with open(state_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Updated local state: {state_file}")

    except Exception as e:
        print(f"Error cancelling batch: {e}")
    finally:
        loop.close()
        loop2 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop2)
        try:
            loop2.run_until_complete(backend.aclose())
        finally:
            loop2.close()


def cancel_all_batches(config: MinimaLlmConfig, prefix: str) -> None:
    """Cancel all batches matching a prefix and delete local state files."""
    states = _list_batch_states(config)
    matching = [(sf, state) for sf, state in states if state.prefix.startswith(prefix)]

    if not matching:
        print(f"No batches found matching prefix: {prefix}")
        return

    print(f"Found {len(matching)} batch(es) matching prefix '{prefix}':")

    for state_file, state in matching:
        print(f"\n  Prefix: {state.prefix}")

        # Cancel remote batch if active
        if state.batch_id and state.status in ("validating", "in_progress", "finalizing"):
            print(f"  Cancelling remote batch {state.batch_id}...")
            try:
                cancel_batch(state.batch_id, config)
            except Exception as e:
                print(f"  Warning: Could not cancel remote batch: {e}")

        # Delete local state file
        if state_file.exists():
            state_file.unlink()
            print(f"  Deleted state file: {state_file}")


def cancel_all_local_batches(config: MinimaLlmConfig) -> None:
    """Cancel ALL local batches and delete state files."""
    states = _list_batch_states(config)

    if not states:
        print("No batch state files found.")
        return

    print(f"Found {len(states)} batch state file(s). Cancelling all...")

    for state_file, state in states:
        print(f"\n  Prefix: {state.prefix}")

        # Cancel remote batch if active
        if state.batch_id and state.status in ("validating", "in_progress", "finalizing"):
            print(f"  Cancelling remote batch {state.batch_id}...")
            try:
                cancel_batch(state.batch_id, config)
            except Exception as e:
                print(f"  Warning: Could not cancel remote batch: {e}")

        # Delete local state file
        if state_file.exists():
            state_file.unlink()
            print(f"  Deleted state file: {state_file}")

    print("\nAll local batch state files deleted.")
