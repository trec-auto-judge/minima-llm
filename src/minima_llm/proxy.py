# proxy.py
"""OpenAI-compatible HTTP proxy that exposes MinimaLlm's caching and backpressure."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from .backend import OpenAIMinimaLlm
from .config import MinimaLlmConfig
from .protocol import MinimaLlmFailure, MinimaLlmRequest, MinimaLlmResponse


class ProxyServer:
    """Localhost HTTP proxy exposing MinimaLlm as an OpenAI-compatible endpoint.

    Routes:
        POST /v1/chat/completions  - Chat completions (non-streaming only)
        GET  /v1/models            - List available models

    All requests go through the backend's caching, semaphore, RPM gate, and
    retry logic. Multiple concurrent HTTP clients are naturally queued by the
    backend's max_outstanding semaphore.
    """

    def __init__(
        self,
        cfg: MinimaLlmConfig,
        *,
        host: str = "127.0.0.1",
        port: int = 8990,
        force_model: bool = False,
    ):
        self.cfg = cfg
        self.host = host
        self.port = port
        self.force_model = force_model
        self.backend = OpenAIMinimaLlm(cfg)

    async def run(self) -> None:
        """Start the proxy server and serve forever."""
        server = await asyncio.start_server(
            self._handle_connection, self.host, self.port,
        )
        addr = server.sockets[0].getsockname()
        print(f"MinimaLlm proxy listening on http://{addr[0]}:{addr[1]}")
        print(f"  Model: {self.cfg.model}")
        print(f"  Force model: {self.force_model}")
        print(f"  Cache: {self.cfg.cache_dir or '<disabled>'}")
        try:
            async with server:
                await server.serve_forever()
        finally:
            await self.backend.aclose()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single HTTP connection."""
        try:
            method, path, headers, body = await self._read_http_request(reader)
        except Exception:
            writer.close()
            await writer.wait_closed()
            return

        try:
            if method == "POST" and path.rstrip("/") == "/v1/chat/completions":
                status, resp_body = await self._handle_chat_completions(body)
            elif method == "GET" and path.rstrip("/") == "/v1/models":
                status, resp_body = self._handle_models()
            else:
                status = 404
                resp_body = json.dumps({"error": {"message": f"Not found: {method} {path}", "type": "not_found"}})
        except Exception as e:
            status = 500
            resp_body = json.dumps({"error": {"message": str(e), "type": "internal_error"}})

        self._write_http_response(writer, status, resp_body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def _handle_chat_completions(self, body: bytes) -> Tuple[int, str]:
        """Handle POST /v1/chat/completions."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            return 400, json.dumps({"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request_error"}})

        if data.get("stream", False):
            return 400, json.dumps({"error": {"message": "Streaming is not supported. Set stream: false.", "type": "invalid_request_error"}})

        messages = data.get("messages")
        if not messages:
            return 400, json.dumps({"error": {"message": "Missing 'messages' field.", "type": "invalid_request_error"}})

        # Determine model
        client_model = data.get("model")
        if self.force_model:
            model = None  # will fall back to cfg.model in backend
        else:
            model = client_model  # pass through (None falls back to cfg.model)

        # Extract optional parameters
        temperature = data.get("temperature")
        max_tokens = data.get("max_tokens")

        # Collect extra parameters (beyond the ones we handle explicitly)
        known_keys = {"model", "messages", "temperature", "max_tokens", "stream"}
        extra = {k: v for k, v in data.items() if k not in known_keys}

        request_id = str(uuid.uuid4())
        req = MinimaLlmRequest(
            request_id=request_id,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra or None,
        )

        result = await self.backend.generate(req)

        if isinstance(result, MinimaLlmResponse):
            if result.cached:
                print(f"  [{request_id[:8]}] cache hit")
            else:
                print(f"  [{request_id[:8]}] llm call ({result.input_tokens}+{result.output_tokens} tok)")

            if result.raw is not None:
                # Return the full OpenAI-compatible response stored from upstream
                return 200, json.dumps(result.raw)
            else:
                # Cache hit from before raw was stored; construct minimal response
                return 200, json.dumps({
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model or self.cfg.model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": result.input_tokens,
                        "completion_tokens": result.output_tokens,
                        "total_tokens": result.input_tokens + result.output_tokens,
                    },
                })

        elif isinstance(result, MinimaLlmFailure):
            print(f"  [{request_id[:8]}] failed: {result.error_type}: {result.message}")
            http_status = result.status or 502
            return http_status, json.dumps({
                "error": {
                    "message": result.message,
                    "type": result.error_type,
                    "code": result.status,
                },
            })

        # Should not happen
        return 500, json.dumps({"error": {"message": "Unexpected result type", "type": "internal_error"}})

    def _handle_models(self) -> Tuple[int, str]:
        """Handle GET /v1/models."""
        return 200, json.dumps({
            "object": "list",
            "data": [{
                "id": self.cfg.model,
                "object": "model",
                "created": 0,
                "owned_by": "minima-llm-proxy",
            }],
        })

    @staticmethod
    async def _read_http_request(
        reader: asyncio.StreamReader,
    ) -> Tuple[str, str, Dict[str, str], bytes]:
        """Parse an HTTP/1.1 request. Returns (method, path, headers, body)."""
        # Read request line
        request_line = await asyncio.wait_for(reader.readline(), timeout=30.0)
        if not request_line:
            raise ConnectionError("Empty request")
        parts = request_line.decode("utf-8", errors="replace").strip().split()
        if len(parts) < 2:
            raise ValueError(f"Malformed request line: {request_line!r}")
        method = parts[0].upper()
        path = parts[1]

        # Read headers
        headers: Dict[str, str] = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=30.0)
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                break
            if ":" in line_str:
                key, value = line_str.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        # Read body
        body = b""
        content_length = int(headers.get("content-length", "0"))
        if content_length > 0:
            body = await asyncio.wait_for(reader.readexactly(content_length), timeout=30.0)

        return method, path, headers, body

    @staticmethod
    def _write_http_response(
        writer: asyncio.StreamWriter, status: int, body: str,
    ) -> None:
        """Write an HTTP/1.1 response."""
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error", 502: "Bad Gateway"}.get(status, "Error")
        body_bytes = body.encode("utf-8")
        header = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        writer.write(header.encode("utf-8"))
        writer.write(body_bytes)
