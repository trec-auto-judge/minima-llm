"""
Tests for cache lookup across model synonyms and read-only lookup caches.

Covers:
- One-time schema migration (model, canonical columns) on an old cache file
- Newest row wins across synonym keys; the winner is copied under the effective name
- Hits from a lookup dir are copied into the primary cache; the lookup dir stays read-only
- A miss everywhere makes a live call and stores model + canonical with the key
- Unrelated model names do not pick up the synonyms
- Config parsing from env and dict
"""

import asyncio
import hashlib
import os
import sqlite3
import time
from dataclasses import replace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from minima_llm import MinimaLlmConfig, MinimaLlmRequest, OpenAIMinimaLlm
from minima_llm.backend import PromptCache


CANON = "GPT-OSS-120b"
ALIAS = "openai/gpt-oss-120b"


@pytest.fixture
def cfg(tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    return MinimaLlmConfig(
        base_url="http://localhost:9999",
        api_key="test-key",
        model=CANON,
        model_synonyms=(ALIAS,),
        max_outstanding=1,
        timeout_s=1.0,
        base_backoff_s=0.01,
        max_backoff_s=0.02,
        max_attempts=3,
        cache_dir=str(primary),
    )


def _req(request_id: str = "r1") -> MinimaLlmRequest:
    return MinimaLlmRequest(
        request_id=request_id,
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
    )


def _key(backend: OpenAIMinimaLlm, req: MinimaLlmRequest, model: str) -> str:
    return backend._hash_key(backend._canonical_request(req, model))


def _seed(db_dir: str, key: str, text: str, created_at: float) -> None:
    cache = PromptCache(os.path.join(db_dir, "minima_llm.db"))
    cache.put(key, text, {"choices": [{"message": {"content": text}}]}, created_at=created_at)
    cache.close()


def _row(db_dir: str, key: str):
    conn = sqlite3.connect(os.path.join(db_dir, "minima_llm.db"))
    row = conn.execute(
        "SELECT response_text, created_at, model, canonical FROM cache WHERE key = ?", (key,)
    ).fetchone()
    conn.close()
    return row


def _generate(backend: OpenAIMinimaLlm, req: MinimaLlmRequest, calls: List[Dict[str, Any]]):
    async def mock_post_json(url, payload):
        calls.append(payload.copy())
        return (200, {}, b'{"choices":[{"message":{"content":"live"}}]}')

    async def run():
        with patch.object(backend, "_post_json", side_effect=mock_post_json):
            try:
                return await backend.generate(req)
            finally:
                await backend.aclose()

    return asyncio.run(run())


class TestSchemaMigration:
    def test_old_cache_file_gets_audit_columns_and_stays_readable(self, tmp_path):
        path = str(tmp_path / "minima_llm.db")
        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE cache (key TEXT PRIMARY KEY, response_text TEXT NOT NULL, "
            "response_raw TEXT, created_at REAL NOT NULL)"
        )
        conn.execute("INSERT INTO cache VALUES (?, ?, ?, ?)", ("k-old", "old text", None, 1.0))
        conn.commit()
        conn.close()

        cache = PromptCache(path)
        cols = {r[1] for r in cache._conn.execute("PRAGMA table_info(cache)")}
        assert {"model", "canonical"} <= cols
        assert cache.get("k-old") == ("old text", None)
        row = cache.get_row("k-old")
        assert row is not None and row.created_at == 1.0
        cache.close()

    def test_read_only_cache_rejects_put_and_requires_file(self, tmp_path):
        path = str(tmp_path / "minima_llm.db")
        with pytest.raises(FileNotFoundError):
            PromptCache(path, read_only=True)
        PromptCache(path).close()
        ro = PromptCache(path, read_only=True)
        with pytest.raises(RuntimeError):
            ro.put("k", "t", None)
        ro.close()


class TestSynonymLookup:
    def test_newest_row_wins_and_is_copied_under_effective_name(self, cfg):
        backend = OpenAIMinimaLlm(cfg)
        req = _req()
        k_canon = _key(backend, req, CANON)
        k_alias = _key(backend, req, ALIAS)
        _seed(cfg.cache_dir, k_canon, "older canonical", created_at=100.0)
        _seed(cfg.cache_dir, k_alias, "newer alias", created_at=200.0)

        calls: List[Dict[str, Any]] = []
        res = _generate(backend, req, calls)

        assert calls == []
        assert res.cached and res.text == "newer alias"
        assert res.cache_source == f"synonym:{ALIAS}"
        text, created_at, model, canonical = _row(cfg.cache_dir, k_canon)
        assert text == "newer alias"
        assert created_at == 200.0
        assert model == CANON
        assert hashlib.sha256(canonical.encode()).hexdigest() == k_canon
        assert backend.get_pulse().cache_hits_synonym == 1

    def test_primary_row_wins_when_newer(self, cfg):
        backend = OpenAIMinimaLlm(cfg)
        req = _req()
        k_canon = _key(backend, req, CANON)
        k_alias = _key(backend, req, ALIAS)
        _seed(cfg.cache_dir, k_canon, "newer canonical", created_at=300.0)
        _seed(cfg.cache_dir, k_alias, "older alias", created_at=200.0)

        res = _generate(backend, req, [])

        assert res.text == "newer canonical"
        assert res.cache_source == "primary"
        assert _row(cfg.cache_dir, k_canon)[1] == 300.0

    def test_unrelated_model_keeps_its_own_key_only(self, cfg):
        backend = OpenAIMinimaLlm(cfg)
        req = replace(_req(), model="some-other-model")
        _seed(cfg.cache_dir, _key(backend, req, ALIAS), "alias row", created_at=200.0)

        calls: List[Dict[str, Any]] = []
        res = _generate(backend, req, calls)

        assert len(calls) == 1 and calls[0]["model"] == "some-other-model"
        assert res.text == "live" and not res.cached


class TestLookupDirs:
    def test_hit_in_lookup_dir_is_copied_into_primary(self, cfg, tmp_path):
        lookup = tmp_path / "lookup"
        lookup.mkdir()
        PromptCache(str(lookup / "minima_llm.db")).close()
        cfg = replace(cfg, cache_lookup_dirs=(str(lookup),))
        backend = OpenAIMinimaLlm(cfg)
        req = _req()
        k_canon = _key(backend, req, CANON)
        k_alias = _key(backend, req, ALIAS)
        _seed(str(lookup), k_alias, "from lookup", created_at=150.0)
        lookup_mtime = os.path.getmtime(lookup / "minima_llm.db")

        calls: List[Dict[str, Any]] = []
        res = _generate(backend, req, calls)

        assert calls == []
        assert res.text == "from lookup"
        assert res.cache_source == f"lookup:{lookup}:synonym:{ALIAS}"
        text, created_at, model, _ = _row(cfg.cache_dir, k_canon)
        assert (text, created_at, model) == ("from lookup", 150.0, CANON)
        assert os.path.getmtime(lookup / "minima_llm.db") == lookup_mtime
        assert _row(str(lookup), k_canon) is None
        assert backend.get_pulse().cache_hits_lookup == 1

    def test_newest_across_primary_and_lookup_dirs(self, cfg, tmp_path):
        lookup = tmp_path / "lookup"
        lookup.mkdir()
        PromptCache(str(lookup / "minima_llm.db")).close()
        cfg = replace(cfg, cache_lookup_dirs=(str(lookup),))
        backend = OpenAIMinimaLlm(cfg)
        req = _req()
        k_canon = _key(backend, req, CANON)
        _seed(cfg.cache_dir, k_canon, "primary", created_at=100.0)
        _seed(str(lookup), k_canon, "lookup newer", created_at=400.0)

        res = _generate(backend, req, [])

        assert res.text == "lookup newer"
        assert res.cache_source == f"lookup:{lookup}"
        assert _row(cfg.cache_dir, k_canon)[:2] == ("lookup newer", 400.0)

    def test_missing_lookup_dir_fails_at_construction(self, cfg, tmp_path):
        cfg = replace(cfg, cache_lookup_dirs=(str(tmp_path / "nope"),))
        with pytest.raises(FileNotFoundError):
            OpenAIMinimaLlm(cfg)

    def test_lookup_dirs_require_primary_cache(self, cfg, tmp_path):
        cfg = replace(cfg, cache_dir=None, cache_lookup_dirs=(str(tmp_path),))
        with pytest.raises(RuntimeError):
            OpenAIMinimaLlm(cfg)


class TestMissStoresCanonical:
    def test_live_call_stores_model_and_canonical(self, cfg):
        backend = OpenAIMinimaLlm(cfg)
        req = _req()
        k_canon = _key(backend, req, CANON)

        calls: List[Dict[str, Any]] = []
        res = _generate(backend, req, calls)

        assert len(calls) == 1 and res.text == "live" and res.cache_source is None
        text, created_at, model, canonical = _row(cfg.cache_dir, k_canon)
        assert text == "live" and model == CANON
        assert canonical == backend._canonical_request(req, CANON)
        assert hashlib.sha256(canonical.encode()).hexdigest() == k_canon
        assert abs(created_at - time.time()) < 60


class TestConfigParsing:
    def test_env_lists(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1")
        monkeypatch.setenv("OPENAI_MODEL", CANON)
        monkeypatch.setenv("CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("CACHE_LOOKUP_DIRS", "/a/b, /c/d")
        monkeypatch.setenv("MODEL_SYNONYMS", f"{ALIAS},gpt-oss-120b")
        cfg = MinimaLlmConfig.from_env()
        assert cfg.cache_lookup_dirs == ("/a/b", "/c/d")
        assert cfg.model_synonyms == (ALIAS, "gpt-oss-120b")

    def test_dict_accepts_list_or_string(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1")
        monkeypatch.setenv("OPENAI_MODEL", CANON)
        monkeypatch.delenv("CACHE_LOOKUP_DIRS", raising=False)
        monkeypatch.delenv("MODEL_SYNONYMS", raising=False)
        cfg = MinimaLlmConfig.from_dict({"model_synonyms": [ALIAS], "cache_lookup_dirs": "/x"})
        assert cfg.model_synonyms == (ALIAS,)
        assert cfg.cache_lookup_dirs == ("/x",)
        assert "model_synonyms" in cfg.describe()
