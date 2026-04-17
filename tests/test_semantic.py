"""Tests for cachly_agents.semantic – CachlySemanticMemory."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from cachly_agents.semantic import CachlySemanticMemory, SemanticRecallResult


# ── Helpers ───────────────────────────────────────────────────────────────────

FAKE_VECTOR_URL = "https://api.cachly.dev/v1/sem/fake-token"


def fake_embed(text: str) -> list[float]:
    """Deterministic fake embedding for testing."""
    return [float(ord(c)) / 256 for c in text[:8].ljust(8)]


class FakeResponse:
    """Minimal mock for requests.Response."""

    def __init__(self, json_data: dict | list | None = None, status_code: int = 200, text: str = ""):
        self._json = json_data
        self.status_code = status_code
        self._text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCachlySemanticMemory:
    """Unit tests for the semantic agent memory."""

    def test_remember_calls_entries_endpoint(self):
        session = MagicMock()
        session.post.return_value = FakeResponse({"id": "entry-123", "ok": True}, 201)

        mem = CachlySemanticMemory(
            vector_url=FAKE_VECTOR_URL,
            namespace="agent:test",
            session=session,
        )
        entry_id = mem.remember("The budget is €50k", embed_fn=fake_embed)

        session.post.assert_called_once()
        call_url = session.post.call_args[0][0]
        assert call_url.endswith("/entries")
        body = session.post.call_args[1]["json"]
        assert body["prompt"] == "The budget is €50k"
        assert body["namespace"] == "agent:test"
        assert isinstance(body["embedding"], list)
        assert len(body["embedding"]) == 8

    def test_remember_with_metadata(self):
        session = MagicMock()
        session.post.return_value = FakeResponse({"id": "entry-456", "ok": True}, 201)

        mem = CachlySemanticMemory(vector_url=FAKE_VECTOR_URL, session=session)
        mem.remember(
            "Budget is 50k",
            embed_fn=fake_embed,
            metadata={"source": "meeting", "confidence": 0.9},
        )

        body = session.post.call_args[1]["json"]
        assert body["prompt"].startswith("[META:")
        assert '"source": "meeting"' in body["prompt"]
        assert "Budget is 50k" in body["prompt"]

    def test_remember_with_ttl(self):
        session = MagicMock()
        session.post.return_value = FakeResponse({"id": "entry-789", "ok": True}, 201)

        mem = CachlySemanticMemory(vector_url=FAKE_VECTOR_URL, session=session)
        mem.remember("temp fact", embed_fn=fake_embed, ttl_seconds=3600)

        body = session.post.call_args[1]["json"]
        assert "expires_at" in body

    def test_recall_hit(self):
        session = MagicMock()

        # Search returns a hit
        search_resp = FakeResponse({
            "found": True,
            "id": "entry-123",
            "similarity": 0.94,
            "threshold_used": 0.85,
        })
        # List entries returns the entry data
        list_resp = FakeResponse({
            "data": [
                {"id": "entry-123", "prompt": "The budget is €50k"},
                {"id": "entry-other", "prompt": "Something else"},
            ],
            "count": 2,
        })
        session.post.return_value = search_resp
        session.get.return_value = list_resp

        mem = CachlySemanticMemory(vector_url=FAKE_VECTOR_URL, session=session)
        result = mem.recall("What is the budget?", embed_fn=fake_embed)

        assert result is not None
        assert result.text == "The budget is €50k"
        assert result.similarity == 0.94
        assert result.entry_id == "entry-123"

    def test_recall_miss(self):
        session = MagicMock()
        session.post.return_value = FakeResponse({
            "found": False,
            "threshold_used": 0.85,
        })

        mem = CachlySemanticMemory(vector_url=FAKE_VECTOR_URL, session=session)
        result = mem.recall("Unknown topic", embed_fn=fake_embed)

        assert result is None

    def test_recall_with_metadata_parsing(self):
        session = MagicMock()
        session.post.return_value = FakeResponse({
            "found": True,
            "id": "entry-meta",
            "similarity": 0.91,
            "threshold_used": 0.85,
        })
        session.get.return_value = FakeResponse({
            "data": [{"id": "entry-meta", "prompt": '[META:{"source":"meeting"}]\nBudget is 50k'}],
        })

        mem = CachlySemanticMemory(vector_url=FAKE_VECTOR_URL, session=session)
        result = mem.recall("budget?", embed_fn=fake_embed)

        assert result is not None
        assert result.text == "Budget is 50k"
        assert result.metadata == {"source": "meeting"}

    def test_recall_custom_threshold(self):
        session = MagicMock()
        session.post.return_value = FakeResponse({"found": False, "threshold_used": 0.95})

        mem = CachlySemanticMemory(vector_url=FAKE_VECTOR_URL, session=session)
        mem.recall("high bar query", embed_fn=fake_embed, threshold=0.95)

        body = session.post.call_args[1]["json"]
        assert body["threshold"] == 0.95

    def test_forget(self):
        session = MagicMock()
        session.delete.return_value = FakeResponse({"ok": True}, 200)

        mem = CachlySemanticMemory(vector_url=FAKE_VECTOR_URL, session=session)
        ok = mem.forget("entry-123")

        assert ok is True
        session.delete.assert_called_once()
        assert "entries/entry-123" in session.delete.call_args[0][0]

    def test_forget_all(self):
        session = MagicMock()
        session.delete.return_value = FakeResponse({"deleted": 42})

        mem = CachlySemanticMemory(vector_url=FAKE_VECTOR_URL, session=session)
        count = mem.forget_all()

        assert count == 42

    def test_count(self):
        session = MagicMock()
        session.get.return_value = FakeResponse({"size": 15})

        mem = CachlySemanticMemory(vector_url=FAKE_VECTOR_URL, session=session)
        assert mem.count() == 15

    def test_context_manager(self):
        session = MagicMock()
        mem = CachlySemanticMemory(vector_url=FAKE_VECTOR_URL, session=session)
        with mem:
            pass
        session.close.assert_called_once()

    def test_format_and_parse_prompt_roundtrip(self):
        metadata = {"source": "chat", "turn": 3}
        text = "The user prefers dark mode"
        prompt = CachlySemanticMemory._format_prompt(text, metadata)
        parsed_text, parsed_meta = CachlySemanticMemory._parse_prompt(prompt)
        assert parsed_text == text
        assert parsed_meta == metadata

    def test_parse_prompt_no_metadata(self):
        text, meta = CachlySemanticMemory._parse_prompt("plain text")
        assert text == "plain text"
        assert meta is None

    def test_semantic_recall_result_bool(self):
        assert bool(SemanticRecallResult(text="hello", entry_id="x", similarity=0.9, threshold_used=0.85))
        assert not bool(SemanticRecallResult(text="", entry_id="x", similarity=0.9, threshold_used=0.85))

    def test_semantic_recall_result_repr(self):
        r = SemanticRecallResult(text="test", entry_id="abc", similarity=0.95, threshold_used=0.85)
        assert "0.950" in repr(r)
