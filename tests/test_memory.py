"""Tests for cachly_agents.memory – CachlyMemory (framework-agnostic)."""
from __future__ import annotations

from cachly_agents.memory import CachlyMemory


class TestCachlyMemory:
    """Unit tests for the generic agent memory store."""

    def test_remember_and_recall(self, redis_url):
        mem = CachlyMemory(redis_url=redis_url, namespace="test:agent")
        mem.remember("name", "Alice")
        assert mem.recall("name") == "Alice"

    def test_recall_missing_returns_default(self, redis_url):
        mem = CachlyMemory(redis_url=redis_url, namespace="test:miss")
        assert mem.recall("nope") is None
        assert mem.recall("nope", default="fallback") == "fallback"

    def test_remember_complex_types(self, redis_url):
        mem = CachlyMemory(redis_url=redis_url, namespace="test:complex")
        mem.remember("list", [1, 2, 3])
        mem.remember("dict", {"nested": True})
        mem.remember("number", 42)
        assert mem.recall("list") == [1, 2, 3]
        assert mem.recall("dict") == {"nested": True}
        assert mem.recall("number") == 42

    def test_forget(self, redis_url):
        mem = CachlyMemory(redis_url=redis_url, namespace="test:forget")
        mem.remember("temp", "value")
        assert mem.recall("temp") == "value"
        mem.forget("temp")
        assert mem.recall("temp") is None

    def test_recall_all(self, redis_url):
        mem = CachlyMemory(redis_url=redis_url, namespace="test:all")
        mem.remember("a", 1)
        mem.remember("b", 2)
        all_mem = mem.recall_all()
        assert all_mem == {"a": 1, "b": 2}

    def test_recall_all_empty(self, redis_url):
        mem = CachlyMemory(redis_url=redis_url, namespace="test:empty-all")
        assert mem.recall_all() == {}

    def test_clear(self, redis_url):
        mem = CachlyMemory(redis_url=redis_url, namespace="test:clear")
        mem.remember("x", 10)
        mem.remember("y", 20)
        mem.clear()
        assert mem.recall_all() == {}

    def test_namespaces_are_isolated(self, redis_url):
        m1 = CachlyMemory(redis_url=redis_url, namespace="ns:a")
        m2 = CachlyMemory(redis_url=redis_url, namespace="ns:b")
        m1.remember("key", "from-a")
        assert m1.recall("key") == "from-a"
        assert m2.recall("key") is None

    def test_ttl_accepted(self, redis_url):
        mem = CachlyMemory(redis_url=redis_url, namespace="test:ttl", default_ttl=3600)
        mem.remember("k", "v")
        assert mem.recall("k") == "v"

    def test_per_key_ttl_override(self, redis_url):
        mem = CachlyMemory(redis_url=redis_url, namespace="test:ttl-ovr")
        mem.remember("short", "value", ttl=60)
        assert mem.recall("short") == "value"
