"""Tests for cachly_agents.autogen – CachlyMessageStore."""
from __future__ import annotations

from cachly_agents.autogen import CachlyMessageStore


class TestCachlyMessageStore:
    """Unit tests for the AutoGen message store."""

    def test_empty_conversation_returns_empty_list(self, redis_url):
        store = CachlyMessageStore(redis_url=redis_url, conversation_id="empty")
        assert store.load() == []

    def test_append_and_load(self, redis_url):
        store = CachlyMessageStore(redis_url=redis_url, conversation_id="conv-1")
        store.append({"role": "user", "content": "Hello"})
        store.append({"role": "assistant", "content": "Hi!"})

        messages = store.load()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["content"] == "Hi!"

    def test_append_many(self, redis_url):
        store = CachlyMessageStore(redis_url=redis_url, conversation_id="conv-batch")
        store.append_many([
            {"role": "user", "content": "msg-1"},
            {"role": "assistant", "content": "msg-2"},
            {"role": "user", "content": "msg-3"},
        ])
        assert len(store.load()) == 3

    def test_append_many_empty_list_is_noop(self, redis_url):
        store = CachlyMessageStore(redis_url=redis_url, conversation_id="conv-empty")
        store.append_many([])
        assert store.load() == []

    def test_clear(self, redis_url):
        store = CachlyMessageStore(redis_url=redis_url, conversation_id="conv-clear")
        store.append({"role": "user", "content": "hello"})
        assert len(store.load()) == 1
        store.clear()
        assert store.load() == []

    def test_separate_conversations_are_isolated(self, redis_url):
        s1 = CachlyMessageStore(redis_url=redis_url, conversation_id="a")
        s2 = CachlyMessageStore(redis_url=redis_url, conversation_id="b")
        s1.append({"role": "user", "content": "only in a"})
        assert len(s1.load()) == 1
        assert s2.load() == []

    def test_ttl_accepted(self, redis_url):
        store = CachlyMessageStore(redis_url=redis_url, conversation_id="ttl", ttl=7200)
        store.append({"role": "user", "content": "timed"})
        assert len(store.load()) == 1

    def test_custom_key_prefix(self, redis_url):
        store = CachlyMessageStore(
            redis_url=redis_url,
            conversation_id="pfx",
            key_prefix="custom:ag:",
        )
        store.append({"role": "user", "content": "test"})
        assert len(store.load()) == 1
