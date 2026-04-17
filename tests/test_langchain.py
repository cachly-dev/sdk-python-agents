"""Tests for cachly_agents.langchain – CachlyChatMessageHistory."""
from __future__ import annotations

import pytest

try:
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    pytest.skip("langchain-core required for these tests", allow_module_level=True)

from cachly_agents.langchain import CachlyChatMessageHistory


class TestCachlyChatMessageHistory:
    """Unit tests for the LangChain history adapter."""

    def test_empty_session_returns_no_messages(self, redis_url):
        history = CachlyChatMessageHistory(session_id="empty", redis_url=redis_url)
        assert history.messages == []

    def test_add_and_retrieve_messages(self, redis_url):
        history = CachlyChatMessageHistory(session_id="sess-1", redis_url=redis_url)
        history.add_message(HumanMessage(content="Hello"))
        history.add_message(AIMessage(content="Hi there!"))

        msgs = history.messages
        assert len(msgs) == 2
        assert msgs[0].content == "Hello"
        assert msgs[1].content == "Hi there!"

    def test_clear_removes_all_messages(self, redis_url):
        history = CachlyChatMessageHistory(session_id="sess-clear", redis_url=redis_url)
        history.add_message(HumanMessage(content="test"))
        assert len(history.messages) == 1
        history.clear()
        assert history.messages == []

    def test_separate_sessions_are_isolated(self, redis_url):
        h1 = CachlyChatMessageHistory(session_id="a", redis_url=redis_url)
        h2 = CachlyChatMessageHistory(session_id="b", redis_url=redis_url)

        h1.add_message(HumanMessage(content="only in a"))
        assert len(h1.messages) == 1
        assert h2.messages == []

    def test_custom_key_prefix(self, redis_url):
        history = CachlyChatMessageHistory(
            session_id="pfx",
            redis_url=redis_url,
            key_prefix="custom:prefix:",
        )
        history.add_message(HumanMessage(content="hello"))
        assert len(history.messages) == 1

    def test_ttl_parameter_accepted(self, redis_url):
        """TTL should be accepted without error (actual expiry needs a real Redis)."""
        history = CachlyChatMessageHistory(session_id="ttl", redis_url=redis_url, ttl=3600)
        history.add_message(HumanMessage(content="timed"))
        assert len(history.messages) == 1
