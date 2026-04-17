"""
cachly_agents.autogen – AutoGen ConversationHistory store backed by cachly.dev.

Provides:
  CachlyMessageStore  – stores AutoGen conversation messages in Valkey/Redis

Install extras::

    pip install cachly-agents[autogen]
"""
from __future__ import annotations

import json
from typing import Any

import redis


class CachlyMessageStore:
    """
    Persistent message store for AutoGen agents, backed by cachly.dev.

    AutoGen agents can be configured with a custom message store to persist
    conversations across restarts, scale-out, and multi-agent orchestration.

    Usage::

        from cachly_agents.autogen import CachlyMessageStore
        from autogen import ConversableAgent

        store = CachlyMessageStore(
            redis_url=os.environ["CACHLY_URL"],
            conversation_id="project-42",
            ttl=86400,  # messages expire after 24h
        )

        assistant = ConversableAgent(
            name="assistant",
            system_message="You are a helpful AI assistant.",
            # Inject history on start
            initial_history=store.load(),
        )

        # After each turn, persist
        store.append({"role": "user", "content": "Hello!"})
        store.append({"role": "assistant", "content": "Hi there!"})
    """

    def __init__(
        self,
        redis_url: str,
        conversation_id: str,
        key_prefix: str = "cachly:autogen:",
        ttl: int | None = None,
    ) -> None:
        self.conversation_id = conversation_id
        self._key = f"{key_prefix}{conversation_id}"
        self._ttl = ttl
        self._r = redis.from_url(redis_url, decode_responses=True)

    def load(self) -> list[dict[str, Any]]:
        """Load all messages for this conversation."""
        raw = self._r.lrange(self._key, 0, -1)
        return [json.loads(m) for m in raw]

    def append(self, message: dict[str, Any]) -> None:
        """Append a single message to the conversation."""
        self._r.rpush(self._key, json.dumps(message, ensure_ascii=False))
        if self._ttl:
            self._r.expire(self._key, self._ttl)

    def append_many(self, messages: list[dict[str, Any]]) -> None:
        """Append multiple messages in one pipeline call."""
        if not messages:
            return
        pipe = self._r.pipeline()
        for msg in messages:
            pipe.rpush(self._key, json.dumps(msg, ensure_ascii=False))
        if self._ttl:
            pipe.expire(self._key, self._ttl)
        pipe.execute()

    def clear(self) -> None:
        """Delete all messages for this conversation."""
        self._r.delete(self._key)

    def __len__(self) -> int:
        return self._r.llen(self._key)

    def close(self) -> None:
        self._r.close()

