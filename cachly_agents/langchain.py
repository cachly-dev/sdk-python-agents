"""
cachly_agents.langchain – LangChain adapters backed by cachly.dev.

Provides:
  CachlyChatMessageHistory  – persistent chat history (BaseChatMessageHistory)
  CachlyVectorStore         – pgvector-backed semantic store (VectorStore) [planned]

Install extras::

    pip install cachly-agents[langchain]
"""
from __future__ import annotations

import json
from typing import List, Sequence

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
except ImportError as e:
    raise ImportError(
        "langchain-core is required for cachly_agents.langchain. "
        "Install with: pip install cachly-agents[langchain]"
    ) from e

import redis


class CachlyChatMessageHistory(BaseChatMessageHistory):
    """
    Persistent LangChain chat history stored in cachly.dev (Valkey/Redis).

    Each *session_id* is stored as a Redis list; messages are JSON-encoded
    LangChain message dicts, preserving full type information (HumanMessage,
    AIMessage, ToolMessage, etc.).

    Usage::

        from cachly_agents.langchain import CachlyChatMessageHistory
        from langchain_openai import ChatOpenAI
        from langchain_core.runnables.history import RunnableWithMessageHistory

        def get_history(session_id: str):
            return CachlyChatMessageHistory(
                session_id=session_id,
                redis_url=os.environ["CACHLY_URL"],
                ttl=3600,      # session expires after 1 h of inactivity
            )

        chain = RunnableWithMessageHistory(
            ChatOpenAI(model="gpt-4o"),
            get_history,
        )

        response = chain.invoke(
            {"role": "user", "content": "What did we discuss last time?"},
            config={"configurable": {"session_id": "user-42"}},
        )
    """

    def __init__(
        self,
        session_id: str,
        redis_url: str,
        key_prefix: str = "cachly:chat:",
        ttl: int | None = None,
    ) -> None:
        self.session_id = session_id
        self._key = f"{key_prefix}{session_id}"
        self._ttl = ttl
        self._r = redis.from_url(redis_url, decode_responses=True)

    # ── BaseChatMessageHistory interface ─────────────────────────────────────

    @property
    def messages(self) -> List[BaseMessage]:
        """Load all messages for this session from Redis."""
        raw_messages = self._r.lrange(self._key, 0, -1)
        if not raw_messages:
            return []
        dicts = [json.loads(m) for m in raw_messages]
        return messages_from_dict(dicts)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Append messages to the history list."""
        if not messages:
            return
        pipe = self._r.pipeline()
        for msg in messages:
            serialised = json.dumps(messages_to_dict([msg])[0], ensure_ascii=False)
            pipe.rpush(self._key, serialised)
        if self._ttl:
            pipe.expire(self._key, self._ttl)
        pipe.execute()

    def clear(self) -> None:
        """Delete all messages for this session."""
        self._r.delete(self._key)

    # ── helpers ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._r.close()

    def __len__(self) -> int:
        return self._r.llen(self._key)

    def __repr__(self) -> str:
        return f"CachlyChatMessageHistory(session_id={self.session_id!r}, key={self._key!r})"

