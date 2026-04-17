"""
cachly_agents.memory – Generic Redis-backed long-term memory for AI agents.

Works independently of any specific agent framework.
"""
from __future__ import annotations

import json
import time
from typing import Any

import redis


class CachlyMemory:
    """
    Key-value long-term memory backed by a cachly.dev Valkey instance.

    Each agent gets its own namespace so memories never bleed across sessions.

    Usage::

        from cachly_agents.memory import CachlyMemory

        mem = CachlyMemory(redis_url=os.environ["CACHLY_URL"], namespace="agent:planner")
        mem.remember("user_name", "Alice")
        mem.remember("last_topic", "invoice dispute", ttl=3600)

        print(mem.recall("user_name"))   # → "Alice"
        print(mem.recall_all())          # → {"user_name": "Alice", "last_topic": "invoice dispute"}
        mem.forget("last_topic")
        mem.clear()
    """

    def __init__(
        self,
        redis_url: str,
        namespace: str = "cachly:agent:mem",
        default_ttl: int | None = None,
    ) -> None:
        self._r = redis.from_url(redis_url, decode_responses=True)
        self._ns = namespace
        self._ttl = default_ttl

    # ── internal ────────────────────────────────────────────────────────────

    def _key(self, name: str) -> str:
        return f"{self._ns}:{name}"

    # ── public API ──────────────────────────────────────────────────────────

    def remember(self, name: str, value: Any, ttl: int | None = None) -> None:
        """Store a memory value (JSON-serialised)."""
        serialised = json.dumps(value, ensure_ascii=False)
        effective_ttl = ttl if ttl is not None else self._ttl
        if effective_ttl:
            self._r.setex(self._key(name), effective_ttl, serialised)
        else:
            self._r.set(self._key(name), serialised)

    def recall(self, name: str, default: Any = None) -> Any:
        """Retrieve a memory value. Returns *default* if not found."""
        raw = self._r.get(self._key(name))
        if raw is None:
            return default
        return json.loads(raw)

    def forget(self, name: str) -> None:
        """Delete a single memory."""
        self._r.delete(self._key(name))

    def recall_all(self) -> dict[str, Any]:
        """Return all memories in this namespace as a plain dict."""
        pattern = f"{self._ns}:*"
        keys = list(self._r.scan_iter(pattern))
        if not keys:
            return {}
        values = self._r.mget(keys)
        prefix_len = len(self._ns) + 1
        return {
            k[prefix_len:]: json.loads(v)
            for k, v in zip(keys, values)
            if v is not None
        }

    def clear(self) -> int:
        """Delete all memories in this namespace. Returns number of deleted keys."""
        keys = list(self._r.scan_iter(f"{self._ns}:*"))
        if keys:
            return self._r.delete(*keys)
        return 0

    def snapshot(self) -> dict[str, Any]:
        """Alias for recall_all() – useful for checkpointing agent state."""
        return self.recall_all()

    def restore(self, snapshot: dict[str, Any], ttl: int | None = None) -> None:
        """Restore a previously taken snapshot into memory."""
        for name, value in snapshot.items():
            self.remember(name, value, ttl=ttl)

    def close(self) -> None:
        self._r.close()

    def __enter__(self) -> "CachlyMemory":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

