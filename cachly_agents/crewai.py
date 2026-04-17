"""
cachly_agents.crewai – CrewAI Memory adapter backed by cachly.dev.

Provides:
  CachlyCrewMemory  – Redis-backed memory store for CrewAI agents

Install extras::

    pip install cachly-agents[crewai]
"""
from __future__ import annotations

import json
import time
from typing import Any

import redis


class CachlyCrewMemory:
    """
    Redis-backed memory store compatible with CrewAI's memory interface.

    CrewAI agents use this to persist short-term and long-term memories across
    crew runs, deployments, and restarts.

    Usage::

        from cachly_agents.crewai import CachlyCrewMemory
        from crewai import Agent, Crew, Task

        memory = CachlyCrewMemory(
            redis_url=os.environ["CACHLY_URL"],
            crew_id="research-team",
            ttl=604800,  # 7 days
        )

        researcher = Agent(
            role="Senior Research Analyst",
            goal="Research and summarise technical topics",
            memory=True,
            # cachly provides the storage backend
        )

        # Manually save / load facts
        memory.save("last_research_topic", "LLM agent memory systems")
        topic = memory.load("last_research_topic")

        # Append to a running log
        memory.log("research_log", {"topic": topic, "ts": time.time()})
        full_log = memory.get_log("research_log")
    """

    def __init__(
        self,
        redis_url: str,
        crew_id: str,
        key_prefix: str = "cachly:crew:",
        ttl: int | None = None,
    ) -> None:
        self.crew_id = crew_id
        self._prefix = f"{key_prefix}{crew_id}"
        self._ttl = ttl
        self._r = redis.from_url(redis_url, decode_responses=True)

    # ── key/value memory ─────────────────────────────────────────────────────

    def save(self, name: str, value: Any, ttl: int | None = None) -> None:
        """Store a named fact."""
        key = f"{self._prefix}:kv:{name}"
        serialised = json.dumps(value, ensure_ascii=False)
        effective_ttl = ttl if ttl is not None else self._ttl
        if effective_ttl:
            self._r.setex(key, effective_ttl, serialised)
        else:
            self._r.set(key, serialised)

    def load(self, name: str, default: Any = None) -> Any:
        """Retrieve a named fact."""
        raw = self._r.get(f"{self._prefix}:kv:{name}")
        return json.loads(raw) if raw is not None else default

    def delete(self, name: str) -> None:
        self._r.delete(f"{self._prefix}:kv:{name}")

    # ── append-only log ──────────────────────────────────────────────────────

    def log(self, log_name: str, entry: Any) -> None:
        """Append an entry to a named log."""
        key = f"{self._prefix}:log:{log_name}"
        record = {"ts": time.time(), "data": entry}
        self._r.rpush(key, json.dumps(record, ensure_ascii=False))
        if self._ttl:
            self._r.expire(key, self._ttl)

    def get_log(self, log_name: str, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve the last *limit* entries of a named log."""
        key = f"{self._prefix}:log:{log_name}"
        raw = self._r.lrange(key, -limit, -1)
        return [json.loads(r) for r in raw]

    # ── housekeeping ─────────────────────────────────────────────────────────

    def clear_all(self) -> int:
        """Delete everything stored for this crew."""
        keys = list(self._r.scan_iter(f"{self._prefix}:*"))
        if keys:
            return self._r.delete(*keys)
        return 0

    def close(self) -> None:
        self._r.close()

    def __enter__(self) -> "CachlyCrewMemory":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

