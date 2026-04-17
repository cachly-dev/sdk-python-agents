"""Tests for cachly_agents.crewai – CachlyCrewMemory."""
from __future__ import annotations

from cachly_agents.crewai import CachlyCrewMemory


class TestCachlyCrewMemory:
    """Unit tests for the CrewAI memory adapter."""

    # ── key/value memory ─────────────────────────────────────────────────

    def test_save_and_load(self, redis_url):
        mem = CachlyCrewMemory(redis_url=redis_url, crew_id="crew-1")
        mem.save("topic", "LLM memory systems")
        assert mem.load("topic") == "LLM memory systems"

    def test_load_missing_key_returns_default(self, redis_url):
        mem = CachlyCrewMemory(redis_url=redis_url, crew_id="crew-2")
        assert mem.load("nope") is None
        assert mem.load("nope", default="fallback") == "fallback"

    def test_save_complex_value(self, redis_url):
        mem = CachlyCrewMemory(redis_url=redis_url, crew_id="crew-3")
        data = {"items": [1, 2, 3], "nested": {"ok": True}}
        mem.save("complex", data)
        assert mem.load("complex") == data

    def test_delete(self, redis_url):
        mem = CachlyCrewMemory(redis_url=redis_url, crew_id="crew-del")
        mem.save("k", "v")
        assert mem.load("k") == "v"
        mem.delete("k")
        assert mem.load("k") is None

    # ── log (list) memory ────────────────────────────────────────────────

    def test_log_and_get_log(self, redis_url):
        mem = CachlyCrewMemory(redis_url=redis_url, crew_id="crew-log")
        mem.log("events", {"action": "search", "query": "cachly"})
        mem.log("events", {"action": "summarise"})
        log = mem.get_log("events")
        assert len(log) == 2
        # log entries are wrapped as {"ts": ..., "data": ...}
        assert log[0]["data"]["action"] == "search"
        assert log[1]["data"]["action"] == "summarise"

    def test_get_log_empty(self, redis_url):
        mem = CachlyCrewMemory(redis_url=redis_url, crew_id="crew-nolog")
        assert mem.get_log("nothing") == []

    # ── clear ────────────────────────────────────────────────────────────

    def test_clear_all(self, redis_url):
        mem = CachlyCrewMemory(redis_url=redis_url, crew_id="crew-clear")
        mem.save("a", 1)
        mem.save("b", 2)
        mem.log("log", {"x": 1})
        mem.clear_all()
        assert mem.load("a") is None
        assert mem.load("b") is None
        assert mem.get_log("log") == []

    # ── isolation ────────────────────────────────────────────────────────

    def test_crew_ids_are_isolated(self, redis_url):
        m1 = CachlyCrewMemory(redis_url=redis_url, crew_id="alpha")
        m2 = CachlyCrewMemory(redis_url=redis_url, crew_id="beta")
        m1.save("key", "alpha-value")
        assert m1.load("key") == "alpha-value"
        assert m2.load("key") is None

    def test_ttl_accepted(self, redis_url):
        mem = CachlyCrewMemory(redis_url=redis_url, crew_id="crew-ttl", ttl=3600)
        mem.save("k", "v")
        assert mem.load("k") == "v"
