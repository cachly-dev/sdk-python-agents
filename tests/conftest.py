"""
Shared fixtures for cachly-agents tests.

Uses fakeredis so tests run without a real Redis / Valkey instance.
"""
from __future__ import annotations

import pytest

try:
    import fakeredis
except ImportError:
    pytest.skip("fakeredis required – pip install fakeredis", allow_module_level=True)


@pytest.fixture()
def redis_url(monkeypatch):
    """Provide a fakeredis URL and patch ``redis.from_url``."""
    server = fakeredis.FakeServer()

    _original_from_url = __import__("redis").from_url

    def _fake_from_url(url: str, **kwargs):
        return fakeredis.FakeRedis(server=server, decode_responses=kwargs.get("decode_responses", False))

    monkeypatch.setattr("redis.from_url", _fake_from_url)
    return "redis://fake:6379"
