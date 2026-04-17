"""
cachly_agents.semantic – Semantic Memory for AI agents.

Combines cachly's pgvector-backed semantic cache with agent memory:
agents can store and retrieve knowledge by *meaning*, not just exact keys.

Works independently of any specific agent framework.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

import requests


class CachlySemanticMemory:
    """
    Semantic long-term memory for AI agents, backed by cachly.dev's pgvector
    semantic cache.

    Instead of storing memories by exact key (like ``CachlyMemory``), this
    class stores knowledge as embeddings and retrieves them by *meaning*.
    An agent can ask "What do we know about the user's budget?" and retrieve
    a memory stored as "The client allocated €50k for Q3" — because the
    meanings are similar, even though the words are different.

    This is the bridge between cachly's Semantic Cache API and agent memory:

    - ``remember(text, embed_fn)`` → indexes the text as a semantic cache entry
    - ``recall(query, embed_fn)``  → searches by vector similarity and returns
      the closest matching memory (if above threshold)
    - ``stream_recall(query, embed_fn)`` → same as recall but returns an SSE
      stream of word-level chunks (useful for streaming agent responses)

    Usage::

        from cachly_agents.semantic import CachlySemanticMemory

        memory = CachlySemanticMemory(
            vector_url=os.environ["CACHLY_VECTOR_URL"],
            namespace="agent:researcher",
        )

        # Store knowledge
        memory.remember(
            "The client's budget for Q3 is €50,000",
            embed_fn=my_embed_function,
        )

        # Recall by meaning (not exact key!)
        result = memory.recall(
            "What is the budget?",
            embed_fn=my_embed_function,
        )
        if result:
            print(result.text)        # → "The client's budget for Q3 is €50,000"
            print(result.similarity)   # → 0.94

        # Stream recall (SSE) for real-time agent UIs
        for chunk in memory.stream_recall("budget info?", embed_fn=my_embed_function):
            print(chunk, end="", flush=True)

    Requires:
        - A cachly.dev instance with semantic cache enabled (vector_token)
        - An embedding function: ``(text: str) -> list[float]``
    """

    def __init__(
        self,
        vector_url: str,
        namespace: str = "cachly:sem",
        threshold: float = 0.85,
        use_adaptive_threshold: bool = True,
        session: requests.Session | None = None,
    ) -> None:
        self._url = vector_url.rstrip("/")
        self._namespace = namespace
        self._threshold = threshold
        self._adaptive = use_adaptive_threshold
        self._session = session or requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "cachly-agents-semantic/1.0",
        })

    # ── public API ──────────────────────────────────────────────────────────

    def remember(
        self,
        text: str,
        embed_fn: Callable[[str], list[float]],
        *,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a piece of knowledge in semantic memory.

        Returns the entry ID (UUID string) for later reference.

        Args:
            text: The knowledge to store (used as both prompt and content).
            embed_fn: Function that converts text to a float embedding vector.
            ttl_seconds: Optional TTL – memory expires after this many seconds.
            metadata: Optional metadata dict (stored as JSON prefix in prompt).
        """
        embedding = embed_fn(text)
        body: dict[str, Any] = {
            "prompt": self._format_prompt(text, metadata),
            "embedding": embedding,
            "namespace": self._namespace,
        }
        if ttl_seconds:
            from datetime import datetime, timezone, timedelta
            expires = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            body["expires_at"] = expires.isoformat()

        resp = self._session.post(f"{self._url}/entries", json=body)
        resp.raise_for_status()
        return resp.json().get("id", "")

    def recall(
        self,
        query: str,
        embed_fn: Callable[[str], list[float]],
        *,
        threshold: float | None = None,
    ) -> SemanticRecallResult | None:
        """
        Retrieve the most relevant memory by meaning.

        Returns ``None`` on a cache miss (no memory similar enough).

        Args:
            query: Natural-language query describing what you want to recall.
            embed_fn: Same embedding function used in ``remember()``.
            threshold: Override the default similarity threshold (0.0–1.0).
        """
        embedding = embed_fn(query)
        body: dict[str, Any] = {
            "embedding": embedding,
            "namespace": self._namespace,
            "threshold": threshold if threshold is not None else self._threshold,
            "use_adaptive_threshold": self._adaptive,
            "prompt": query,
        }

        resp = self._session.post(f"{self._url}/search", json=body)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("found"):
            return None

        # Fetch the full entry to get the stored text.
        entry_id = data.get("id", "")
        prompt = self._fetch_entry_prompt(entry_id)

        text, metadata = self._parse_prompt(prompt)
        return SemanticRecallResult(
            text=text,
            entry_id=entry_id,
            similarity=data.get("similarity", 0.0),
            threshold_used=data.get("threshold_used", self._threshold),
            metadata=metadata,
        )

    def stream_recall(
        self,
        query: str,
        embed_fn: Callable[[str], list[float]],
        *,
        threshold: float | None = None,
    ):
        """
        Recall by meaning and stream the result as word-level chunks (SSE).

        Yields text chunks as they arrive. Useful for real-time agent UIs
        that want to display cached responses progressively.

        Yields nothing if no matching memory is found.

        Args:
            query: Natural-language query describing what you want to recall.
            embed_fn: Same embedding function used in ``remember()``.
            threshold: Override the default similarity threshold.
        """
        embedding = embed_fn(query)
        body: dict[str, Any] = {
            "embedding": embedding,
            "namespace": self._namespace,
            "threshold": threshold if threshold is not None else self._threshold,
            "use_adaptive_threshold": self._adaptive,
            "prompt": query,
        }

        resp = self._session.post(
            f"{self._url}/search/stream",
            json=body,
            stream=True,
            headers={"Accept": "text/event-stream"},
        )
        resp.raise_for_status()

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[6:]
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                # "search" event tells us if it was a hit
                if "found" in data and not data["found"]:
                    return  # miss – nothing to stream

                # "chunk" events contain the text pieces
                if "text" in data:
                    yield data["text"]

                # "done" event – stream complete
                if data == {}:
                    return

    def forget(self, entry_id: str) -> bool:
        """Delete a specific memory by its entry ID."""
        resp = self._session.delete(f"{self._url}/entries/{entry_id}")
        return resp.status_code == 200

    def forget_all(self) -> int:
        """Delete all memories in this namespace. Returns count of deleted entries."""
        resp = self._session.delete(
            f"{self._url}/flush",
            params={"namespace": self._namespace},
        )
        resp.raise_for_status()
        return resp.json().get("deleted", 0)

    def count(self) -> int:
        """Return the number of memories in this namespace."""
        resp = self._session.get(
            f"{self._url}/size",
            params={"namespace": self._namespace},
        )
        resp.raise_for_status()
        return resp.json().get("size", 0)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "CachlySemanticMemory":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── internals ───────────────────────────────────────────────────────────

    def _fetch_entry_prompt(self, entry_id: str) -> str:
        """Fetch the prompt text of an entry by listing entries and matching ID."""
        resp = self._session.get(
            f"{self._url}/entries",
            params={"namespace": self._namespace},
        )
        if resp.status_code != 200:
            return ""
        for entry in resp.json().get("data", []):
            if entry.get("id") == entry_id:
                return entry.get("prompt", "")
        return ""

    @staticmethod
    def _format_prompt(text: str, metadata: dict[str, Any] | None) -> str:
        """Optionally prefix the prompt with JSON metadata."""
        if metadata:
            return f"[META:{json.dumps(metadata, ensure_ascii=False)}]\n{text}"
        return text

    @staticmethod
    def _parse_prompt(prompt: str) -> tuple[str, dict[str, Any] | None]:
        """Extract metadata prefix if present, return (text, metadata)."""
        if prompt.startswith("[META:"):
            end = prompt.index("]\n")
            meta_str = prompt[6:end]
            text = prompt[end + 2:]
            try:
                return text, json.loads(meta_str)
            except json.JSONDecodeError:
                return prompt, None
        return prompt, None


class SemanticRecallResult:
    """Result of a semantic recall operation."""

    __slots__ = ("text", "entry_id", "similarity", "threshold_used", "metadata")

    def __init__(
        self,
        text: str,
        entry_id: str,
        similarity: float,
        threshold_used: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.text = text
        self.entry_id = entry_id
        self.similarity = similarity
        self.threshold_used = threshold_used
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"SemanticRecallResult(text={self.text!r:.60}, "
            f"similarity={self.similarity:.3f}, entry_id={self.entry_id!r:.12})"
        )

    def __bool__(self) -> bool:
        return bool(self.text)
