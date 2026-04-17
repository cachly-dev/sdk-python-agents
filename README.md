# cachly-agents

> Official **cachly.dev** adapters for AI agent frameworks — persistent memory and semantic cache for LangChain, AutoGen, and CrewAI.

[![PyPI](https://img.shields.io/pypi/v/cachly-agents?color=blue)](https://pypi.org/project/cachly-agents/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://pypi.org/project/cachly-agents/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)
[![GDPR: EU-only](https://img.shields.io/badge/GDPR-EU%20only-green)](https://cachly.dev/legal)

---

## Why cachly-agents?

| Problem | Without cachly | With cachly-agents |
|---------|---------------|-------------------|
| Agent memory | Ephemeral — lost on restart | Persistent across restarts, replicas, and agent runs |
| Multi-agent sessions | Each agent has isolated memory | Shared namespace, one source of truth |
| LLM costs | Every request regenerates | Semantic cache cuts 60 %+ of LLM calls |
| Agent knowledge | Brittle keyword search | Recall by *meaning* — pgvector HNSW |
| GDPR / data residency | Cloud provider may store in the US | German servers, EU data only |
| Scale-out | In-process dict breaks with replicas | Redis scales horizontally out of the box |

---

## Installation

```bash
pip install cachly-agents                    # core + memory
pip install "cachly-agents[langchain]"       # + LangChain adapter
pip install "cachly-agents[autogen]"         # + AutoGen adapter
pip install "cachly-agents[crewai]"          # + CrewAI adapter
pip install "cachly-agents[all]"             # everything
```

> Requires Python 3.10+ · A free cachly.dev instance · `pip install cachly`

---

## Quick Start — 30 seconds

```python
import os
from cachly_agents.memory import CachlyMemory

with CachlyMemory(redis_url=os.environ["CACHLY_URL"], namespace="agent:planner") as mem:
    mem.remember("user_name", "Alice")
    mem.remember("budget", 50_000, ttl=3600)

    name   = mem.recall("user_name")       # → "Alice"
    budget = mem.recall("budget", 0)       # → 50000

    snapshot = mem.snapshot()              # full dict for checkpointing
    mem.clear()
    mem.restore(snapshot)                  # restore from checkpoint
```

Create your free instance at **[cachly.dev](https://cachly.dev)** — no credit card required.

---

## LangChain — Persistent Chat History

```python
import os
from cachly_agents.langchain import CachlyChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_history(session_id: str) -> CachlyChatMessageHistory:
    return CachlyChatMessageHistory(
        session_id=session_id,
        redis_url=os.environ["CACHLY_URL"],
        ttl=3600,          # session expires after 1 h of inactivity
    )

chain = RunnableWithMessageHistory(
    ChatOpenAI(model="gpt-4o"),
    get_history,
)

# Messages persist across restarts, scale-out, and agent runs
response = chain.invoke(
    {"role": "user", "content": "What did we discuss last time?"},
    config={"configurable": {"session_id": "user-42"}},
)
```

---

## AutoGen — Persistent Conversation Store

```python
import os
from cachly_agents.autogen import CachlyMessageStore
from autogen import ConversableAgent

store = CachlyMessageStore(
    redis_url=os.environ["CACHLY_URL"],
    conversation_id="project-42",
    ttl=86400,         # 24 h
)

# Inject persisted history on agent start
assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful AI assistant.",
    initial_history=store.load(),
)

# Persist messages after each turn
store.append({"role": "user",      "content": "Draft a project plan."})
store.append({"role": "assistant", "content": "Sure! Here's the plan..."})
```

---

## CrewAI — Persistent Memory Adapter

```python
import os
from cachly_agents.crewai import CachlyCrewMemory
from crewai import Agent, Crew, Task

memory = CachlyCrewMemory(
    redis_url=os.environ["CACHLY_URL"],
    crew_id="research-team",
    ttl=604800,        # 7 days
)

# Store and recall facts
memory.save("last_topic", "LLM memory systems")
topic = memory.load("last_topic")          # → "LLM memory systems"

# Append-only research log
memory.log("research_log", {"topic": topic, "quality": "high"})
log = memory.get_log("research_log", limit=50)

researcher = Agent(
    role="Senior Research Analyst",
    goal=f"Continue research on {topic}",
    memory=True,
)
```

---

## Semantic Long-Term Memory

Store and recall knowledge **by meaning** — not by exact key.
Powered by pgvector HNSW similarity search.

```python
import os
from cachly_agents.semantic import CachlySemanticMemory

with CachlySemanticMemory(
    vector_url=os.environ["CACHLY_VECTOR_URL"],  # https://api.cachly.dev/v1/sem/{token}
    embed_fn=lambda text: openai.embeddings.create(
        input=text, model="text-embedding-3-small"
    ).data[0].embedding,
) as mem:
    # Store facts — they become vector embeddings
    mem.remember("cachly is a managed Valkey cache for AI apps")
    mem.remember("GDPR compliance is critical for EU companies")
    mem.remember("pgvector enables sub-millisecond ANN search", metadata={"topic": "tech"})

    # Recall by meaning — fuzzy, not exact
    results = mem.recall("What cache service works for AI?", top_k=3)
    for r in results:
        print(f"  [{r.similarity:.2f}] {r.text}")
        # → [0.94] cachly is a managed Valkey cache for AI apps

    # Streaming recall (SSE) — replay cached answer at LLM-like speed
    for chunk in mem.stream_recall("How does cachly handle GDPR?"):
        print(chunk, end="", flush=True)

    # Forget specific entries or flush all
    mem.forget(results[0].entry_id)
    mem.forget_all()
```

### Key-Value vs. Semantic Memory

| Feature | `CachlyMemory` | `CachlySemanticMemory` |
|---------|----------------|------------------------|
| Lookup | Exact key match | Meaning-based similarity (cosine) |
| Data model | `name → value` dict | Free-text knowledge base |
| Best for | Structured config, user prefs | Unstructured knowledge, RAG, Q&A |
| Streaming | No | Yes (SSE word-level chunks) |

---

## AI Dev Brain — Persistent Memory for Your Coding Assistant

cachly ships a **30-tool MCP server** that gives Claude Code, Cursor, GitHub Copilot, and Windsurf a persistent memory across sessions — so they never forget your architecture, lessons learned, or last session context.

```bash
# One-time setup
npx @cachly-dev/init
```

Or configure manually in your editor (`~/.vscode/mcp.json` / `.cursor/mcp.json`):

```json
{
  "servers": {
    "cachly": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@cachly-dev/mcp-server"],
      "env": { "CACHLY_JWT": "your-jwt-token" }
    }
  }
}
```

Add to your AI assistant instructions (e.g. `.github/copilot-instructions.md`):

```markdown
## cachly AI Brain

At the START of every session:
session_start(instance_id = "your-instance-id", focus = "what you're working on today")

At the END of every session:
session_end(instance_id = "your-instance-id", summary = "...", files_changed = [...])

After any bug fix or deploy:
learn_from_attempts(instance_id = "your-instance-id", topic = "category:keyword",
  outcome = "success", what_worked = "...", what_failed = "...", severity = "major")
```

`session_start` returns a full briefing in **one call**: last session summary, relevant lessons, open failures, brain health. 60 % fewer file reads, instant context, zero re-discovery.

→ Full docs: [cachly.dev/docs/ai-memory](https://cachly.dev/docs/ai-memory)

---

## Real-World Use Cases

### 1. Customer Support Bot — Cut LLM Costs 60–80 %

Users ask the same questions daily ("How do I reset my password?"). Cache LLM responses by meaning. The 51st user asking "password reset help" gets an instant cached answer — free.

```python
from cachly_agents.semantic import CachlySemanticMemory

memory = CachlySemanticMemory(
    vector_url=os.environ["CACHLY_VECTOR_URL"],
    embed_fn=openai_embed,
    threshold=0.85,
)

# "How do I reset my password?" → LLM call → cached
# "I forgot my password, help!" → cache HIT (0.92 similarity) → instant, $0.00
result = await memory.recall("I forgot my password, help!")
```

**Impact:** 60–80 % fewer LLM calls → **$500+/month saved** on a typical support bot.

---

### 2. RAG Pipeline with Persistent Context (CrewAI)

Research agents lose context between runs. Every restart means re-fetching and re-embedding documents. Use cachly as the shared memory layer — agents remember research across sessions.

```python
from cachly_agents.crewai import CachlyCrewMemory

memory = CachlyCrewMemory(
    redis_url=os.environ["CACHLY_URL"],
    vector_url=os.environ["CACHLY_VECTOR_URL"],
    embed_fn=openai_embed,
)

# Agent stores research findings
await memory.store("research:eu-ai-act", {
    "finding": "EU AI Act requires audit trails for all AI decisions",
    "source": "Official Journal L 2024/1689",
})

# Next run — recall by meaning, no re-research needed
findings = await memory.semantic_recall("What are the compliance requirements?")
# → returns the finding above (similarity: 0.91)
```

---

### 3. Multi-Agent Research Team (AutoGen)

Multiple agents working on the same topic duplicate LLM calls. Shared semantic cache — when one agent gets an answer, all agents benefit.

```python
from cachly_agents.autogen import CachlyAutoGenCache

cache = CachlyAutoGenCache(
    redis_url=os.environ["CACHLY_URL"],
    vector_url=os.environ["CACHLY_VECTOR_URL"],
    embed_fn=openai_embed,
)

# Researcher: "What is the market size for AI caching?" → LLM call, cached
# Writer:     "How big is the AI cache market?" → cache HIT (0.93 sim) → free
```

---

### 4. Code Review Assistant with Memory

Same anti-pattern across 20 PRs triggers 20 identical LLM analyses. Cache code patterns by semantic similarity — consistent feedback, zero repeated calls.

```python
review_memory = CachlySemanticMemory(
    vector_url=os.environ["CACHLY_VECTOR_URL"],
    embed_fn=code_embed,
    threshold=0.90,
    namespace="code-review",
)

# PR #1: "SELECT * WHERE id = " + user_input → "SQL injection, use params" → cached
# PR #15: similar pattern → cache HIT → instant review comment, consistent feedback
```

---

## API Reference

### `CachlyMemory`

| Method | Description |
|--------|-------------|
| `remember(name, value, ttl?)` | Store any JSON-serialisable value |
| `recall(name, default?)` | Retrieve a value (returns default on miss) |
| `forget(name)` | Delete a single memory |
| `recall_all()` | Return all memories in this namespace |
| `snapshot()` | Alias for `recall_all()` — for checkpointing |
| `restore(snapshot, ttl?)` | Bulk-restore from a snapshot |
| `clear()` | Delete all memories in this namespace |

### `CachlyChatMessageHistory`

| Method | Description |
|--------|-------------|
| `messages` | List of `BaseMessage` objects |
| `add_messages(messages)` | Append messages to history |
| `clear()` | Delete all messages for this session |

### `CachlyMessageStore`

| Method | Description |
|--------|-------------|
| `load()` | Load all messages as list of dicts |
| `append(message)` | Append a single message |
| `append_many(messages)` | Bulk-append via pipeline |
| `clear()` | Delete all messages |

### `CachlyCrewMemory`

| Method | Description |
|--------|-------------|
| `save(name, value, ttl?)` | Store a named fact |
| `load(name, default?)` | Retrieve a named fact |
| `delete(name)` | Remove a fact |
| `log(log_name, entry)` | Append to an append-only log |
| `get_log(log_name, limit?)` | Retrieve last N log entries |
| `clear_all()` | Delete everything for this crew |

### `CachlySemanticMemory`

| Method | Description |
|--------|-------------|
| `remember(text, metadata?, ttl?)` | Store text as a vector embedding |
| `recall(query, top_k?, threshold?)` | Retrieve similar entries by meaning |
| `stream_recall(query, top_k?, threshold?)` | SSE streaming recall (word-level chunks) |
| `forget(entry_id)` | Delete a specific entry by ID |
| `forget_all()` | Flush the entire namespace |
| `count()` | Number of live entries |

### `SemanticRecallResult`

| Property | Description |
|----------|-------------|
| `text` | The recalled text content |
| `entry_id` | UUID of the entry (use with `forget()`) |
| `similarity` | Cosine similarity score (0.0–1.0) |
| `threshold_used` | The threshold that was applied |
| `metadata` | Optional dict attached at storage time |

---

## Environment Variables

```bash
CACHLY_URL=redis://:your-password@my-app.cachly.dev:30101
CACHLY_VECTOR_URL=https://api.cachly.dev/v1/sem/{your-vector-token}
```

Get your connection string and vector token at [cachly.dev/instances](https://cachly.dev/instances).

---

## Links

- 📖 [cachly.dev docs](https://cachly.dev/docs)
- 🧠 [AI Memory / MCP Server](https://cachly.dev/docs/ai-memory)
- 🐛 [Issues](https://github.com/cachly-dev/sdk-python/issues)
- 📦 [PyPI](https://pypi.org/project/cachly-agents/)

---

MIT © [cachly.dev](https://cachly.dev)
