# cachly-agents

Official **cachly.dev** adapters for AI agent frameworks – persistent memory and
semantic cache for LangChain, AutoGen, and CrewAI agents.

**DSGVO-compliant · German servers · 30s setup**

## Installation

```bash
pip install cachly-agents                   # core + memory
pip install "cachly-agents[langchain]"      # + LangChain adapter
pip install "cachly-agents[autogen]"        # + AutoGen adapter
pip install "cachly-agents[crewai]"         # + CrewAI adapter
pip install "cachly-agents[all]"            # everything
```

> Requires Python 3.10+ · A free cachly.dev instance · `pip install cachly`

---

## LangChain – Persistent Chat History

```python
import os
from cachly_agents.langchain import CachlyChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_history(session_id: str) -> CachlyChatMessageHistory:
    return CachlyChatMessageHistory(
        session_id=session_id,
        redis_url=os.environ["CACHLY_URL"],
        ttl=3600,           # session expires after 1 h of inactivity
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

## AutoGen – Persistent Conversation Store

```python
import os
from cachly_agents.autogen import CachlyMessageStore
from autogen import ConversableAgent

store = CachlyMessageStore(
    redis_url=os.environ["CACHLY_URL"],
    conversation_id="project-42",
    ttl=86400,          # 24 h
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

## CrewAI – Persistent Memory Adapter

```python
import os
from cachly_agents.crewai import CachlyCrewMemory
from crewai import Agent, Crew, Task

memory = CachlyCrewMemory(
    redis_url=os.environ["CACHLY_URL"],
    crew_id="research-team",
    ttl=604800,         # 7 days
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

## Generic Key-Value Memory (framework-agnostic)

```python
import os
from cachly_agents.memory import CachlyMemory

with CachlyMemory(redis_url=os.environ["CACHLY_URL"], namespace="agent:planner") as mem:
    mem.remember("user_name", "Alice")
    mem.remember("budget", 50_000, ttl=3600)

    name   = mem.recall("user_name")      # → "Alice"
    budget = mem.recall("budget", 0)      # → 50000

    snapshot = mem.snapshot()             # full dict for checkpointing
    mem.clear()
    mem.restore(snapshot)                 # restore from checkpoint
```

---

## Semantic Long-Term Memory (NEW ✨)

Store and recall knowledge **by meaning** — not by exact key.
Uses cachly's pgvector-backed semantic cache for similarity search.

```python
import os
from cachly_agents.semantic import CachlySemanticMemory

with CachlySemanticMemory(
    vector_url=os.environ["CACHLY_VECTOR_URL"],   # https://api.cachly.dev/v1/sem/{token}
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

### Why Semantic Memory?

| Feature | Key-Value (`CachlyMemory`) | Semantic (`CachlySemanticMemory`) |
|---------|---------------------------|-----------------------------------|
| Lookup method | Exact key match | Meaning-based similarity (cosine) |
| Data model | `name → value` dict | Free-text knowledge base |
| Best for | Structured config, user prefs | Unstructured knowledge, RAG, Q&A |
| Threshold | N/A | Adaptive F1-calibrated (auto-tuned) |
| Streaming | No | Yes (SSE word-level chunks) |

---

## Why cachly for AI agents?

| Problem | Without cachly | With cachly-agents |
|---------|---------------|-------------------|
| Agent memory | Ephemeral – lost on restart | Persistent across restarts, replicas, agent runs |
| Multi-agent sessions | Each agent has isolated memory | Shared namespace, one source of truth |
| LLM costs | Every user message re-generates | Semantic cache cuts 60% of LLM calls |
| Agent knowledge | Brittle keyword search | Semantic recall by *meaning* — pgvector HNSW |
| GDPR / data residency | Cloud provider may store in US | German servers, EU data only |
| Scale-out | In-process dict breaks with replicas | Redis scales horizontally out of the box |

---

## API Reference

### `CachlyMemory`
| Method | Description |
|--------|-------------|
| `remember(name, value, ttl?)` | Store any JSON-serialisable value |
| `recall(name, default?)` | Retrieve a value (returns default on miss) |
| `forget(name)` | Delete a single memory |
| `recall_all()` | Return all memories in this namespace |
| `snapshot()` | Alias for `recall_all()` – for checkpointing |
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
CACHLY_VECTOR_URL=https://api.cachly.dev/v1/sem/{your-vector-token}   # for CachlySemanticMemory
```

Get your connection string and vector token at [cachly.dev/instances](https://cachly.dev/instances).

---

## Real-World Use Cases

### 1. Customer Support Bot (LangChain + Semantic Cache)

**Problem:** Your support bot calls GPT-4o for every question — even when users ask the same thing 50 times a day ("How do I reset my password?").

**Solution:** Cache LLM responses by meaning. The 51st user asking "password reset help" gets an instant cached answer.

```python
from cachly_agents.langchain import CachlyChatMessageHistory
from cachly_agents.memory import CachlySemanticMemory

memory = CachlySemanticMemory(
    vector_url=os.environ["CACHLY_VECTOR_URL"],
    embed_fn=openai_embed,
    threshold=0.85,
)

# First user asks "How do I reset my password?" → LLM call → cached
# Second user asks "I forgot my password, help!" → cache HIT (0.92 similarity)
result = await memory.recall("I forgot my password, help!")
if result:
    return result.response  # instant, $0.00
```

**Impact:** 60-80% fewer LLM calls → **$500+/month saved** on a typical support bot.

---

### 2. RAG Pipeline with Persistent Context (CrewAI)

**Problem:** Your research agents lose context between runs. Every restart means re-fetching and re-embedding documents.

**Solution:** Use cachly as the shared memory layer. Agents remember research across sessions.

```python
from cachly_agents.crewai import CachlyCrewMemory

memory = CachlyCrewMemory(
    redis_url=os.environ["CACHLY_URL"],
    vector_url=os.environ["CACHLY_VECTOR_URL"],
    embed_fn=openai_embed,
)

# Agent stores research findings
await memory.store("research:market-analysis", {
    "finding": "EU AI Act requires audit trails for all AI decisions",
    "source": "Official Journal L 2024/1689",
    "embedding": embed("EU AI Act audit requirements"),
})

# Next run — agent recalls relevant findings by meaning
findings = await memory.semantic_recall("What are the compliance requirements?")
# Returns the market analysis finding (similarity: 0.91) — no re-research needed
```

---

### 3. Multi-Agent Research Team (AutoGen)

**Problem:** Multiple AutoGen agents working on the same topic duplicate LLM calls. The "Researcher" and "Writer" agents both ask similar questions to GPT-4o.

**Solution:** Shared semantic cache — when one agent gets an answer, all agents benefit.

```python
from cachly_agents.autogen import CachlyAutoGenCache

cache = CachlyAutoGenCache(
    redis_url=os.environ["CACHLY_URL"],
    vector_url=os.environ["CACHLY_VECTOR_URL"],
    embed_fn=openai_embed,
)

# Researcher agent asks: "What is the market size for AI caching?"
# → LLM call, result cached semantically

# Writer agent later asks: "How big is the AI cache market?"
# → Cache HIT (similarity 0.93) → instant response, no LLM call
```

---

### 4. E-Commerce Recommendation Engine

**Problem:** Your product recommendation agent makes expensive LLM calls to generate personalized suggestions. Similar customers get similar recommendations, but each call costs $0.03.

**Solution:** Cache recommendations by user-profile similarity.

```python
# User profile embedding captures preferences
profile_embedding = embed(f"likes:{categories} budget:{range} style:{preferences}")

# Check if a similar user already got recommendations
cached = await memory.recall_by_embedding(profile_embedding, threshold=0.88)
if cached:
    return cached.recommendations  # instant, free

# No cache hit — generate fresh recommendations
recs = await llm.generate_recommendations(user_profile)
await memory.index(profile_embedding, recs, namespace="recommendations", ttl=3600)
return recs
```

**Impact:** At 10,000 users/day with 70% cache hit rate → **7,000 free responses** daily.

---

### 5. Code Review Assistant with Memory

**Problem:** Your AI code reviewer forgets patterns it already flagged. The same anti-pattern across 20 PRs triggers 20 identical LLM analyses.

**Solution:** Semantic cache for code patterns + persistent memory for project conventions.

```python
from cachly_agents.memory import CachlySemanticMemory

review_memory = CachlySemanticMemory(
    vector_url=os.environ["CACHLY_VECTOR_URL"],
    embed_fn=code_embed,  # code-specific embedding model
    threshold=0.90,
    namespace="code-review",
)

# First PR with "SELECT * FROM users WHERE id = " + user_input
# → LLM: "SQL injection vulnerability, use parameterized queries"
# → Cached

# 15th PR with similar pattern
# → Cache HIT → instant review comment, consistent feedback
```

---

## License

MIT © [cachly.dev](https://cachly.dev)

