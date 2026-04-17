"""
cachly_agents – cachly.dev adapters for AI agent frameworks.

Submodules:
  cachly_agents.langchain   – LangChain ChatMessageHistory + VectorStore
  cachly_agents.autogen     – AutoGen ConversationHistory store
  cachly_agents.crewai      – CrewAI Memory adapter
  cachly_agents.memory      – Generic key-value long-term memory
  cachly_agents.semantic    – Semantic long-term memory (pgvector similarity)
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cachly-agents")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]

