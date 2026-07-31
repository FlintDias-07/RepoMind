"""AI intelligence layer built on top of retrieval."""

from .citation_manager import Citation, CitationManager
from .ai_pipeline import AIPipeline, AIPipelineResult
from .ai_service import AIService, IntelligenceReport, RepositoryHealth, RepositoryProfile
from .context_builder import AIContext, AIContextBuilder, ContextChunk, RepositorySnapshot
from .conversation_memory import ConversationMemory, MemoryTurn, RepositoryConversation
from .llm_provider import BaseLLMProvider, LLMProviderFactory, LLMResponse, ProviderSettings
from .prompt_builder import PromptBuilder, PromptPackage, PromptRole
from .response_formatter import FormattedResponse, ResponseFormatter

__all__ = [
    "AIContext",
    "AIContextBuilder",
    "AIPipeline",
    "AIPipelineResult",
    "AIService",
    "BaseLLMProvider",
    "Citation",
    "CitationManager",
    "ConversationMemory",
    "ContextChunk",
    "FormattedResponse",
    "IntelligenceReport",
    "LLMProviderFactory",
    "LLMResponse",
    "MemoryTurn",
    "PromptBuilder",
    "PromptPackage",
    "PromptRole",
    "ProviderSettings",
    "RepositoryHealth",
    "RepositoryConversation",
    "RepositoryProfile",
    "RepositorySnapshot",
    "ResponseFormatter",
]
