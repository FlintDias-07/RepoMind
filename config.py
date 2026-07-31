"""
Configuration management for RAG Application
Handles environment variables and application settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project Paths
PROJECT_ROOT = Path(__file__).parent
REPOSITORIES_DIR = PROJECT_ROOT / "repositories"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
EVALUATION_REPORTS_DIR = DATA_DIR / "evaluation_reports"

# Create directories if they don't exist
for directory in [REPOSITORIES_DIR, VECTOR_DB_DIR, DATA_DIR, LOGS_DIR, BENCHMARKS_DIR, EVALUATION_REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# GitHub Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "openai" or "groq"
LLM_MODEL = os.getenv("LLM_MODEL", "mixtral-8x7b-32768")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
AI_CONTEXT_TOKEN_LIMIT = int(os.getenv("AI_CONTEXT_TOKEN_LIMIT", "2800"))
AI_MEMORY_MAX_TURNS = int(os.getenv("AI_MEMORY_MAX_TURNS", "8"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", LLM_MODEL or "gpt-4o-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", LLM_MODEL or "mixtral-8x7b-32768")

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, fast embeddings
# Alternative: "sentence-transformers/all-mpnet-base-v2"  # Better quality but slower

# ChromaDB Configuration
CHROMA_DB_PATH = str(VECTOR_DB_DIR / "chroma_db")
CHROMA_PERSISTENCE = True

# Retrieval Configuration
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))  # Top K chunks to retrieve
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# Repository Configuration
IGNORE_DIRS = {
    ".git",
    "node_modules",
    "build",
    "target",
    "dist",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    ".venv311",
    ".next",
    "virtualenv",
    "vector_db",
    "site-packages",
    ".tox",
    ".eggs",
    "*.egg-info",
    ".gradle",
    ".maven",
    "bin",
    "obj",
}

SUPPORTED_EXTENSIONS = {
    ".py",
    ".java",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".xml",
    ".html",
    ".css",
    ".sql",
}

# Chunking Configuration
MAX_CHUNK_SIZE = 1500  # Maximum characters per chunk
OVERLAP_SIZE = 100  # Character overlap between chunks
MIN_CHUNK_SIZE = 100  # Minimum chunk size to keep

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Streamlit Configuration
STREAMLIT_PAGE_CONFIG = {
    "page_title": "RepoMind AI",
    "page_icon": "RM",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Feature Flags
ENABLE_CACHING = True
ENABLE_ANALYTICS = False
ENABLE_ADVANCED_FEATURES = True
