# 👨‍💻 Development Guide - RAG Application

Complete guide for developing and extending the RAG Application.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Architecture](#project-architecture)
3. [Code Style & Standards](#code-style--standards)
4. [Phase 1: Deep Dive](#phase-1-deep-dive)
5. [Testing](#testing)
6. [Common Tasks](#common-tasks)
7. [Debugging](#debugging)

## Development Setup

### Initial Setup

```bash
# Clone repository
git clone https://github.com/yourusername/RAG_application.git
cd RAG_application

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio black flake8 mypy

# Create .env file
cp .env.example .env
# Edit .env with your API keys
```

### IDE Setup (VS Code)

#### Recommended Extensions
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Black Formatter (ms-python.black-formatter)
- Flake8 (ms-python.flake8)
- Pytest (littlefoxteam.vscode-python-test-adapter)

#### Settings (.vscode/settings.json)
```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": ["--max-line-length=100"],
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.python",
        "editor.formatOnSave": true
    }
}
```

## Project Architecture

### Module Overview

```
┌─────────────────────────────────────────────┐
│         Streamlit Web Interface             │
│              (app.py)                       │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
    ┌───▼──┐   ┌──▼───┐  ┌───▼───┐
    │Parser│   │Chunker│  │Embed. │
    └───┬──┘   └──┬────┘  └───┬───┘
        │         │           │
    ┌───▼─────────▼───────────▼────┐
    │   RAG Pipeline (rag_pipeline) │
    └───┬──────────────────────────┘
        │
    ┌───▼──────────────────┐
    │ ChromaDB Vector Store │
    └──────────────────────┘
```

### Data Flow

1. **Repository Ingestion** → Clone via GitPython
2. **Code Parsing** → Extract file contents
3. **Chunking** → Break into semantic chunks
4. **Embedding** → Generate embeddings
5. **Vector Storage** → Store in ChromaDB
6. **Retrieval** → Semantic search
7. **LLM Processing** → Generate answer
8. **Response** → Return to user

## Code Style & Standards

### Python Style Guide

We follow PEP 8 with these additions:

```python
# Line length: 100 characters
# Type hints: Always use for functions
# Docstrings: Google style

def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Longer description explaining what the function does,
    how it works, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something is invalid
        
    Examples:
        >>> example_function("test", 42)
        True
    """
    return True
```

### Code Quality Tools

```bash
# Format code with Black
black .

# Lint code with Flake8
flake8 . --max-line-length=100

# Type checking with mypy
mypy .

# All in one
black . && flake8 . && mypy .
```

## Phase 1: Deep Dive

### Understanding the Components

#### 1. GitHubRepositoryLoader

**Purpose:** Clone and manage GitHub repositories

**Key Methods:**

```python
from github_loader import GitHubRepositoryLoader

# Initialize
loader = GitHubRepositoryLoader("https://github.com/user/repo.git")

# Clone repository
success, msg = loader.clone_repository(overwrite=False)

# Verify it's valid
is_valid, msg = loader.verify_repository()

# Get repository information
info = loader.get_repository_info()
# {
#     "name": "repo",
#     "url": "https://github.com/user/repo.git",
#     "branch": "main",
#     "commit": "abc1234",
#     "local_path": "/path/to/repositories/repo"
# }

# Get supported files
files = loader.get_file_list(relative=True)
# ["src/main.py", "src/utils.py", "README.md"]

# Read a file
content = loader.read_file("src/main.py")

# Get statistics
stats = loader.get_repository_stats()
# {
#     "total_files": 45,
#     "total_size": "5.2 MB",
#     "files_by_extension": {".py": 30, ".md": 5}
# }
```

**Implementation Details:**

- Uses GitPython for repository operations
- Handles SSH and HTTPS URLs
- Validates repository integrity
- Ignores unnecessary directories
- Supports private repositories with GitHub tokens

#### 2. Configuration System

**Purpose:** Centralized configuration management

**Key Variables:**

```python
from config import (
    # Paths
    REPOSITORIES_DIR,      # Where repos are cloned
    VECTOR_DB_DIR,        # ChromaDB storage
    DATA_DIR,             # Metadata storage
    
    # Supported files
    SUPPORTED_EXTENSIONS,  # {'.py', '.java', ...}
    IGNORE_DIRS,          # {'.git', 'node_modules', ...}
    
    # LLM Settings
    LLM_PROVIDER,         # "openai" or "groq"
    LLM_MODEL,            # Model name
    TEMPERATURE,          # 0.0-1.0
    
    # Retrieval
    RETRIEVAL_K,          # Top K chunks
    SIMILARITY_THRESHOLD, # Min similarity
)
```

#### 3. Utility Functions

**Purpose:** Common operations and helpers

```python
from utils import (
    # Validation
    validate_github_url(),
    is_supported_file(),
    is_ignored_directory(),
    
    # File operations
    get_file_extension(),
    count_files(),
    format_file_size(),
    
    # Statistics
    get_repo_stats(),
    
    # Logging
    get_logger(),
    
    # Metadata
    save_metadata(),
    load_metadata(),
    RepositoryMetadata,
)
```

### Extending Phase 1

#### Adding Support for New File Types

```python
# In config.py
SUPPORTED_EXTENSIONS = {
    ".py",
    ".java",
    ".go",  # New: Go files
}
```

#### Custom Repository Preprocessing

```python
# In github_loader.py
class GitHubRepositoryLoader:
    def preprocess_repository(self) -> bool:
        """Custom preprocessing steps"""
        # Remove large files
        # Clean up build artifacts
        # Normalize line endings
        pass
```

#### Adding Repository Validation Rules

```python
# Create validation.py
class RepositoryValidator:
    def validate_structure(self, repo_path: str) -> List[str]:
        """Return list of warnings about repo structure"""
        warnings = []
        
        if not (Path(repo_path) / "README.md").exists():
            warnings.append("Missing README.md")
        
        return warnings
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_github_loader.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

```python
# tests/test_github_loader.py
import pytest
from github_loader import GitHubRepositoryLoader

class TestGitHubRepositoryLoader:
    @pytest.fixture
    def loader(self):
        return GitHubRepositoryLoader("https://github.com/test/repo.git")
    
    def test_sanitize_repo_name(self, loader):
        """Test repository name sanitization"""
        assert loader.repo_name == "repo"
    
    def test_clone_repository_invalid_url(self):
        """Test cloning with invalid URL"""
        with pytest.raises(ValueError):
            GitHubRepositoryLoader("not-a-url")
    
    @pytest.mark.asyncio
    async def test_clone_repository_async(self, loader):
        """Test asynchronous cloning"""
        # Test async operations
        pass
```

## Common Tasks

### Clone and Analyze a Repository

```python
from github_loader import GitHubRepositoryLoader

# Create loader
loader = GitHubRepositoryLoader("https://github.com/pallets/flask.git")

# Clone repository
success, msg = loader.clone_repository()
if not success:
    print(f"Error: {msg}")
    exit(1)

# Get information
info = loader.get_repository_info()
print(f"Repository: {info['name']}")
print(f"Branch: {info['branch']}")

# Get file list
files = loader.get_file_list()
print(f"Total files: {len(files)}")

# Get statistics
stats = loader.get_repository_stats()
print(f"Total size: {stats['total_size']}")
print(f"Files by extension:")
for ext, count in stats['files_by_extension'].items():
    print(f"  {ext}: {count}")
```

### Manage Multiple Repositories

```python
from github_loader import RepositoryManager

manager = RepositoryManager()

# Get all cloned repositories
repos = manager.get_cloned_repositories()
for repo in repos:
    print(f"- {repo['repo_name']}: {repo['stats']['total_files']} files")

# Delete a repository
success, msg = manager.delete_repository("flask")
print(msg)
```

### Add Logging to Your Code

```python
from utils import get_logger

logger = get_logger(__name__)

logger.info("This is an info message")
logger.warning("This is a warning")
logger.error("This is an error")

# For exceptions
try:
    # Some operation
    pass
except Exception as e:
    logger.error(f"Operation failed: {str(e)}", exc_info=True)
```

## Debugging

### Enable Debug Logging

```python
import logging
from config import LOGS_DIR

logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
import os
os.environ['LOG_LEVEL'] = 'DEBUG'
```

### Debug in Streamlit

```python
# In app.py
import streamlit as st

# Display debug info
if st.checkbox("Show debug info"):
    st.write(st.session_state)
    
    # Show config
    import config
    st.json({k: str(v) for k, v in config.__dict__.items()})
```

### Common Issues & Solutions

#### Issue: "Permission denied" when cloning

**Solution:**
```bash
# Generate GitHub SSH key
ssh-keygen -t ed25519 -C "your@email.com"

# Add to GitHub account
cat ~/.ssh/id_ed25519.pub

# Use SSH URL instead of HTTPS
loader = GitHubRepositoryLoader("git@github.com:user/repo.git")
```

#### Issue: Repository already exists

**Solution:**
```python
# Use overwrite parameter
success, msg = loader.clone_repository(overwrite=True)
```

#### Issue: API rate limiting

**Solution:**
```bash
# Set GitHub token
export GITHUB_TOKEN=your_token_here

# Or add to .env
GITHUB_TOKEN=your_token_here
```

### Profiling and Performance

```python
import cProfile
import pstats

def profile_code():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Code to profile
    from github_loader import GitHubRepositoryLoader
    loader = GitHubRepositoryLoader("https://github.com/user/repo.git")
    loader.clone_repository()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(10)  # Top 10 functions

if __name__ == "__main__":
    profile_code()
```

## Next Steps

After completing Phase 1, you're ready for:

### Phase 2: Code Parsing
- Parse Python, Java, JavaScript, TypeScript
- Extract functions, classes, methods
- Create semantic chunks

### Phase 3: Embeddings
- Generate embeddings using Sentence Transformers
- Store in ChromaDB
- Implement vector search

### Phase 4: Retrieval
- Semantic search implementation
- Context building
- Relevance ranking

## Resources

- [GitPython Documentation](https://gitpython.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Python Best Practices](https://pep8.org/)

---

**Happy Coding! 🚀**
