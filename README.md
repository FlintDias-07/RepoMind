# 🤖 GitHub Repository Understanding Assistant (RAG)

A production-quality Retrieval-Augmented Generation (RAG) application that analyzes GitHub repositories and answers questions about codebases using AI.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Phase 1: Repository Ingestion](#phase-1-repository-ingestion)
- [Development Roadmap](#development-roadmap)
- [Contributing](#contributing)

## 🎯 Overview

This RAG application helps developers understand any GitHub repository by:

1. **Cloning** the repository from GitHub
2. **Parsing** source code files intelligently
3. **Chunking** code into semantic units
4. **Embedding** chunks for semantic search
5. **Retrieving** relevant code snippets
6. **Generating** AI-powered answers with code references

## ✨ Features

### Phase 1: Repository Ingestion ✅
- Clone GitHub repositories automatically
- Validate repository structure
- Parse multiple programming languages
- Extract repository metadata
- Generate repository statistics
- Support for public and private repositories (with GitHub token)

### Phase 2: Code Parsing 🔄
- Intelligent code chunking (function/class/method level)
- Extract metadata: file path, class name, function name
- Support for:
  - Python (.py)
  - Java (.java)
  - JavaScript/TypeScript (.js, .ts, .jsx, .tsx)
  - Markdown (.md)
  - YAML (.yaml, .yml)
  - JSON, XML, SQL, HTML, CSS

### Phase 3: Embeddings 📦
- Generate embeddings using Sentence Transformers
- ChromaDB vector database integration
- Persistent storage and retrieval

### Phase 4: Retrieval 🔍
- Semantic search across codebase
- Top-K similar chunks retrieval
- Similarity score ranking
- Source file references

### Phase 5: LLM Integration 🧠
- OpenAI GPT-4 or Groq Mixtral support
- Context-aware code analysis
- Natural language question answering

### Phase 6: Streamlit UI 🎨
- Chat interface
- Repository management
- Code visualization
- Repository insights

### Phase 7: Advanced Features 🚀
- Architecture summary generation
- Developer onboarding guide
- Automatic documentation
- Testing & deployment

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Web Framework** | Streamlit |
| **LLM Framework** | LangChain |
| **Embeddings** | Sentence Transformers |
| **Vector DB** | ChromaDB |
| **Repository Mgmt** | GitPython |
| **LLM Providers** | OpenAI, Groq |
| **Testing** | Pytest |

## 📁 Project Structure

```
RAG_application/
│
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration management
├── utils.py                    # Utility functions
├── github_loader.py            # GitHub repository cloning
├── parser.py                   # Code parsing (Phase 2)
├── chunker.py                  # Code chunking (Phase 2)
├── embeddings.py               # Embedding generation (Phase 3)
├── vector_db.py                # ChromaDB integration (Phase 3)
├── retriever.py                # Retrieval pipeline (Phase 4)
├── rag_pipeline.py             # RAG orchestration (Phase 5)
│
├── vector_db/                  # ChromaDB storage
├── repositories/               # Cloned repositories
├── data/                       # Metadata and statistics
├── logs/                       # Application logs
│
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── setup.py                    # Package setup
├── pytest.ini                  # Test configuration
├── README.md                   # This file
└── DEVELOPMENT.md              # Development guide
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- 2GB disk space minimum
- Internet connection

### Installation

1. **Clone this repository:**
```bash
git clone https://github.com/yourusername/RAG_application.git
cd RAG_application
```

2. **Create and activate virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your API keys
# OPENAI_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here
```

5. **Run the application:**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📚 Phase 1: Repository Ingestion

### What is Phase 1?

Phase 1 focuses on:
- ✅ Accepting GitHub repository URLs
- ✅ Cloning repositories locally
- ✅ Validating repository structure
- ✅ Extracting repository metadata
- ✅ Generating statistics

### How to Use Phase 1

1. **Open the Application:**
   - Run `streamlit run app.py`
   - Navigate to "Repository Upload" tab

2. **Enter Repository URL:**
   - Example: `https://github.com/username/repository.git`
   - Or: `https://github.com/username/repository`

3. **Clone Repository:**
   - Click "Clone Repository" button
   - Wait for cloning to complete
   - View repository information and statistics

4. **Manage Repositories:**
   - Go to "Manage Repositories" tab
   - View cloned repositories
   - Delete repositories you no longer need

### Example Output

```
Repository Information
- Name: example-repo
- Branch: main
- Latest Commit: abc1234
- Local Path: /path/to/repositories/example-repo

Repository Statistics
- Total Files: 234
- Total Size: 5.2 MB
- Python Files: 45
- Java Files: 12
- Configuration Files: 23
```

### Key Components

#### 1. **GitHubRepositoryLoader**
Handles repository cloning and validation.

```python
from github_loader import GitHubRepositoryLoader

loader = GitHubRepositoryLoader("https://github.com/user/repo.git")
success, message = loader.clone_repository()

if success:
    files = loader.get_file_list()
    stats = loader.get_repository_stats()
```

#### 2. **RepositoryManager**
Manages multiple cloned repositories.

```python
from github_loader import RepositoryManager

manager = RepositoryManager()
repos = manager.get_cloned_repositories()
manager.delete_repository("repo-name")
```

#### 3. **Configuration**
Centralized configuration in `config.py`:

```python
from config import (
    REPOSITORIES_DIR,
    SUPPORTED_EXTENSIONS,
    IGNORE_DIRS,
    RETRIEVAL_K,
)
```

### Phase 1 API Reference

#### GitHubRepositoryLoader

```python
class GitHubRepositoryLoader:
    def __init__(repo_url: str, base_path: str = REPOSITORIES_DIR)
    def clone_repository(overwrite: bool = False) -> Tuple[bool, str]
    def verify_repository() -> Tuple[bool, str]
    def get_repository_info() -> Optional[dict]
    def get_file_list(relative: bool = True) -> List[str]
    def read_file(relative_path: str) -> Optional[str]
    def get_repository_stats() -> dict
    def save_repository_metadata() -> bool
    def cleanup() -> bool
```

## 🗺️ Development Roadmap

### ✅ Phase 1: Repository Ingestion (COMPLETE)
- [x] Folder structure
- [x] Dependency installation
- [x] Repository cloning
- [x] Repository validation
- [x] File listing
- [x] Statistics generation

### 🔄 Phase 2: Code Parsing (IN PROGRESS)
- [ ] Code parsing for Python, Java, JS/TS
- [ ] Function-level extraction
- [ ] Class-level extraction
- [ ] Metadata extraction

### 📦 Phase 3: Embeddings & Vector DB
- [ ] Sentence Transformers integration
- [ ] ChromaDB setup
- [ ] Embedding generation
- [ ] Persistent storage

### 🔍 Phase 4: Retrieval Pipeline
- [ ] Semantic search implementation
- [ ] Similarity scoring
- [ ] Top-K retrieval
- [ ] Context building

### 💬 Phase 5: LLM Integration
- [ ] LangChain pipeline
- [ ] OpenAI integration
- [ ] Groq integration
- [ ] Prompt engineering

### 🎨 Phase 6: Streamlit UI Enhancement
- [ ] Chat interface
- [ ] Code visualization
- [ ] Repository insights
- [ ] Advanced features

### 🚀 Phase 7: Testing & Deployment
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance testing
- [ ] Docker containerization
- [ ] Deployment guide

## 📖 Documentation

### Setup and Installation
- [Quick Start](#quick-start) - Get started in 5 minutes
- [DEVELOPMENT.md](DEVELOPMENT.md) - Detailed development guide

### Phase Documentation
- **Phase 1:** [Repository Ingestion](#phase-1-repository-ingestion)
- Phase 2-7: (Coming in future phases)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋 Support

For issues, questions, or suggestions, please:

1. Check existing GitHub Issues
2. Create a new Issue with:
   - Description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS

## 🎓 Learning Resources

This project is designed as a learning resource. It demonstrates:

- ✅ Professional Python project structure
- ✅ Clean code principles
- ✅ RAG pattern implementation
- ✅ Streamlit application development
- ✅ Git and repository management
- ✅ Vector database integration
- ✅ LLM integration best practices
- ✅ Production-ready error handling

## 👨‍💻 Author

Created as a portfolio project to demonstrate:
- Full-stack AI application development
- Production-grade Python practices
- Enterprise-level architecture design

---

**Made with ❤️ for developers by developers**

*Last Updated: June 2026*
