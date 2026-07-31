# Deployment Guide

This document describes supported and unsupported deployment options for RepoMind AI.

## Local Deployment

Local Streamlit deployment is supported.

### Prerequisites

- Python 3.11 recommended
- Git
- A Groq or OpenAI API key

### Steps

```bash
git clone https://github.com/your-username/RAG_application.git
cd RAG_application
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add local secrets:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

Run the app:

```bash
streamlit run app.py
```

The app opens at:

```text
http://localhost:8501
```

## Docker Deployment

Docker deployment is not currently supported in this repository.

Reason: there is no `Dockerfile`, `docker-compose.yml`, or container-specific runtime configuration checked in. Add those files before publishing Docker instructions.

## Streamlit Community Cloud

Streamlit Community Cloud deployment may require additional preparation and is not currently documented as fully supported.

Important considerations:

- Runtime storage such as `repositories/`, `vector_db/`, `data/`, and `logs/` is local and ignored by Git.
- Secrets must be configured through Streamlit secrets management, not committed to the repository.
- Some ML dependencies can be large and may need platform-specific package handling.
- A public deployment should avoid indexing private repositories unless authentication and data handling are explicitly designed for that environment.

## Production Notes

- Do not commit `.env`.
- Do not commit cloned repositories or vector database files.
- Keep API keys in environment variables or managed secrets.
- Validate retrieval, indexing, and LLM provider behavior in the target environment before sharing a public URL.

