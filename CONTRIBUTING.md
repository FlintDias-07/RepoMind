# Contributing to RepoMind AI

Thanks for considering a contribution to RepoMind AI. This project is a Streamlit-based repository intelligence tool with parsing, embeddings, hybrid retrieval, evaluation, and grounded AI reporting.

## Local Setup

1. Fork and clone the repository.
2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate it:

Windows:

```powershell
.\.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Copy `.env.example` to `.env` and add your own API keys locally. Do not commit `.env`.

## Run the App

```bash
streamlit run app.py
```

## Run Tests

```bash
pytest
```

Before opening a pull request, make sure the full regression suite passes.

## Pull Request Guidelines

- Keep changes focused and easy to review.
- Include screenshots for UI or documentation asset changes.
- Add or update tests when changing behavior.
- Do not commit secrets, local repositories, vector database files, caches, virtual environments, or logs.
- Clearly describe what changed, why it changed, and how it was validated.

