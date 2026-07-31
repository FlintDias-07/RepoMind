# Release Checklist

Use this checklist before the first GitHub commit and push.

## Commit These Files and Directories

- `README.md`
- `LICENSE`
- `CONTRIBUTING.md`
- `DEPLOYMENT.md`
- `DEVELOPMENT.md`
- `RELEASE_CHECKLIST.md`
- `.gitignore`
- `.env.example`
- `.streamlit/config.toml`
- `app.py`
- `config.py`
- `github_loader.py`
- `utils.py`
- `requirements.txt`
- `pytest.ini`
- `setup.py`
- `ai/`
- `assets/`
- `embedding/`
- `evaluation/`
- `parser/`
- `retrieval/`
- `tests/`
- `ui/`
- `vectorstore/`

## Keep Ignored

- `.env`
- `.venv/`
- `.venv311/`
- `__pycache__/`
- `.pytest_cache/`
- `.cache/`
- `logs/`
- `data/`
- `repositories/`
- `vector_db/`
- IDE folders such as `.vscode/` and `.idea/`
- OS files such as `.DS_Store`, `Thumbs.db`, and `Desktop.ini`

## Pre-Push Checks

- [ ] `git status --short` contains only intentional changes.
- [ ] No secrets or API keys are staged.
- [ ] `.env` is not staged.
- [ ] All README image links resolve.
- [ ] Screenshot and demo assets are present.
- [ ] `pytest` passes.
- [ ] Streamlit launches locally.
- [ ] The final commit message summarizes the release clearly.

## Remaining Manual Steps

- Remove local cache directories from the workspace if you want a visually clean release folder.
- Stage intentional new directories and files.
- Review `git diff --stat` before committing.

