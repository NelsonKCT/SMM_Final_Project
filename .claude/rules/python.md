<!-- copied from AI-Harness 2026-07-04; re-run /adopt-project to refresh -->

---
paths:
  - "**/*.py"
---

# Python conventions

- Always use the project's venv (`.venv/`); never install into system Python.
  Create with `python3 -m venv .venv` if missing.
- New projects use `pyproject.toml`; if a `requirements.txt` already exists, keep
  it as the source of truth and keep it updated when adding dependencies.
- Type hints on public function signatures; f-strings over `%`/`.format()`;
  `pathlib.Path` over `os.path`.
- Long-running services (FastAPI, bots, workers): config via environment
  variables loaded at startup, never hardcoded tokens or paths.
