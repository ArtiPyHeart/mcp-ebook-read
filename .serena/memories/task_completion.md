# Task completion checklist
- Run `uv run ruff format .` and `uv run ruff check .` after code changes.
- Run targeted `uv run pytest ...` for touched areas.
- If changing releaseable package behavior, update README and `pyproject.toml` version only when preparing release.
- Release flow: merge to main, push main, create version tag, push tag.