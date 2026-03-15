# Style and conventions
- Use `uv run ...` for local commands.
- Chat responses should be in Simplified Chinese; project files/comments/docs use English unless task requires otherwise.
- Prefer fail-fast, single-path implementations; avoid compatibility layers.
- Do not modify `extern/`.
- Format with `uv run ruff format .` and lint with `uv run ruff check .` after code changes.