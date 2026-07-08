# Contributing to Flight-Guard

Thanks for your interest in contributing!

## Development setup

```bash
git clone https://github.com/atharvadevne123/Flight-Guard
cd Flight-Guard
make install
cp .env.example .env
```

## Workflow

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with tests.
3. Run the quality gates locally:
   ```bash
   make lint
   make test
   ```
4. Open a pull request with a clear description of the change.

## Code style

- Python 3.11+, formatted and linted with `ruff` (config in `pyproject.toml`).
- Type annotations on all public functions.
- Google-style docstrings on modules, classes, and public functions.
- Tests live in `tests/` and use `pytest` with fixtures from `conftest.py`.

## Commit messages

Use conventional commit prefixes: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `ci:`, `refactor:`.

## Reporting bugs

Open a GitHub issue with reproduction steps, expected behaviour, and actual behaviour.
