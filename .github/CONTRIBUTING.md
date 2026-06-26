# Contributing

Thanks for contributing to `subtitle-engine`.

## Before You Start

- Read the `README.md` for installation and usage details.
- Search existing issues and pull requests before opening a new one.
- For security issues, do not open a public issue. Follow `SECURITY.md`.

## Development Setup

```bash
git clone https://github.com/leevipun/subtitle-engine.git
cd subtitle-engine
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Making Changes

- Keep changes focused and minimal.
- Add or update tests when behavior changes.
- Update documentation when CLI behavior or user-facing functionality changes.
- Follow the existing project structure and style.

## Running Tests

```bash
pytest
```

## Pull Requests

- Explain what changed and why.
- Link the related issue when applicable.
- Include testing notes.
- Make sure the pull request template checklist is completed.

## Issues

- Use the bug report template for defects.
- Use the feature request template for enhancements.
- Include clear reproduction steps, expected behavior, and environment details.

## Code of Conduct

By participating in this project, you agree to follow the `CODE_OF_CONDUCT.md`.
