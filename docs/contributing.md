# Contributing to resonance

Thank you for your interest in contributing to resonance.

## Development setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/ALS-RSOXS/auto-reflect.git
   cd auto-reflect
   ```
2. Install dependencies:
   ```bash
   uv sync --all-groups
   ```
3. Install hooks:
   ```bash
   prek install
   ```

## Making changes

1. Create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Implement your changes and run tests:
   ```bash
   make test
   ```
3. Run quality checks:
   ```bash
   make lint
   make format-check
   make type-check
   ```
4. Commit using Conventional Commits:
   ```bash
   git commit -m "feat: describe your change"
   ```

## Commit message format

We use [Conventional Commits](https://www.conventionalcommits.org/). Examples:

- `feat: add beamline feature`
- `fix: handle scan edge case`
- `docs: update quickstart`
- `refactor: simplify scan planner`
- `test: add executor regression test`
- `chore: update dependency pins`

## Pull request process

1. Update documentation for behavior changes
2. Add tests for new functionality
3. Ensure all checks pass
4. Open a pull request with a clear summary

## Code style

- [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
- [ty](https://docs.astral.sh/ty/) for type checking
- Typed public APIs and complete docstrings
