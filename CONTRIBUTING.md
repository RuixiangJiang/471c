# Setup Development Environment

Before contributing, make sure your environment is ready.

## Install UV

We use UV for dependency management, running tests, and pre-commit hooks.

- [Installing UV](https://docs.astral.sh/uv/getting-started/installation/)

## Fork and Clone the Repository

1. Go to the project page in GitLab:  
   `https://gitlab.eecis.udel.edu/clause/471c`

2. Click the **Fork** button at the top-right of the page.

3. Choose your personal namespace or group for the fork. Make sure the fork has private visibility.

4. Add `clause@udel.edu` and `wahid@udel.edu` as project members with the developer role.

5. Clone your fork locally either via command line or by using your preferred IDE.

6. Update the badge URLs in README.md by replacing `clause` with `<NAMESPACE>` and `471c` with `<PROJECT>`

## Install project dependencies

```bash
cd <PROJECT>
uv sync --all-extras --all-packages
uv run pre-commit run --all-files
```

## Verify setup

```bash
uv run pytest
```

# Code Formatting & Style Requirements

To keep the codebase consistent and maintainable, please follow these formatting rules:

## Python Style

We use ruff (via UV) to enforce code style and linting rules: See pyproject.toml for the configuration. A pre-commit hook runs ruff automatically and will prevent commits of poor quality code. You can have ruff automatically fix issues by running the following from the project root.

```bash
uv run ruff fix .
```

However, if you're using VS Code, you should install the project's recommended extensions. If you do so, the provided configuration in .vscode/settings.json will reformat on save.

### Type Annotations

Type annotations are required for all functions, methods, and class attributes wherever feasible. They help with:

- Static type checking. We are using pylance.

- Improved code readability and maintainability

- Better editor/IDE support (autocomplete, linting, refactoring)

# Testing & Coverage

All contributions must be tested and tests must be executable via `uv run pytest`. Code coverage is reported and 100% branch coverage is expected. Pragmas may be used to work around tool limitations (e.g., `#pragma: no branch` with `match`) but other adjustments should onlt be used as a last effort and must be clearly explained.

Note, that a high-level of coverage should be viewed as a sanity check (low coverage is a bad sign but don't assume high coverage means correctness). Compiler passes are complex and require careful test design to achieve confidence in their correctness.

The CI pipeline expects all tests to pass. Use the appropriate annotation to either skip problematic tests (`@pytest.mark.skip`) or to indicate that the corresponding feature has not been implemented (`@pytest.mark.xfail`).
