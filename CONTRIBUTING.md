# Contributing

Contribute against `main` of [Abel-ai-lab/Simons](https://github.com/Abel-ai-lab/Simons).

## Setup

On a CPU-only machine, install the CPU build of PyTorch:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`. Then:

```shell
git clone https://github.com/Abel-ai-lab/Simons && cd Simons
pip install -e ".[dev]"
```

The tests need no GPU; they run with either PyTorch build.

## Tests and lint

```shell
pytest
uvx ruff@0.14.3 check .   # or: pip install ruff==0.14.3 && ruff check .
```

CI runs tests and lint on Python 3.10 and 3.13. Tests run on CPU without network access,
using generated data and `tests/fixtures`.

## Code style

- Small modules, plain functions and dataclasses. The code should read on its own; add a short prose
  docstring to public functions that need one.
- Explicit input checks that raise a `ValueError` with a clear message.
- Line length 100 and the lint rules in `pyproject.toml`. The code is not auto-formatted, so
  match the layout of the code around your change.

## Forecasts and training

Keep forecasts, training losses and weights identical. For changes that alter outputs, document
the reason and measure the effect, such as benchmark scores before and after. Put new training
features behind a config option whose default leaves the recipe unchanged.
