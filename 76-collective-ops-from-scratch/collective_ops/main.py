"""Load ``code/main.py`` under a name that does not shadow the stdlib.

Python ships a top-level ``code`` module (used by ``pdb`` and ``codeop``).
Putting a ``code`` package on ``sys.path`` makes pytest's own ``import pdb``
fail, so tests and spawned workers import this module instead. The spec's
implementation remains in ``code/main.py`` and is executed here as
``collective_ops.main``; spawned functions pickle under that name.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_SOURCE = Path(__file__).resolve().parents[1] / "code" / "main.py"


def _load_impl() -> ModuleType:
    spec = importlib.util.spec_from_file_location(__name__, _SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load collective implementation from {_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()
globals().update(
    {name: value for name, value in vars(_impl).items() if not name.startswith("__")}
)
