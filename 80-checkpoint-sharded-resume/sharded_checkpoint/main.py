"""Load ``code/main.py`` as ``sharded_checkpoint.main``.

The spec keeps the implementation in ``code/main.py``.  Importing it through a
top-level ``code`` package would shadow the stdlib module of the same name, so
this loader executes the file with this module's name and re-exports its
symbols.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_SOURCE = Path(__file__).resolve().parents[1] / "code" / "main.py"


def _load_impl() -> ModuleType:
    spec = importlib.util.spec_from_file_location(__name__, _SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load implementation from {_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()
globals().update(
    {
        name: value
        for name, value in vars(_impl).items()
        if not name.startswith("__")
    }
)
