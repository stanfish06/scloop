import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _ensure_packages() -> None:
    if "scloop" not in sys.modules:
        pkg = types.ModuleType("scloop")
        pkg.__path__ = [str(SRC / "scloop")]
        sys.modules["scloop"] = pkg
    if "scloop.data" not in sys.modules:
        subpkg = types.ModuleType("scloop.data")
        subpkg.__path__ = [str(SRC / "scloop" / "data")]
        sys.modules["scloop.data"] = subpkg


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def scloop_utils():
    _ensure_packages()
    _load_module("scloop.data.types", SRC / "scloop" / "data" / "types.py")
    return _load_module("scloop.data.utils", SRC / "scloop" / "data" / "utils.py")
