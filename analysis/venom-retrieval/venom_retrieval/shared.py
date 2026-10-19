"""Import the repository's existing shared modules from `notebooks/`.

`notebooks/` is not a package, so the modules in it cannot be imported by name from here.
This loads the two this benchmark reuses straight from their files, which keeps one copy
of each:

  cys_framework_utils   cysteine framework strings, loop lengths and the framework regex,
                        shared with the cys-frameworks Nextflow pipeline.
  hp_conservation_utils ALPHABET_CLUSTERS, the residue classes of every reduced alphabet
                        kmerseek has.

Loading by path rather than copying the functions: the framework regex is the falsification
test for the whole cys-frameworks line of work, and two copies of it that disagree would
produce different numbers rather than an error.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: The repository root, four levels up from this file:
#: <repo>/analysis/venom-retrieval/venom_retrieval/shared.py
REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOKS = REPO_ROOT / "notebooks"


def _load(name: str) -> ModuleType:
    """Import `notebooks/<name>.py` under `name`, reusing it if already imported."""
    if name in sys.modules:
        return sys.modules[name]
    path = NOTEBOOKS / f"{name}.py"
    if not path.exists():
        raise ModuleNotFoundError(
            f"{path} is missing. This benchmark reuses it rather than keeping its own "
            f"copy; check out a branch that has it (olgabot/cys-frameworks or later)."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cys_framework_utils = _load("cys_framework_utils")
hp_conservation_utils = _load("hp_conservation_utils")
