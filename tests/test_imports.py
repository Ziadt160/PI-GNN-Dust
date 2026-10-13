"""Every library module imports, and every experiment script at least compiles.

Experiment scripts are compiled, never imported: many of them open the dataset
at module level, and CI has no data. Compiling still catches the syntax errors
a move or a bulk edit leaves behind.
"""
import importlib
import pathlib
import py_compile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _module_name(path):
    parts = path.relative_to(REPO_ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


SRC_MODULES = sorted(_module_name(p) for p in (REPO_ROOT / "src").rglob("*.py"))
EXPERIMENT_SCRIPTS = sorted(
    p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "experiments").rglob("*.py"))


@pytest.mark.parametrize("module", SRC_MODULES)
def test_src_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("script", EXPERIMENT_SCRIPTS)
def test_experiment_script_compiles(script, tmp_path):
    # cfile keeps the bytecode out of the repository's __pycache__ folders.
    py_compile.compile(str(REPO_ROOT / script), cfile=str(tmp_path / "out.pyc"),
                       doraise=True)
