"""
Pytest configuration.

Each algorithm folder has its own ``src`` package, and several folders contain
modules with identical file names (e.g. ``vectorized_implementation.py``).
To avoid import collisions we register a unique import alias for every
``src`` directory (e.g. ``linear_regression_src``), so tests can do:

    from linear_regression_src.vectorized_implementation import ...
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALGOS_DIR = ROOT / "ml-algorithms-deep-dive"


def _register_src(alias: str, src_dir: Path) -> None:
    """Register ``src_dir`` as an importable package under ``alias``."""
    init_file = src_dir / "__init__.py"
    if not init_file.exists():
        init_file.touch()  # make it a regular package
    spec = importlib.util.spec_from_file_location(
        alias, str(src_dir / "__init__.py"), submodule_search_locations=[str(src_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)


_ALIASES = {
    "linear_regression_src": "linear-regression",
    "logistic_regression_src": "logistic-regression",
    "decision_trees_src": "decision-trees",
    "kmeans_src": "k-means",
    "random_forest_src": "random-forest",
    "neural_network_src": "neural-network",
    "pca_src": "pca",
}

for _alias, _folder in _ALIASES.items():
    _src = ALGOS_DIR / _folder / "src"
    if _src.is_dir():
        _register_src(_alias, _src)
