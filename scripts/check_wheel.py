"""Check a runtime-only wheel installation using ``python -I`` outside the repo."""

from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

import rfphate


def main():
    repository = Path(__file__).resolve().parents[1]
    installed_module = Path(rfphate.__file__).resolve()
    assert sys.flags.isolated, "Run this check with python -I"
    assert not installed_module.is_relative_to(repository), installed_module
    for package in ("pytest", "setuptools", "seaborn", "plotly", "nbformat", "ipykernel"):
        assert find_spec(package) is None, f"Unexpected optional package: {package}"
    for name in ("iris", "titanic", "auto-mpg", "artificial_tree"):
        data = rfphate.load_data(name)
        assert data.shape[0] > 0 and data.shape[1] > 1, name
        print(f"Loaded packaged dataset {name}: {data.shape}")
    print(f"Testing installed wheel: {installed_module}")
    suite = unittest.defaultTestLoader.discover(
        str(repository / "tests"), pattern="test_forestgeom_integration.py"
    )
    assert suite.countTestCases() == 3, "Original integration suite was not discovered"
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
