"""Structural checks for the dataset examples in ``docs/dataset_examples``.

The dataset examples themselves are only run on a schedule (see
``pytest_for_datasets_with_conda.yml``). These tests are cheap and verify that
the examples are still discovered and that their configuration files load,
without building or solving any model.
"""

import importlib.util
import re
from pathlib import Path

import pytest

from zen_garden.config import Config

DATASET_EXAMPLES_DIR = (
    Path(__file__).parents[3] / "docs" / "dataset_examples"
).resolve()


def _load_run_datasets():
    """Import ``docs/dataset_examples/run_datasets.py`` as a module."""
    spec = importlib.util.spec_from_file_location(
        "run_datasets", DATASET_EXAMPLES_DIR / "run_datasets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _numbered_example_dirs():
    """All folders following the ``<number>_<name>`` naming convention."""
    pattern = re.compile(r"^\d+_.+$")
    return sorted(
        path
        for path in DATASET_EXAMPLES_DIR.iterdir()
        if path.is_dir() and pattern.match(path.name)
    )


def test_all_dataset_examples_are_discovered():
    """``run_datasets.py`` finds every numbered dataset example folder."""
    found = {Path(p).name for p in _load_run_datasets().find_dataset_files()}
    expected = {path.name for path in _numbered_example_dirs()}

    assert expected, f"No dataset examples found in {DATASET_EXAMPLES_DIR}"
    assert found == expected, (
        f"Dataset examples not discovered by run_datasets.py: "
        f"{sorted(expected - found)}"
    )


@pytest.mark.parametrize(
    "dataset_dir", _numbered_example_dirs(), ids=lambda path: path.name
)
def test_dataset_example_config_is_valid(dataset_dir):
    """Main config and the dataset's system file load and validate."""
    config = Config.from_file(
        DATASET_EXAMPLES_DIR / "config.yaml", dataset_path=dataset_dir
    )
    config.validate_configurations()

    assert Path(config.analysis.dataset) == dataset_dir
