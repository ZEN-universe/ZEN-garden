"""Unit tests for Config.from_file loading from fixture files."""

import json
from pathlib import Path

import pytest

from zen_garden.config import Config


@pytest.fixture
def config_json_path() -> Path:
    """Path to the JSON config fixture file."""
    return Path(__file__).parent / "fixtures" / "config.json"


@pytest.fixture
def config_yaml_path() -> Path:
    """Path to the YAML config fixture file."""
    return Path(__file__).parent / "fixtures" / "config.yaml"


@pytest.fixture
def config_invalid_misspelled_path() -> Path:
    """Path to a fixture with a misspelled config key."""
    return Path(__file__).parent / "fixtures" / "config_invalid_misspelled.json"


class TestConfig:
    """Tests for loading config data from JSON and YAML files."""

    def test_threads_override_existing_key(self, config_json_path):
        """A threads override replaces the existing option, keeping its key."""
        config = Config.from_file(config_json_path, threads=4)

        assert config.solver.solver_options["Threads"] == 4
        assert "threads" not in config.solver.solver_options

    @pytest.mark.parametrize(
        ("solver", "expected_key"), [("gurobi", "Threads"), ("highs", "threads")]
    )
    def test_threads_override_adds_key_by_solver(
        self, config_json_path, tmp_path, solver, expected_key
    ):
        """Without an existing thread option, the key name follows the solver."""
        data = json.loads(config_json_path.read_text())
        data["solver"]["name"] = solver
        data["solver"]["solver_options"].pop("Threads")
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data))

        config = Config.from_file(path, threads=2)

        assert config.solver.solver_options[expected_key] == 2

    def test_threads_not_given_keeps_config_value(self, config_json_path):
        """Without a threads override the config value is untouched."""
        config = Config.from_file(config_json_path)

        assert config.solver.solver_options["Threads"] == 128

    def test_load_config_from_json_fixture(self, config_json_path):
        """Config.from_file loads provided JSON fixture correctly."""
        config = Config.from_file(config_json_path)

        assert isinstance(config, Config)
        assert config.analysis.dataset == "Test"
        assert config.analysis.time_series_aggregation.solver == "gurobi"
        assert config.solver.name == "gurobi"
        assert config.solver.solver_options == {
            "Method": 2,
            "NodeMethod": 2,
            "BarHomogeneous": 1,
            "DualReductions": 0,
            "Threads": 128,
            "Crossover": 0,
            "ScaleFlag": 2,
            "BarOrder": 0,
            "BarConvTol": 1e-6,
        }
        assert config.solver.save_duals is True
        assert config.solver.use_scaling is True
        assert config.solver.run_diagnostics is True
        assert config.solver.scaling_include_rhs is True
        assert config.solver.selected_saved_duals == []

    def test_load_config_from_yaml_fixture(self, config_yaml_path):
        """Config.from_file loads provided YAML fixture correctly."""
        config = Config.from_file(config_yaml_path)

        assert isinstance(config, Config)
        assert config.analysis.dataset == "Test"
        assert config.analysis.time_series_aggregation.solver == "gurobi"
        assert config.solver.name == "gurobi"
        assert config.solver.solver_options == {
            "Method": 2,
            "NodeMethod": 2,
            "BarHomogeneous": 1,
            "DualReductions": 0,
            "Threads": 128,
            "Crossover": 0,
            "ScaleFlag": 2,
            "BarOrder": 0,
            "BarConvTol": 1e-6,
        }
        assert config.solver.save_duals is True
        assert config.solver.use_scaling is True
        assert config.solver.run_diagnostics is True
        assert config.solver.scaling_include_rhs is True
        assert config.solver.selected_saved_duals == []

    def test_misspelled_entry_raises_validation_error(
        self, config_invalid_misspelled_path
    ):
        """Misspelled keys should fail strict schema validation."""
        with pytest.raises(ValueError, match="Failed to validate configuration"):
            Config.from_file(config_invalid_misspelled_path)
