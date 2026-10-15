"""Unit tests for entry-point based plugin discovery and loading."""

from importlib.metadata import EntryPoint

import pytest

from zen_garden.config import Config
from zen_garden.plugin_system.events import EventPublisher
from zen_garden.plugin_system.loader import (
    ENTRY_POINT_GROUP,
    _get_installed_plugins,
    deregister_plugins,
    register_plugins,
)

FAKE_PLUGIN_ENTRY_POINT = EntryPoint(
    name="fake_plugin",
    value="tests.unit_tests.plugins.fixtures.fake_plugin.plugin",
    group=ENTRY_POINT_GROUP,
)


@pytest.fixture(scope="function", autouse=True)
def cleanup(request: pytest.FixtureRequest):
    """
    Pytest fixture to clean up registered observers after each test.
    """
    request.addfinalizer(EventPublisher.deregister_all)


class TestPluginsLoader:
    """Tests for the plugin loader.

    Verifies plugin import and config passing behavior of `register_plugins`.
    """

    def test_loads_selected_plugin_from_entry_point(self, monkeypatch):
        """The configured plugin is loaded from its installed entry point."""
        # Arrange
        monkeypatch.setattr(
            "zen_garden.plugin_system.loader.entry_points",
            lambda *, group: [FAKE_PLUGIN_ENTRY_POINT],
        )
        config = Config(plugins={"fake_plugin": {}})

        # Act
        result = register_plugins(config)
        from tests.unit_tests.plugins.fixtures.fake_plugin import plugin

        # Assert
        assert result == {"fake_plugin": plugin}

    def test_validates_plugin_config_and_adds_defaults(self, monkeypatch):
        """Plugin configuration is validated and written back to root config."""
        # Arrange
        monkeypatch.setattr(
            "zen_garden.plugin_system.loader.entry_points",
            lambda *, group: [FAKE_PLUGIN_ENTRY_POINT],
        )
        config = Config(plugins={"fake_plugin": {"any_parameter": "any_value"}})

        # Act
        register_plugins(config)

        # Assert
        assert config.plugins["fake_plugin"] == {
            "any_parameter": "any_value",
            "default_parameter": "default_value",
        }

    def test_reports_configured_plugin_that_is_not_installed(self, monkeypatch):
        """A useful error is raised when no matching entry point exists."""
        monkeypatch.setattr(
            "zen_garden.plugin_system.loader.entry_points",
            lambda *, group: [],
        )
        config = Config(plugins={"missing_plugin": {}})

        with pytest.raises(ModuleNotFoundError, match="missing_plugin.*not installed"):
            register_plugins(config)

    def test_rejects_duplicate_installed_plugin_names(self, monkeypatch):
        """Ambiguous entry points with the same name are rejected."""
        duplicate = EntryPoint(
            name="fake_plugin",
            value="another_package.plugin",
            group=ENTRY_POINT_GROUP,
        )
        monkeypatch.setattr(
            "zen_garden.plugin_system.loader.entry_points",
            lambda *, group: [FAKE_PLUGIN_ENTRY_POINT, duplicate],
        )

        with pytest.raises(RuntimeError, match="Multiple plugins.*fake_plugin"):
            _get_installed_plugins()

    def test_deregister_all_plugins(self, monkeypatch):
        """Deregister all plugins.

        Ensures that all registered observers are removed after calling
        `EventPublisher.deregister_all`.
        """
        monkeypatch.setattr(
            "zen_garden.plugin_system.loader.entry_points",
            lambda *, group: [FAKE_PLUGIN_ENTRY_POINT],
        )
        config = Config(plugins={"fake_plugin": {}})
        register_plugins(config)

        deregister_plugins()

        # Assert
        assert len(EventPublisher.observers()) == 0
