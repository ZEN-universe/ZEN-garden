"""Unit tests for the plugin loader.

Unit tests for `register_plugins` ensuring modules are imported and configs passed.
"""

import pytest

from zen_garden.plugin_system.events import EventPublisher
from zen_garden.plugin_system.loader import deregister_plugins, register_plugins


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

    def test_import_selected_plugin_import_corresponding_module(self):
        """Selected plugin is imported from the given package.

        Ensures `register_plugins` returns the imported plugin module.
        """
        # Arrange
        plugins = {"fake_plugin": {}}

        # Act
        result = register_plugins(plugins, source_package="tests.unit_tests")
        from tests.unit_tests.fake_plugin import plugin

        # Assert
        assert result["fake_plugin"] == plugin

    def test_pass_config_to_selected_plugins(self):
        """Plugin configuration is passed to the plugin module.

        Ensures the plugin's `config` attribute equals the provided dict.
        """
        # Arrange
        plugins = {"fake_plugin": {"any_parameter": "any_value"}}

        # Act
        register_plugins(plugins, source_package="tests.unit_tests")
        from tests.unit_tests.fake_plugin import plugin

        # Assert
        assert plugin.config == plugins["fake_plugin"]

    def test_deregister_all_plugins(self):
        """Deregister all plugins.

        Ensures that all registered observers are removed after calling
        `EventPublisher.deregister_all`.
        """
        plugins = {"fake_plugin": {}}
        register_plugins(plugins, source_package="tests.unit_tests")

        deregister_plugins()

        # Assert
        assert len(EventPublisher.observers()) == 0
