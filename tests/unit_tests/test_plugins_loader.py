"""Unit tests for the plugin loader.

Unit tests for `register_plugins` ensuring modules are imported and configs passed.
"""

from zen_garden.plugin_system.loader import register_plugins


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
