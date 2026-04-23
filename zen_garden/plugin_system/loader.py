"""Plugin loader.

Import plugin modules from a package and update their config dictionaries
according to the user's configuration (as defined in config.json).
"""

import importlib
from types import ModuleType

from zen_garden.plugin_system.events import EventPublisher


def register_plugins(
    plugins_config: dict[str, dict], source_package: str = "zen_garden.plugins"
) -> dict[str, ModuleType]:
    """Import plugin modules and apply provided configurations.

    Upon import all callables are registered to their respective events. The plugin's
    `config` dictionary is updated with the provided configuration.

    Args:
        plugins_config (dict[str, dict]): Mapping of plugin name to config dict.
        source_package (str): Root package where plugin packages live.

    Returns:
        dict[str, ModuleType]: Mapping of plugin name to the imported module.

    Example:

        >>> plugins_config = {"ExamplePlugin": {"param1": "value1", "param2": "value2"}}
        >>> register_plugins(plugins_config)
    """
    output = {}
    for plugin, config in plugins_config.items():
        output[plugin] = importlib.import_module(
            name=f"{source_package}.{plugin}.plugin"
        )
        output[plugin].config.update(config)
    return output


def deregister_plugins():
    EventPublisher.deregister_all()
