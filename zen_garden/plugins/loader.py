"""
This module only functionality is to import the modules from the plugins package
and update their config dictionaries based on the user specifications from
config.json.
"""

import importlib
from types import ModuleType


def register_plugins(
    plugins_config: dict[str, dict], source_package: str = "zen_garden.plugins"
) -> dict[str, ModuleType]:
    output = {}
    for plugin, config in plugins_config.items():
        output[plugin] = importlib.import_module(
            name=f"{source_package}.{plugin}.plugin"
        )
        output[plugin].config.update(config)
    return output